# 大模型加载 Trick

- [大模型加载 Trick](#大模型加载-trick)
  - [一、前言](#一前言)
  - [二、问题分析](#二问题分析)
  - [三、问题解决](#三问题解决)
    - [3.1 使用torch.load(mmap=True)](#31-使用torchloadmmaptrue)
    - [3.2 使用 torch.device('meta')](#32-使用-torchdevicemeta)
  - [四、实验结果](#四实验结果)
    - [4.1 加速前](#41-加速前)
    - [4.2 加速后](#42-加速后)
  - [致谢](#致谢)

## 一、前言

在深度学习领域，大模型的训练和推理通常需要消耗大量的计算和内存。如何高效地加载和使用大模型是一个相当关键的问题。在这篇博客中，我将分享一些关于更快加载大模型和减少内存的技巧.

## 二、问题分析

假设现在我们有一个236B 超大模型的原始权重的 checkpoint.pth 文件, 比如 DeepSeek Chat V2, 以BF16 格式存储, 一个标准的加载流程如下

> 注：DeepSeek Chat V2 https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat

```s
import torch

state_dict = torch.load(checkpoint_file)
my_model = BigModelClass(...)
my_model.load_state_dict(state_dict)
```

在这段代码的中, my_model = BigModelClass(...) 会初始化一个模型, torch.load(checkpoint_file)函数会将模型权重从磁盘加载到内存中。然后，my_model.load_state_dict(state_dict)函数会将权重从内存加载到模型的参数中。这两个步骤都可能会消耗大量的时间和内存。理想情况下, 一个236B BF16格式的模型需要占据 472GB 的内存, 上面的代码会有两个模型副本, 这意味着峰值需要944GB 内存, 接近1T ,这是非常夸张的也是不可接受的.

我们用一段简单的代码来验证上面的推断, 首先初始化一个 1B size 的模型并存下来,

```s
import torch

def count_parameters(model):
    total_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e9 

def model_memory_size_in_megabytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  

    bytes_in_gb = 1024 * 1024 * 1024 
    return param_size / bytes_in_gb

class BigModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(size, size) for i in range(10)])

    def forward(self, x):
        return self.linears(x)

size = 10000
model = BigModel(size)

# 打印模型的参数量
print(f'The model has {count_parameters(model):,} B trainable parameters')
print(f"The model's memory size is approximately {model_memory_size_in_megabytes(model):.2f} GB.")
torch.save(model.state_dict(), 'checkpoint.pth')
```

> output
```s
The model has 1.0001 B trainable parameters
The model's memory size is approximately 3.73 GB.
```

然后 按照上面的方式加载模型, 并统计cpu 内存占用, torch 默认是FP32 格式, 1B模型占用约 4GB 内存(实际为3.73GB左右), 下面代码验证后基本符合预期

```s
def print_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print(f'memory: {memory_use:.2f} GB')
    print('CPU percent:', psutil.cpu_percent())

print('Before Load the state_dict:')
print_usage()
```

> output
```s
Before Load the state_dict:
memory: 0.34 GB
CPU percent: 8.5
```

```s
start_time = time.time()
state_dict = torch.load('checkpoint.pth')
print(f'Loading the state_dict took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict:')
print_usage()
```

> output
```s
Loading the state_dict took 2.09 seconds
After Load the state_dict:
memory: 4.06 GB
CPU percent: 7.0
```

4.06 - 0.34 = 3.72基本一致

```s
start_time = time.time()
model = BigModel(size)
print(f'Init the model took {time.time() - start_time:.2f} seconds')
print('After Init the model:')
print_usage()
```

> output
```s
Init the model took 7.23 seconds
After Init the model:
memory: 7.79 GB
CPU percent: 7.6
```

7.79 - 4.06 = 3.73 基本一致

```s
start_time = time.time()
model.load_state_dict(state_dict)
print(f'Loading the state_dict to model took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict to model:')
print_usage()
```

> output
```s
Loading the state_dict to model took 2.63 seconds
After Load the state_dict to model:
memory: 7.79 GB
CPU percent: 16.4
```

## 三、问题解决

分析清楚在加载和初始化环节中各个流程的开销, 我们来看看可以如何加速每个过程.

### 3.1 使用torch.load(mmap=True)

首先，让我们考虑一下当我们使用 加载检查点时会发生什么torch.load。当我们使用 保存检查点时torch.save，张量存储会使用保存它们的设备进行标记。使用torch.load，张量存储将加载到它们标记的设备（除非使用标志覆盖此行为 map_location）。为了便于解释，我们假设张量保存在 CPU 上。这意味着在第一行，所有张量存储都将加载到 CPU RAM 中，这在以下情况下可能是不可行的：

- CPU RAM 小于检查点的大小。
- 等待整个检查点加载到 RAM 中，然后再执行某些按张量处理等操作。

```s
start_time = time.time()
state_dict = torch.load('checkpoint.pth')
end_time = time.time()
print(f"loading time without mmap={end_time - start_time}")
print_usage()
```

> output
```s
loading time without mmap=2.0737619400024414
memory: 4.06 GB
CPU percent: 8.7
```

torch.load中的mmap参数图解决上述两个问题。顾名思义，mmap关键字参数 totorch.load 使用mmap 调用 ，将磁盘上的文件映射到虚拟内存，并让操作系统自动处理到物理内存的加载和卸载。当这个标志被传递时，张量存储将被内存映射。

```s
start_time = time.time()
state_dict = torch.load('checkpoint.pth', mmap=True)
end_time = time.time()
print(f"loading time with mmap={end_time - start_time}")
print_usage()
```

> output
```s
loading time with mmap=0.003424406051635742
memory: 0.34 GB
```

通过上面对比,我们可以发现 使用mmap可以加速模型加载并减少内存占用, 对于236B的模型, 我们实际上并不需要 1TB的 CPU内存来完成转换

### 3.2 使用 torch.device('meta')

当模型size 巨大时, 模型初始化也需要巨大时间, 我们扩大一下模型size到25B, 初始化一个模型就需要接近3分钟.

```s
size = 50000
start_time = time.time()
model = BigModel(size)
end_time = time.time()
print(f"init time={end_time - start_time}")
print(f'The model has {count_parameters(model):,} B trainable parameters')
print(f"The model's memory size is approximately {model_memory_size_in_megabytes(model):.2f} GB.")
```

> output
```s
init time=184.56671452522278
The model has 25.0005 B trainable parameters
The model's memory size is approximately 93.13 GB.
```

但在load 模型时, 初始化这一步是多余的, 我们实际上只需要知道模型的所有 key 和 对应的 shape,

这个时候, torch.device('meta') 这个 上下文就可以发挥作用了, torch.device() 上下文管理器确保工厂调用将像它们被传递了指定的"device"作为参数一样执行。在 torch.device('meta') 上的张量不携带数据。然而，它们具有张量所具有的所有其他元数据，例如.size()、.stride()、.requires_grad等。

```s
with torch.device('meta'):
   model = BigModel(size)
model.load_state_dict(state_dict, assign=True)

for n, p in model.named_parameters():
    assert p.device.type != "meta", f"{n} has not been loaded!"
```

注意, 在使用 torch.device('meta')后, 我们需要加上 assign=True参数来让参数被加载. 最后一段代码可以check 所有参数被正确加载了, 加载后的参数的 device应该不再是 meta 了.

## 四、实验结果

最后, 我们直接上一个100B size大小的大模型来对比, 是否使用 torch.load(mmap=True) 和torch.device('meta') 速度差别.

```s
size = 100000
model = BigModel(size)

# 打印模型的参数量
print(f'The model has {count_parameters(model):,} B trainable parameters')
print(f"The model's memory size is approximately {model_memory_size_in_megabytes(model):.2f} GB.")
```

> output
```s
torch.save(model.state_dict(), 'checkpoint.pth')
The model has 100.001 B trainable parameters
The model's memory size is approximately 186.27 GB.
```

### 4.1 加速前

```s
start_time = time.time()
state_dict = torch.load('checkpoint.pth')
print(f'Loading the state_dict took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict:')
print_usage()

start_time = time.time()
model = BigModel(size)
print(f'Init the model took {time.time() - start_time:.2f} seconds')
print('After Init the model:')
print_usage()

start_time = time.time()
model.load_state_dict(state_dict)
print(f'Loading the state_dict to model took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict to model:')
print_usage()

start_time = time.time()
input = torch.randn(1, size)
output = model(input)
print(output)
print(f'One time forward {time.time() - start_time:.2f} seconds')
print_usage()
```

> output
```s
Before Load the state_dict:
memory: 0.34 GB
CPU percent: 9.1
Loading the state_dict took 852.06 seconds
After Load the state_dict:
memory: 372.87 GB
CPU percent: 5.0
Init the model took 518.15 seconds
After Init the model:
memory: 745.41 GB
CPU percent: 4.9
Loading the state_dict to model took 125.63 seconds
After Load the state_dict to model:
memory: 745.41 GB
CPU percent: 11.7
tensor([[-0.0015, 0.0017, -0.0009, ..., -0.0036, 0.0041, 0.0052]],
grad_fn=\)
One time forward 6.95 seconds
memory: 745.42 GB
CPU percent: 11.4
```

### 4.2 加速后

```s
start_time = time.time()
state_dict = torch.load('checkpoint.pth', mmap=True)
print(f'Loading the state_dict took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict:')
print_usage()

start_time = time.time()
with torch.device('meta'):
  model = BigModel(size)
print(f'Init the model took {time.time() - start_time:.2f} seconds')
print('After Init the model:')
print_usage()

start_time = time.time()
model.load_state_dict(state_dict, assign=True)
print(f'Loading the state_dict to model took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict to model:')
print_usage()

for i in range(2):
    start_time = time.time()
    input = torch.randn(1, size)
    output = model(input)
    print(output)
    print(f'One time forward {time.time() - start_time:.2f} seconds')
    print_usage()
```

> output
```s
Before Load the state_dict:
memory: 0.34 GB
CPU percent: 9.1
Loading the state_dict took 0.11 seconds
After Load the state_dict:
memory: 0.34 GB
CPU percent: 6.1
Init the model took 0.00 seconds
After Init the model:
memory: 0.34 GB
CPU percent: 4.3
Loading the state_dict to model took 0.00 seconds
After Load the state_dict to model:
memory: 0.34 GB
CPU percent: 10.0
tensor([[ 0.0080, -0.0017, -0.0027, ..., -0.0011, 0.0097, -0.0048]],
grad_fn=\)
One time forward 48.37 seconds
memory: 372.85 GB
CPU percent: 5.2
tensor([[ 0.0038, 0.0014, -0.0076, ..., -0.0016, 0.0004, -0.0018]],
grad_fn=\)
One time forward 3.28 seconds
memory: 372.86 GB
CPU percent: 13.4
```

通过上面的对比, 加速前100B模型加载时间为

852.06 + 518.15 + 125.63 = 1495(s) = 25 (min)

而使用 mmap + meta device 加载几乎没有时间开销, 只有模型真正运行时才会从硬盘拷贝权重到CPU RAM

## 致谢

- 超大模型加载转换Trick https://zhuanlan.zhihu.com/p/698950172

