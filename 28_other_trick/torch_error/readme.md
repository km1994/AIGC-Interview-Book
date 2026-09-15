# pytorch 开发 坑/bug 梳理篇

## 一、F.dropout ( nn.functional.dropout ) 使用问题

- 问题描述：

```s
Class DropoutFC(nn.Module):
    def __init__(self):
        super(DropoutFC, self).__init__()
        self.fc = nn.Linear(100,20)

    def forward(self, input):
        out = self.fc(input)
        out = F.dropout(out, p=0.5)
        return out

Net = DropoutFC()
Net.train()

# train the Net
```

这段代码中的F.dropout实际上是没有任何用的

- 问题定位：

上述代码中的F.dropout实际上是没有任何用的, **因为它的training状态一直是默认值False. 由于F.dropout只是相当于引用的一个外部函数, 模型整体的training状态变化也不会引起F.dropout这个函数的training状态发生变化**. 所以, 此处的out = F.dropout(out) 就是 out = out.  

- 解决方法：

将模型整体的training状态参数传入dropout函数

```s
Class DropoutFC(nn.Module):
   def __init__(self):
       super(DropoutFC, self).__init__()
       self.fc = nn.Linear(100,20)

   def forward(self, input):
       out = self.fc(input)
       out = F.dropout(out, p=0.5, training=self.training)
       return out

Net = DropoutFC()
Net.train()
# train the Net
```

或者直接使用nn.Dropout() (nn.Dropout()实际上是对F.dropout的一个包装, 也将self.training传入了) 

```s
Class DropoutFC(nn.Module):
  def __init__(self):
      super(DropoutFC, self).__init__()
      self.fc = nn.Linear(100,20)
      self.dropout = nn.Dropout(p=0.5)

  def forward(self, input):
      out = self.fc(input)
      out = self.dropout(out)
      return out
Net = DropoutFC()
Net.train()

# train the Net
```

## 二、记录loss信息的时候直接使用了输出的Variable 导致 爆显问题

- 问题描述

```s
for data, label in trainloader:
    ......
    out = model(data)
    loss = criterion(out, label)
    loss_sum += loss     # <--- 这里
    ......
```

运行着就发现显存炸了

- 问题定位

随着每个batch显存消耗在不断增大，最终导致 显存炸了。

这是因为输出的loss的数据类型是Variable。而PyTorch的动态图机制就是通过Variable来构建图。主要是**使用Variable计算的时候，会记录下新产生的Variable的运算符号，在反向传播求导的时候进行使用**。

如果这里直接将loss加起来，系统会认为这里也是计算图的一部分，也就是说网络会一直延伸变大~那么消耗的显存也就越来越大~~

- 问题解决

```s
    loss += loss.detach()
```

使用loss += loss.detach()来获取不需要梯度回传的部分。

## 致谢

- PyTorch 有哪些坑/bug？ https://www.zhihu.com/question/67209417







