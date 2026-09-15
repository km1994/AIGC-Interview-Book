# LLMs 其他 Trick

- [LLMs 其他 Trick](#llms-其他-trick)
  - [huggingface 下载不了模型问题？](#huggingface-下载不了模型问题)

## huggingface 下载不了模型问题？

- 方法一：在modelscope 下载你想要的模型

```s
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('damo/nlp_xlmr_named-entity-recognition_viet-ecommerce-title', cache_dir='path/to/local/dir', revision='v1.0.1')
```

- 方法二：[大语言模型下载站](https://aliendao.cn/)

HuggingFace.co资源下载网站，为AI开发者提供模型镜像加速服务，通过下载器可以达到10M/s的下载速度，解决大模型下载时间长、经常断线、需要反复重试等问题，实现镜像加速、断点续传、无人值守下载，

![](img/微信截图_20230919222949.png)




