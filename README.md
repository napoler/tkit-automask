# tkitAutoMask

自动构建掩码
加入多种动态掩码合集，上下三角和动态片段，以及默认的概率

-上三角，实现类似从左到右的预测，就是单向注意，用于续写。
- 片段，连续多个mask，更加适合解决补全。


未来尝试加入 模板预测掩码


```
pip install tkitAutoMask


```



```python
from tkitAutoMask import autoMask
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-2_H-128") 
# dir(tokenizer)
tomask = autoMask(
    # transformer,
    mask_token_id = tokenizer.mask_token_id,          # the token id reserved for masking
    pad_token_id = -100,           # the token id for padding
    mask_prob = 0.05,           # 仅仅是常规的掩码比例 masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = [tokenizer.cls_token_id,tokenizer.eos_token_id]  # other tokens to exclude from masking, include the [cls] and [sep] here
)


x=torch.ones(5,5)
for i in range(100):
  a,b=tomask(x)
  # a,b
  print(b)
 
```
labels：形状为[batch_size, seq_length] ，代表MLM任务的标签，注意这里对于原本未被遮盖的词设置为-100，被遮盖词才会有它们对应的id，和任务设置是反过来的。
例如，原始句子是I want to [MASK] an apple，这里我把单词eat给遮住了输入模型，对应的label设置为[-100, -100, -100, 【eat对应的id】, -100, -100]；
为什么要设置为-100而不是其他数？ 因为torch.nn.CrossEntropyLoss默认的ignore_index=-100，也就是说对于标签为100的类别输入不会计算loss。

```
tensor([[1., 1., 1., 0., 1.],
        [0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 1., 0., 0., 1.]])
tensor([[1., 1., 0., 0., 0.],
        [0., 1., 1., 0., 0.],
        [1., 0., 1., 1., 0.],
        [0., 1., 0., 1., 1.],
        [0., 0., 0., 0., 1.]])
tensor([[1., 1., 1., 0., 1.],
        [0., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1.],
        [1., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.]])
tensor([[0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0.]])
tensor([[1., 1., 1., 0., 0.],
        [0., 1., 1., 1., 0.],
        [1., 0., 1., 1., 1.],
        [0., 1., 0., 1., 1.],
        [0., 0., 0., 1., 1.]])
tensor([[0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0.]])
tensor([[0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 1.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 1.]])
tensor([[1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0.]])

```


其他测试

https://colab.research.google.com/drive/1CvkoJ1pZQDRWGPA-5IzJufvocBM-RVT2#scrollTo=UwkociF5ZF-d

详细参考

> dev.md


