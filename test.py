# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
import os
from tkitAutoMask import BertRandomMaskingScheme

from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
inputs = tokenizer("Hello, my dog is cute")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

# args, tokens, pad, mask_id=None,

# BertRandomMaskingScheme(args, tokens, pad, mask_id)
# tokenizer

print("inputs",inputs)
class ARG:
  def __init__(self):
    self.mask_ratio=0.4
args=ARG()
args.mask_ratio

# tokens=21128
mk=BertRandomMaskingScheme(args, tokenizer.vocab_size, tokenizer.pad_token_id ,tokenizer.mask_token_id)
print(dir(mk))


out = mk.mask(inputs['input_ids'])
print(out)



if __name__ == '__main__':
    pass
