# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
import os
from tkitAutoMask import BertRandomMaskingScheme
from types import SimpleNamespace
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MAX_LENGTH=512
# model = BertModel.from_pretrained('bert-base-chinese')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

text="""与MarkusOdenthal类似，我的列表中有一个非字符串类型。我通过将列转换为字符串，然后将其转换为列表，然后将其拆分为训练段和测试段来修复它。你会这么做吗与MarkusOdenthal类似，我的列表中有一个非字符串类型。我通过将列转换为字符串，然后将其转换为列表，然后将其拆分为训练段和测试段来修复它。你会这么做吗"""
# inputs = tokenizer([text]*5, padding="max_length", max_length=MAX_LENGTH, truncation=True)

inputs = tokenizer([text]*5, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

# args, tokens, pad, mask_id=None,

# BertRandomMaskingScheme(args, tokens, pad, mask_id)
# tokenizer

print("inputs",inputs)
# class ARG:
#   def __init__(self):
#     self.mask_ratio=0.15
args=SimpleNamespace(mask_ratio=0.15)
args.mask_ratio

# tokens=21128
mk=BertRandomMaskingScheme(args, tokenizer.vocab_size, tokenizer.pad_token_id ,tokenizer.mask_token_id)
# print(dir(mk))

out = mk.mask(inputs['input_ids'])
print(out)
# (array([  103,   100,   117,  5745, 13030,  8310, 12322,  8154,   102]), array([  101,     0,     0,  8422,     0,     0, 12322,     0,   102]), None)

print(torch.Tensor(out[0]),torch.Tensor(out[1]))



#
# mk= PairWithSpanMaskingScheme(args, tokenizer.vocab_size, tokenizer.pad_token_id ,tokenizer.mask_token_id)
# print(dir(mk))
#
# out = mk.mask(inputs['input_ids'])
# print(out)
# # (array([  103,   100,   117,  5745, 13030,  8310, 12322,  8154,   102]), array([  101,     0,     0,  8422,     0,     0, 12322,     0,   102]), None)
#
#













if __name__ == '__main__':
    pass
