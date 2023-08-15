import torch
from transformers import BertTokenizerFast
from main import *
from model.Bert_model import *
import pandas as pd


pthfile = 'out/bert_model_06_01_14_13/checkpoints-50/model.pth'
model = torch.load(pthfile, map_location=torch.device('cpu'))
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

## 测试1
text = [["我帅吗？"], ["你好"], ["今天天气怎么样？"]]
input = tokenizer(text, truncation=True, padding=True,  is_split_into_words=True, return_tensors="pt")
text_pred = model.predict(input)

