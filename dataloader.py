import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import argparse
from transformers import BertTokenizerFast, RobertaTokenizerFast, AlbertTokenizerFast, GPT2TokenizerFast
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, GPT2Tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    parser.add_argument("--tokenizer_name", type=str, default='bert-base-uncased')
    parser.add_argument("--polishedText_pth", type=str, default="./data_generation/PolishedText.csv")
    parser.add_argument("--generatedText_pth", type=str, default="./data_generation/GeneratedText.csv")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_period", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument('--load_model_path',type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kfold_num",type=int,default=5)
    args = parser.parse_args()
    return args


# 自定义数据集类
class BaseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, index):
        batch_data = {}
        text = self.data['text'][index]
        label = self.data['label'][index]
        textEncoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, is_split_into_words=True, return_tensors='pt')
        batch_data.update(textEncoding)
        batch_data.update({'labels': torch.tensor(label, dtype=torch.long)})
        return batch_data
    
def load_tokenizer(args):
    if args.tokenizer_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise NameError
    return tokenizer

# 0 for polished, 1 for generated
def load_data(args):
    generatedTextPd = pd.read_csv(args.generatedText_pth)
    polishedTextPd = pd.read_csv(args.polishedText_pth)
    generatedTextList = generatedTextPd['GenetatedText'].tolist()
    polishedTextList = polishedTextPd['PolishedText'].tolist()

    totalText = []
    totalLabel = []
    for i in range(min(len(generatedTextList), len(polishedTextList))):
        totalText.append(generatedTextList[i])
        totalLabel.append(1)
        totalText.append(polishedTextList[i])
        totalLabel.append(0)

    assert len(totalText) == len(totalLabel)
    return {'text': totalText, 'label': totalLabel}

def split_train_test_data(inputData):
    totalText = inputData['text']
    totalLabel = inputData['label']
    
    trainDataText = totalText[:int(len(totalText) * 0.7)]    
    trainDataLabel = totalLabel[:int(len(totalLabel) * 0.7)]

    testDataText = totalText[int(len(totalText) * 0.7):]
    testDataLabel = totalLabel[int(len(totalLabel) * 0.7):]
    
    return [{'text': trainDataText, 'label': trainDataLabel}], [{'text': testDataText, 'label': testDataLabel}], len(trainDataText), len(testDataText)

def test_dataloader(dataloader):
    for batch in dataloader:
        print(batch)
    return batch

def split_train_test_data_kfold(args, inputData):
    totalText = inputData['text']
    totalLabel = inputData['label']
    len_data = len(totalText)
    fold_size = len_data // args.kfold_num
    trainDataList = []
    testDataList = []
    for k in range(args.kfold_num):
        train_data = {}
        train_data['text'] = totalText[:k * fold_size]+ totalText[(k+1) * fold_size:]
        train_data['label'] = totalLabel[:k * fold_size]+ totalLabel[(k+1) * fold_size:]
        test_data = {}
        test_data['text'] = totalText[k * fold_size:(k+1)*fold_size]
        test_data['label'] = totalLabel[k * fold_size:(k+1)*fold_size]

        trainDataList.append(train_data)
        testDataList.append(test_data)

    return trainDataList, testDataList, len(train_data['text']), len(test_data['text'])

def init_data(args):
    np.random.seed(args.seed)
    tokenizer = load_tokenizer(args)
    inputData = load_data(args)
    if args.kfold_flag:
        trainData, testData, trainDataNums, testDataNums = split_train_test_data_kfold(args, inputData)

        trainDataloaderList = []
        testDataloaderList = []
        for i in range(len(trainData)):
            trainDataset = BaseDataset(trainData[i], tokenizer,  max_length=args.max_length)
            testDataset = BaseDataset(testData[i], tokenizer,  max_length=args.max_length)
            trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
            testDataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)
            trainDataloaderList.append(trainDataloader)
            testDataloaderList.append(testDataloader)
        return tokenizer, trainDataloaderList, testDataloaderList, trainDataNums, testDataNums

    else:
        trainData, testData, trainDataNums, testDataNums = split_train_test_data(inputData)

        trainDataset = BaseDataset(trainData, tokenizer,  max_length=args.max_length)
        testDataset = BaseDataset(testData, tokenizer,  max_length=args.max_length)
        trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
        testDataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

        return tokenizer, trainDataloader, testDataloader, trainDataNums, testDataNums

if __name__ == '__main__':
    args = parse_args()
    tokenizer, trainDataloader, testDataloader, trainDataNums, testDataNums = init_data(args)