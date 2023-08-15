import os
import torch
import random
import numpy as np
from pathlib import Path
from dataloader import *
import argparse 
from train import model_Train_Test
from model.model import *
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    parser.add_argument("--tokenizer_name", type=str, default='bert-base-uncased')
    parser.add_argument("--polishedText_pth", type=str, default="./data_generation/PolishedText.csv")
    parser.add_argument("--generatedText_pth", type=str, default="./data_generation/GeneratedText.csv")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_period", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument('--load_model_path',type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kfold_flag", type=bool, default=True)
    parser.add_argument("--kfold_num",type=int,default=5)
    args = parser.parse_args()
    return args

def main(args):
    device = args.device
    tokenizer, trainDataloader, testDataloader, trainDataNums, testDataNums = init_data(args)
    
    print('data loaded ...')
    print('训练集数量: ', trainDataNums)
    print('测试集数量: ', testDataNums)
    print('kfold: ', args.kfold_flag)

    model = get_model(args.model_name)
    print('model loaded ...')
    trainer = model_Train_Test(args, device, model, trainDataloader, testDataloader, trainDataNums, testDataNums)
    print('trainer loaded ...')
    if args.kfold_flag:
        trainer.kfold_train_test()
    else:
        trainer.train()
        trainer.test()

if __name__=='__main__':
    args = parse_args()
    main(args)
    