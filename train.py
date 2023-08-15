# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from tqdm import tqdm
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import coo_matrix
import numpy as np
import copy
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from scipy import sparse


# 训练模型
class model_Train_Test():
    def __init__(self, args, device, model, train_dataloader, test_dataloader, traindata_num, testdata_num):
        self.args = args
        self.device = device
        self.model = model

        # data setting
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataloader_origin = train_dataloader
        self.test_dataloader_origin = test_dataloader
        self.traindata_num = traindata_num
        self.testdata_num = testdata_num

        # kfold setting
        self.best_model = None
        self.best_acc = 0

        # model param setting
        self.PRINT_MODEL = False
        self.init_state = copy.deepcopy(self.model.state_dict())
        self.optimizer = AdamW(self.model.parameters(), self.args.lr)
        self.loss_fnt = nn.CrossEntropyLoss()

        # printing
        if self.PRINT_MODEL:
            for name, param in self.model.named_parameters():
                print(name + ' : ' + str(param))

    # 保存模型
    def save_model(self, model, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model, os.path.join(path, 'model.pth'))

    # 计算准确率
    def compute_metric(self, preds, labels):
        pred = np.argmax(preds, axis=-1)
        label = labels
        acc = accuracy_score(label, pred)
        p = precision_score(label, pred, average='macro')
        r = recall_score(label, pred, average='macro')
        f1 = f1_score(label, pred, average='macro')
        return p, acc, r, f1

    def save_best_model(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = copy.deepcopy(self.model.state_dict())

    # 模型训练
    def train(self):
        self.model.train()
        for epoch in range(self.args.num_train_epochs + 1):
            total_loss = 0
            iter_num = 0
            total_iter = len(self.train_dataloader)
            loop = tqdm(total = self.traindata_num, desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
            for batch in self.train_dataloader:
                
                inputs = batch
                for key in inputs.keys():
                    if len(inputs[key].shape) > 2:
                        inputs[key] = inputs[key].squeeze(1).to(self.device)

                result = self.model(**inputs)
                self.optimizer.zero_grad()
                result['loss'].backward()
                self.optimizer.step()
                total_loss += result['loss'].item()
                # # print loss
                # if (iter_num % 100 == 0):
                #     print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" %(epoch, iter_num, result['loss'].item(), iter_num / total_iter * 100))
                loop.update(self.args.batch_size)
                iter_num += 1

            if self.args.num_train_epochs == self.args.save_period:
                print('save model')
                checkpoints_dirname = "./out/" + self.args.model_name +"_model_"
                os.makedirs(checkpoints_dirname, exist_ok=True)
                self.save_model(self.model, checkpoints_dirname + '/checkpoints-{}/'.format(epoch))
            print("Epoch: %d, Average training loss: %.4f" % (epoch, total_loss / len(self.train_dataloader)))
        loop.close()
        

    # 模型测试
    def test(self):
        total_acc = []
        total_prec = []
        total_r = []
        total_f1 = []

        print('测试中...')
        self.model.eval()
        loop = tqdm(total= self.testdata_num)
        with torch.no_grad():
            for batch in self.test_dataloader:
                inputs = batch
                for key in inputs.keys():
                    if len(inputs[key].shape) > 2:
                        inputs[key] = inputs[key].squeeze(1).to(self.device)

                result = self.model(**inputs)
                
                label_predictions = result['logits'].detach().cpu().numpy()
                label_target = inputs['labels'].to('cpu').numpy()

                # 计算文本label准确率
                p, acc, r, f1 = self.compute_metric(label_predictions, label_target)
                total_prec.append(p)
                total_acc.append(acc)
                total_r.append(r)
                total_f1.append(f1)

                loop.update(self.args.batch_size)
            
            self.save_best_model(np.mean(total_acc))
            loop.close()

            # 呈现测试结果
            print('============================================================================================')
            print("测试Label:-----precision is {}----acc is {}---recall is {}---f1 is {}-----".format(np.around(np.mean(total_prec), decimals=3), 
                                                                                                      np.around(np.mean(total_acc), decimals=3),
                                                                                                      np.around(np.mean(total_r), decimals=3), 
                                                                                                      np.around(np.mean(total_f1), decimals=3)))
            print('============================================================================================')
            return np.mean(total_acc)

    def kfold_train_test(self):
        zero_result = []
        fintune_result = []
        self.best_acc = 0
        for k in range(self.args.kfold_num):
            self.model.load_state_dict(self.init_state)
            print('[{}-fold] training begin.'.format(k))
            self.train_dataloader = self.train_dataloader_origin[k]
            self.test_dataloader = self.test_dataloader_origin[k]

            # zeroshot_test
            zero_acc = self.test()
            zero_result.append(zero_acc)

            # finetune_test
            self.model.train()
            self.train()
            fintune_acc = self.test()
            fintune_result.append(fintune_acc)

        # best_test
        self.model.load_state_dict(self.best_model)
        self.test()
        best_result = self.best_acc

        # print result
        print(f'{self.args.kfold_num}-fold best score: {best_result}')
        print(f'{self.args.kfold_num}-fold fintune score: {np.mean(fintune_result)}')
        print(f'{self.args.kfold_num}-fold fintune score list: {fintune_result}')
        print(f'{self.args.kfold_num}-fold zero-shot score: {np.mean(zero_result)}')
        print(f'{self.args.kfold_num}-fold zero-shot score list: {zero_result}')

        # save model
        print('save model')
        checkpoints_dirname = "out/" + self.args.model_name +"_model"
        os.makedirs(checkpoints_dirname, exist_ok=True)
        self.save_model(self.model, checkpoints_dirname + '/checkpoints-best/')
