import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import argparse
from tqdm import tqdm
import backoff
import openai
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--result_pth", type=str, default="../data_generation/polish_news.csv")
parser.add_argument("--api_key", type=str, default="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")
parser.add_argument("--data_num", type=int, default=500)
args = parser.parse_args()

openai.api_key = args.api_key

print(openai.api_key)

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout), max_time=60)
def request_post(**kwargs):
    try: 
        response = openai.ChatCompletion.create(**kwargs)
    except:
        sleep(100)
    return response

def acquire(prompt):
    messages =[]
    system_msg = "You are a news editor now. \
                 Can you generate a news abstract based on the following news-related information? \
                 Simply return the abstract text. "
    
    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    params = {
        "model": "gpt-3.5-turbo",
        'messages': messages,
        'n':1,
        'temperature':0,
        'top_p':1.0,
        'frequency_penalty':0,
        'presence_penalty':0,
    }
    response = request_post(**params)
    reply = response["choices"][0]["message"]["content"]   
    return reply   

### 读取新闻信息
def load_mind_news():
    news_Info_df = pd.read_table('../data/MIND_small/MINDsmall_train/news.tsv',
                                header=None,
                                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                       'title_entities', 'abstract_entities'])
    return news_Info_df


def Gprompt_generation(news_info_df):
    categoryList = news_info_df['category'].tolist()
    subcategoryList = news_info_df['subcategory'].tolist()
    titleList = news_info_df['title'].tolist()
    abstractList = news_info_df['abstract'].tolist()
    GpromptList = []
    for i in range(news_info_df.shape[0]):
        # 过滤
        if str(abstractList[i]) == 'nan':
            continue
        if len(abstractList[i].split(' ')) < 50 :
            continue
        # prompt生成
        #  'Can you generate a news abstract of less than ' + str(len(abstractList[i].split(' '))) + ' words?'  + '\n' + \
        prompt = 'news title: ' + titleList[i] + '\n' + \
                 'news category: ' + categoryList[i] + '\n' + \
                 'news subcategory: ' + subcategoryList[i] + '\n' + \
                 'Can you generate a news abstract of less than ' + str(100) + ' words?'  + '\n' + \
                 'Please only return the generated text!'
        GpromptList.append(prompt)
    GpromptPd = pd.DataFrame()
    GpromptPd['Gprompt'] = GpromptList
    GpromptPd.to_csv('../data_generation/Gprompt.csv')
    return GpromptList

def Hprompt_generation(news_info_df):
    abstractList = news_info_df['abstract'].tolist()
    HpromptList = []
    for i in range(news_info_df.shape[0]):
        # 过滤
        if str(abstractList[i]) == 'nan':
            continue
        if len(abstractList[i].split(' ')) < 50 :
            continue
        # prompt生成
        # 'Can you polish follow news abstract of less than ' + str(len(abstractList[i].split(' '))) + ' words?'  + '\n' + \
        prompt = 'Can you polish follow news abstract of less than ' + str(100) + ' words?'  + '\n' + \
                 'Please only return the generated text!' + '\n' + \
                  abstractList[i]
        HpromptList.append(prompt)
    HpromptPd = pd.DataFrame()
    HpromptPd['Hprompt'] = HpromptList
    HpromptPd.to_csv('../data_generation/Hprompt.csv')
    return HpromptList

def extraction_polish_news_abstract(Hprompt, args):
    promptSelect = Hprompt[100:args.data_num]
    PolishedTextList = []
    count = 0
    try: 
        for query in promptSelect:
            print(count)
            print(query)
            PolishedText = acquire(query)
            PolishedTextList.append(PolishedText)
            print(PolishedText)
            print('=====')
            count += 1
    except:
        print('The server is overloaded or not ready yet')
    PolishedTextPd = pd.DataFrame()
    PolishedTextPd['PolishedText'] = PolishedTextList
    PolishedTextPd.to_csv('../data_generation/PolishedText_500.csv')
    return PolishedTextList

def extraction_origin_news_abstract(GPrompt, args):
    promptSelect = GPrompt[100:args.data_num]
    GenetatedTextList = []
    count = 0
    try: 
        for query in promptSelect:
            print(count)
            print(query)
            GenetatedText = acquire(query)
            GenetatedTextList.append(GenetatedText)
            print(GenetatedText)
            print('=====')
            count += 1
    except:
        print('The server is overloaded or not ready yet')
    GenetatedTextPd = pd.DataFrame()
    GenetatedTextPd['GenetatedText'] = GenetatedTextList
    GenetatedTextPd.to_csv('../data_generation/GenetatedText_500.csv')
    return GenetatedTextPd

if __name__ == '__main__':
    news_Info_df = load_mind_news()
    GpromptList = Gprompt_generation(news_Info_df)
    HpromptList = Hprompt_generation(news_Info_df)
    GenetatedTextList = extraction_origin_news_abstract(GpromptList, args)
    PolishedTextList = extraction_polish_news_abstract(HpromptList, args)
