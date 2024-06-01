import pandas as pd
import os
import re
import sys
import ast
from evaluate import load
from ignite.metrics import Bleu
from torchmetrics.text.rouge import ROUGEScore
from typing import Callable, Dict, Iterable, List, Union, Tuple
from os import listdir
from os.path import isfile, join
from pathlib import Path

def concat_gen(x, q_num, TYPE2CONTROL_SIGNAL):
    tot_gen = []
    for i in TYPE2CONTROL_SIGNAL:
        tot_gen += x[i][:q_num]
    return tot_gen

def tokenize(x):
    temp = []
    for i in x:
        try:
            temp.append(i.split())
        except:
            continue
    return temp

def rougel_eval(df):
    rouge = ROUGEScore(use_stemmer=True, rouge_keys='rougeL', accumulate='best')
    
    for j in df.index:
        for c in df.loc[j, 'question']:
            rouge(c, [df.loc[j, 'tot_gen']])
            
    rougel_list = rouge.compute()
    print('rouge-L:', rougel_list)
    
    return rougel_list

def selfbleu_eval(df):
    m = Bleu(smooth='smooth1')

    for j in df.index:
        tokenized_preds = tokenize(df.loc[j, 'tot_gen'])
        for p in range(len(df.loc[j, 'tot_gen'])):
            temp = tokenized_preds.copy()
            temp.pop(p)
            m.update(([tokenized_preds[p]], [temp]))
            
    bleu_list = m.compute().item()
    print('Self-BLEU:', bleu_list)
    
    return bleu_list

def bertscore_eval(df):
    bertscore = load("bertscore")
    results = []

    for j in df.index:
        for c in df.loc[j, 'question']:
            references = [c for i in range(len(df.loc[j, 'tot_gen']))]
            predictions = df.loc[j, 'tot_gen']
            result = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large", device='cuda:0')
            results.append(max(result['f1']))
            
    score = sum(results) / len(results)        
    print('BERScore:', score)
    return score

def bleurt_eval(df):
    bleurt = load("bleurt", "bleurt-20", module_type="metric")
    results = []

    for j in df.index:
        for c in df.loc[j, 'question']:
            references = [c for i in range(len(df.loc[j, 'tot_gen']))]
            predictions = df.loc[j, 'tot_gen']
            result = bleurt.compute(predictions=predictions, references=references)
            results.append(max(result["scores"]))

    score = sum(results) / len(results)        
    print('BLEURT:', score)
    return score

if __name__ == "__main__":
    TYPE2CONTROL_SIGNAL = ['What', 'How', 'Who', 'Why', 'Where', 'When', 'Which']

    # path for generated result
    result_dir = 'output/pred_q.csv'
    result_df = pd.read_csv(result_dir)
    for i in TYPE2CONTROL_SIGNAL:
        result_df[i] = result_df[i].apply(lambda x: ast.literal_eval(x))
        
    # path for golden data
    test_dir = 'data/ftqa_wh_2_test1.csv'
    gold_df = pd.read_csv(test_dir)
    gold_df = gold_df.loc[:,['cor_section', 'question']]
    gold_df.columns = ['context', 'question']

    # merge dataframe
    gb = gold_df.groupby(['context'])
    result = pd.DataFrame(gb['question'].unique())

    tot_df = pd.merge(result, result_df, on="context")
    tot_df = tot_df.dropna()
    tot_df['tot_gen'] = tot_df.apply(lambda x: concat_gen(x, 4, TYPE2CONTROL_SIGNAL), axis=1)

    #rougel
    rougel_eval(tot_df)

    #selfbleu
    selfbleu_eval(tot_df)

    #bertscore
    bertscore_eval(tot_df)

    #bleurt
    bleurt_eval(tot_df)