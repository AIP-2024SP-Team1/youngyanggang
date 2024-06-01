import pandas as pd
import os
import re
import sys
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