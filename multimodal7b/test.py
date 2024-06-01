import torch
import pandas as pd
import numpy as np

import util.misc as misc
import llama
from llama.llama_adapter import LLaMA_adapter
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from bleurt import score as bl
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

import os
import argparse
import time
import re

def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 eval', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='./model/', type=str,
                        help='path to LLaMA pretrained checkpoint')
    """parser.add_argument('--pretrained_path', default='./ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth', type=str,
                        help='path to checkpoint from pretrain stage')"""
    parser.add_argument('--pretrained_path', default='./model/7Badapter-7B_v2.pth', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=1024, type=int,
                        help='max number of input words')

    # Dataset parameters
    parser.add_argument('--dataset_path', default='./data/finetune/ftqa_wh_2_test1.xlsx', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    return parser

def calculate_evaluation_metrics(generated, reference):
    rouge_l = []
    bert_scores = []
    bleurt = []
    self_bleu = []

    scorer_r = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scorer_b = bl.BleurtScorer("./bleurt/BLEURT-20")
    chencherry = SmoothingFunction()

    for generated, ground_truth in zip(generated, reference):
        # Split the generated text into separate questions
        items = re.split(r'\n\d+\.\s*', generated)
        generated_questions = [item for item in items if item]

        # Calculate Rouge-L F1
        tmp = []
        for gen_q in generated_questions:
            score = scorer_r.score(ground_truth, gen_q)['rougeL'].fmeasure
            tmp.append(score)
        rouge_l.append(tmp)

        # Calculate BERTScore
        P, R, F1 = bert_score(generated_questions, [ground_truth]*len(generated_questions), lang="en", verbose=False)
        bert_scores.append(list(F1.numpy()))

        # Calculate BLEURT
        tmp = []
        for gen_q in generated_questions:
            score = scorer_b.score(references=[ground_truth], candidates=[gen_q])
            tmp.append(score[0])
        bleurt.append(tmp)

        # Calculate Self-BLEU
        tmp = []
        for i, gen_q in enumerate(generated_questions):
            # Use other generated questions as references for the current candidate
            other_questions = [q for j, q in enumerate(generated_questions) if i != j]
            ref = [word_tokenize(q) for q in other_questions]
            candidate = word_tokenize(gen_q)
            if ref:  
                bleu_score = sentence_bleu(ref, candidate, smoothing_function=chencherry.method1)
                tmp.append(bleu_score)
            else:
                tmp.append(0)
        self_bleu.append(tmp)

    return rouge_l, bert_scores, bleurt, self_bleu

def main(args):
    device = torch.device(args.device)
    torch.cuda.empty_cache()

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, max_seq_len=args.max_words)

    misc.load_model(model, args.pretrained_path)
    model.eval()
    model.to(device)

    dataset = pd.read_excel(args.dataset_path) 

    context = dataset['cor_section'].tolist()[:225]
    question = dataset['question'].tolist()[:225]
    generated_texts = []

    with torch.no_grad():
        for input in context:
            prompt = llama.format_prompt(
                "Generate appropriate multiple questions considering the context of the input. \
                The interrogative word of the question should be one of the following seven: [what, when, where, which, who, why, how].", 
                input
            )
            img = torch.zeros(3, 224, 224)
            img = img.unsqueeze(0).to(device)
            generated_text = model.generate(img, [prompt], max_gen_len=args.max_words)
            generated_texts.extend(generated_text)

    id = [i+1 for i in range(len(context))]
    data = pd.DataFrame({
        'id': id,
        'context': context,
        'question': question,
        'generated': generated_texts,
    })   
    data.to_csv('./output/result(before).csv')
    data = pd.read_csv('./output/result(before).csv')

    # Extract generated texts and ground truth questions
    reference = data['question'].tolist()
    generated = data['generated'].tolist()

    # Calculate evaluation metrics
    rouge_l, bert_scores, bleurt, self_bleu = calculate_evaluation_metrics(generated, reference)

     # Print the results    
    print('Rouge-L F1 Average:', np.mean(list(np.mean(item) for item in rouge_l)))
    print('BERTScore F1 Average:', np.mean(list(np.mean(item) for item in bert_scores)))
    print('BLEURT Average:', np.mean(list(np.mean(item) for item in bleurt)))
    print('Self-BLEU Average:', np.mean(list(np.mean(item) for item in self_bleu)))

    score = pd.DataFrame({
        'Rouge-L': rouge_l,
        'BERTScore': bert_scores,
        'BLEURT': bleurt,
        'Self-BLEU': self_bleu
    })

    merged = pd.concat([data, score], axis=1)
    merged = merged.drop([merged.columns[0]], axis=1)
    merged.to_csv('./output/merged(before).csv')

if __name__ == '__main__':
    nltk.download('punkt')
    args = get_args_parser()
    args = args.parse_args()
    main(args)
