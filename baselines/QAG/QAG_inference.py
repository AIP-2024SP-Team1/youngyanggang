
import argparse
from typing import Dict, List
import argparse
import logging
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from allennlp.predictors.predictor import Predictor
import spacy

from QAG_Generation_E2E.lightning_base import BaseTransformer
from QAG_Generation_E2E.utils import (
    ROUGE_KEYS,
    freeze_params,
    lmap,
    use_task_specific_params,
)

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"
TYPE2CONTROL_SIGNAL = ['What', 'How', 'Who', 'Why', 'Where', 'When', 'Which']

class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, self.mode)

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)


    def generate(self, input_ids, attention_mask, **generate_kwargs):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=20,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=3,
            early_stopping=True,
            use_cache=False,
            **generate_kwargs
        )
        return generated_ids

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries_or_translations(
    examples: List[str],
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    task="summarization",
    prefix='',
    args=None,
    **generate_kwargs,
) -> Dict:
    output_list = []

    model_name = str(model_name)

    model: SummarizationModule = SummarizationModule(args)    
    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()
    
    if args.device != 'cpu':
        model.to('cuda:{}'.format(args.device))

    
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})
    model.resize_token_embeddings(len(tokenizer))
    
    print('#############################################')
    print("# model is loaded from", args.ckpt_path)
    print('# tokenizer.all_special_tokens =', model.tokenizer.all_special_tokens)
    print('# tokenizer.all_special_ids =', model.tokenizer.all_special_ids)
    print('#############################################')
    
    use_task_specific_params(model, task)
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]
        
        if device=='cpu':
            batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest")
        else:
            batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to('cuda:{}'.format(device))
        
        if len(batch.input_ids[0]) > 1024:
            
            if device=='cpu':
                end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:]
            else:
                end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:].to('cuda:{}'.format(device))
            
            input_ids = torch.cat((batch.input_ids[0][:1022], end_token), 0).unsqueeze(0)
            batch.input_ids = input_ids
            
        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            **generate_kwargs,
        )
        dec = model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            output_list.append(hypothesis)

    return output_list

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def run_generate(examples, args):
    parsed = {}

    generate_results = generate_summaries_or_translations(
        examples,
        args.model_name_or_path,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        args=args,
        **parsed,
    )

    return generate_results

def e2e(input_content, predictor, nlp, args):
    
    input_for_ACQ_model = []
    
    doc = nlp(input_content)
    json_samples = []
    for sent in doc.sents:
        json_samples.append({'sentence': sent.text})
    srl_results = predictor.predict_batch_json(json_samples)

    each_answer = ""

    unique_answers = []

    for ent in doc.ents:
        each_answer = ent.text.replace("\n", "")

        each_answer = each_answer.strip('.,\'').strip() + ' .'

        if each_answer in unique_answers:
            continue

        if len(each_answer)>1:
            input_for_ACQ_model.append(each_answer + " </s> " + input_content.replace("\n", ""))
            unique_answers.append( each_answer )
        
    
    for chunk in doc.noun_chunks:
        each_answer = chunk.text.replace("\n", "")
        each_answer = each_answer.strip('.,\'').strip() + ' .'

        if each_answer in unique_answers:
            continue

        if len(chunk.text.split(" ")) >= 2:
            input_for_ACQ_model.append(each_answer + " </s> " + input_content.replace("\n", ""))
            unique_answers.append( each_answer )

    for k in srl_results:
        if k['verbs'] == []:
            continue
        relevant_words = []
        for idx in range(len(k['verbs'][0]['tags'])):
            if k['verbs'][0]['tags'][idx] != 'O':
                relevant_words.append(k['words'][idx])
        target = ' '.join(relevant_words)

        each_answer = target.replace("\n", "").strip('.,\'').strip() + ' .'
        
        if each_answer in unique_answers:
            continue
        
        input_for_ACQ_model.append(each_answer + " </s> " + input_content.replace("\n", ""))
        unique_answers.append( each_answer )

    output_for_ACQ_model = run_generate(input_for_ACQ_model, args)

    for i in range(len(output_for_ACQ_model)):
        output_for_ACQ_model[i] = output_for_ACQ_model[i].replace("\n", "").split("?")[0].strip(' ?') + " ?"

    data_AC_Q = []

    for i in range(len(input_for_ACQ_model)):
        data_AC_Q.append( [input_for_ACQ_model[i].strip().lower().split(' </s> ')[0], input_for_ACQ_model[i].strip().lower().split(' </s> ')[1], output_for_ACQ_model[i].strip().lower() ] )
    

    return data_AC_Q

def load_data(data_dir):
    try:
        df = pd.read_excel(data_dir)
        df = df.loc[:,['cor_section', 'question']]
        df.columns = ['context', 'question']
    except:
        df = pd.read_csv(data_dir)
        df = df.loc[:,['cor_section', 'question']]
        df.columns = ['context', 'question']
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default='', help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=1, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
            "--cache_dir",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
    parser.add_argument(
            "--ckpt_path",
            default="./QG_model_epoch=2_new.ckpt",
            type=str,
            help='path tooo stored model checkpoints',
        )
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/bart-large", help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--config_name", type=str, default="facebook/bart-large")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-large")
    parser.add_argument("--test_max_target_length", type=int)
    parser.add_argument("--eval_max_length", type=int)
    parser.add_argument("--type_embedding_enabled", type=bool, default=True)
    args, rest = parser.parse_known_args()
    
    if DEFAULT_DEVICE != 'cpu':
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz", cuda_device=0)
    else:
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz", )
    
    nlp = spacy.load("en_core_web_sm")    

    result_dir= os.path.join(args.output_dir, 'pred_q')      
    df = load_data(args.data_dir)

    df_context = df['context'].unique().copy()
    tot_batch = round(len(df_context)/args.bs)
    result_df = pd.DataFrame(columns = TYPE2CONTROL_SIGNAL)
    result_df['context'] = df_context
    gen_dict = {qt: [] for qt in TYPE2CONTROL_SIGNAL}
    for i in range(tot_batch):
        if i != tot_batch - 1:
            examples = df_context[i*args.bs: (i+1)*args.bs]
            for qt in TYPE2CONTROL_SIGNAL:
                generated_sents = e2e(examples, predictor, nlp, args)
                gen_dict[qt] += generated_sents
        else:
            examples = df_context[(tot_batch - 1)*args.bs:]  
            for qt in TYPE2CONTROL_SIGNAL: 
                generated_sents = e2e(examples, predictor, nlp, args)
                gen_dict[qt] += generated_sents
        print((i+1)/tot_batch*100)    
    gen_dict = pd.DataFrame(gen_dict)
    result_df = pd.concat([result_df['context'], gen_dict], axis=1)
    with torch.cuda.device('cuda:{}'.format(args.device)):
        torch.cuda.empty_cache()
    result_df.to_csv(result_dir+'.csv', index=False)