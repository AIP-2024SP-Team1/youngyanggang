from rouge_score import rouge_scorer
from bert_score import score as bert_score
from bleurt import score as bl
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

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