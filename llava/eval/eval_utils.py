"""Placeholder for metrics."""

from functools import partial
import evaluate
import numpy as np
import torch
import string


# CAPTIONING METRICS
def bleu(predictions, ground_truths, order):
    bleu_eval = evaluate.load("bleu")
    return bleu_eval.compute(
        predictions=predictions, references=ground_truths, max_order=order
    )["bleu"]


def meteor(predictions, ground_truths):
    # https://github.com/huggingface/evaluate/issues/115
    meteor_eval = evaluate.load("meteor")
    return meteor_eval.compute(predictions=predictions, references=ground_truths)[
        "meteor"
    ]


def rouge(predictions, ground_truths):
    rouge_eval = evaluate.load("rouge")
    return rouge_eval.compute(predictions=predictions, references=ground_truths)[
        "rougeL"
    ]


def rouge1(predictions, ground_truths):
    rouge_eval = evaluate.load("rouge")
    return rouge_eval.compute(predictions=predictions, references=ground_truths)[
        "rouge1"
    ]


def bertscore(predictions, ground_truths):
    bertscore_eval = evaluate.load("bertscore")
    score = bertscore_eval.compute(
        predictions=predictions, references=ground_truths, lang="en"
    )["f1"]
    return np.mean(score)


def vocab_diversity(predictions, references):
    train_caps_tokenized = [
        train_cap.translate(str.maketrans("", "", string.punctuation)).lower().split()
        for train_cap in references
    ]
    gen_caps_tokenized = [
        gen_cap.translate(str.maketrans("", "", string.punctuation)).lower().split()
        for gen_cap in predictions
    ]
    training_vocab = Vocabulary(train_caps_tokenized, min_count=2).idx2word
    generated_vocab = Vocabulary(gen_caps_tokenized, min_count=1).idx2word

    return len(generated_vocab) / len(training_vocab)


def vocab_novelty(predictions, tr_ground_truths):
    predictions_token, tr_ground_truths_token = [], []
    for gen, ref in zip(predictions, tr_ground_truths):
        predictions_token.extend(gen.lower().replace(",", "").replace(".", "").split())
        tr_ground_truths_token.extend(
            ref.lower().replace(",", "").replace(".", "").split()
        )

    predictions_vocab = set(predictions_token)
    new_vocab = predictions_vocab.difference(set(tr_ground_truths_token))

    vocab_size = len(predictions_vocab)
    novel_v = len(new_vocab) / vocab_size
    return vocab_size, novel_v


def caption_novelty(predictions, tr_ground_truths):
    unique_pred_captions = set(predictions)
    unique_train_captions = set(tr_ground_truths)

    new_caption = unique_pred_captions.difference(unique_train_captions)
    novel_c = len(new_caption) / len(unique_pred_captions)
    return novel_c
