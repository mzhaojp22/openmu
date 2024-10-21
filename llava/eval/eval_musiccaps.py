import json
import datasets
import statistics
from eval_utils import bleu as lpbleu


# ------ SacreBLEU ------ #
metrics = datasets.load_metric("sacrebleu")  # , 'sacrebleu', 'meteor', 'bertscore')
with open("musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        metrics.add_batch(predictions=[data["text"]], references=[[data["gold_label"]]])
score = metrics.compute()
print("\nsacrebleu", score)

# ------ SacreBLEU-MJpost ------ #
from sacrebleu.metrics import BLEU

gold, pred = [], []
with open("musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        gold.append([data["gold_label"]])
        pred.append(data["text"])
bleu = BLEU()
bleu.corpus_score(pred, gold)
print("\nsacrebleu-py", score)

# ------ LPBLEU ------ #
gold, pred = [], []
with open("musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        gold.append([data["gold_label"]])
        pred.append(data["text"])
lpord1 = lpbleu(pred, gold, 1)
print("lpbleu1", lpord1)
lpord2 = lpbleu(pred, gold, 2)
print("lpbleu2", lpord2)
lpord3 = lpbleu(pred, gold, 3)
print("lpbleu3", lpord3)
lpord4 = lpbleu(pred, gold, 4)
print("lpbleu4", lpord4)

# ------ LPROUGE ------ #
from eval_utils import rouge as lprouge

gold, pred = [], []
with open("musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        gold.append(data["gold_label"])
        pred.append(data["text"])
rougeL = lprouge(pred, gold)
print("rougeL is ", rougeL)

# ------ LPROUGE1 ------ #
from eval_utils import rouge1 as lprouge1

gold, pred = [], []
with open("musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        gold.append(data["gold_label"])
        pred.append(data["text"])
rouge1 = lprouge1(pred, gold)
print("rouge1 is ", rouge1)


# ------ BERTScore ------ #
metrics = datasets.load_metric("bertscore")  # , 'sacrebleu', 'meteor', 'bertscore')
with open("musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        metrics.add_batch(predictions=[data["text"]], references=[data["gold_label"]])

score = metrics.compute(lang="en")
score = statistics.mean(score["f1"])
print("\nbert score", score)

# ------ METEOR ------ #
metrics = datasets.load_metric("meteor")  # , 'sacrebleu', 'meteor', 'bertscore')
with open(musiccap_test.jsonl", "r") as fin:
    for line in fin:
        data = json.loads(line)
        metrics.add_batch(predictions=[data["text"]], references=[data["gold_label"]])

score = metrics.compute()
print("\nmeteor", score)
