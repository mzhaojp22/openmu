import json
import datasets
import statistics
from eval_utils import bleu as lpbleu

# Assuming two fields of model predictions:
# "text": outputs from the model
# "gold_label": gold standard used for evaluation

MODEL_PREDICTION_JSON = ""

# SacreBLEU
metrics = datasets.load_metric("sacrebleu")
with open(MODEL_PREDICTION_JSON, "r") as fin:
    for line in fin:
        data = json.loads(line)
        metrics.add_batch(predictions=[data["text"]], references=[[data["gold_label"]]])
score = metrics.compute()
print("\nsacrebleu", score)

# BLEU
gold, pred = [], []
with open(MODEL_PREDICTION_JSON, "r") as fin:
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

# RougeL
from eval_utils import rouge as lprouge

gold, pred = [], []
with open(MODEL_PREDICTION_JSON, "r") as fin:
    for line in fin:
        data = json.loads(line)
        gold.append(data["gold_label"])
        pred.append(data["text"])
rougeL = lprouge(pred, gold)
print("rougeL is ", rougeL)

# Rouge1
from eval_utils import rouge1 as lprouge1

gold, pred = [], []
with open(MODEL_PREDICTION_JSON, "r") as fin:
    for line in fin:
        data = json.loads(line)
        gold.append(data["gold_label"])
        pred.append(data["text"])
rouge1 = lprouge1(pred, gold)
print("rouge1 is ", rouge1)


# BertScore
metrics = datasets.load_metric("bertscore")  # , 'sacrebleu', 'meteor', 'bertscore')
with open(MODEL_PREDICTION_JSON, "r") as fin:
    for line in fin:
        data = json.loads(line)
        metrics.add_batch(predictions=[data["text"]], references=[data["gold_label"]])

score = metrics.compute(lang="en")
score = statistics.mean(score["f1"])
print("\nbert score", score)

# Meteor
metrics = datasets.load_metric("meteor")  # , 'sacrebleu', 'meteor', 'bertscore')
with open(MODEL_PREDICTION_JSON, "r") as fin:
    for line in fin:
        data = json.loads(line)
        metrics.add_batch(predictions=[data["text"]], references=[data["gold_label"]])

score = metrics.compute()
print("\nmeteor", score)
