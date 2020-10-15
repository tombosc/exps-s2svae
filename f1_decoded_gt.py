import json
import argparse
import fasttext as fT
import numpy as np
from sklearn.metrics import f1_score


def read_text(input_fn):
    lines = []
    with open(input_fn, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def read_ground_truth(input_fn):
    lines = []
    with open(input_fn, 'r') as f:
        for line in f:
            lines.append(int(line.strip()))
    return np.asarray(lines)


def most_probable_label(list_documents, classifier):
    predictions = []
    for doc in list_documents:
        # most probable labels and proba
        labels, proba = classifier.predict(doc)
        predictions.append(int(labels[0][-1]))
    return np.asarray(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_pretrained_fasttext", type=str)
    parser.add_argument("fn_hyp", type=str)
    parser.add_argument("fn_ref", type=str)
    parser.add_argument("fn_label", type=str)
    parser.add_argument("fn_results_out", type=str)
    args = parser.parse_args()

    hyp = read_text(args.fn_hyp)
    ref = read_text(args.fn_ref)
    labels = read_ground_truth(args.fn_label)
    clf = fT.load_model(args.fn_pretrained_fasttext)
    pred_ref = most_probable_label(ref, clf)
    pred_hyp = most_probable_label(hyp, clf)
    results = {}
    for average in ['weighted', 'macro']:
        f1_ref_hyp = f1_score(pred_ref, pred_hyp, average=average)
        f1_gt_hyp = f1_score(labels, pred_hyp, average=average)
        f1_gt_ref = f1_score(labels, pred_ref, average=average)
        results[average + '_F1_ref_hyp'] = f1_ref_hyp
        results[average + '_F1_gt_hyp'] = f1_gt_hyp
        results[average + '_F1_gt_ref'] = f1_gt_ref
    with open(args.fn_results_out, "w") as f:
        json.dump(results, f)
