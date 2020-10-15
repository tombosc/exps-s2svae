import argparse
import json
import numpy as np


def read_no_labels(fn):
    documents = []
    with open(fn, 'r') as f:
        for doc in f:
            doc = doc.strip().split(' ')
            documents.append(doc)
    return documents


def reconstruction_rates_per_positions(hyp, ref):
    max_len_ref = max([len(r) for r in ref])
    correct_reconstruction = np.zeros(max_len_ref)
    normalization = np.zeros(max_len_ref)
    correct_len = 0
    for h, r in zip(hyp, ref):
        for i in range(min(len(r), len(h))):
            if h[i] == r[i]:
                correct_reconstruction[i] += 1
            normalization[i] += 1
        if len(r) == len(h):
            correct_len += 1
    return correct_reconstruction / normalization, correct_len / len(ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp_fn")
    parser.add_argument("ref_fn")
    args = parser.parse_args()
    doc_hyp = read_no_labels(args.hyp_fn)
    doc_ref = read_no_labels(args.ref_fn)
    r, correct_len = reconstruction_rates_per_positions(doc_hyp, doc_ref)
    results = {
        "retrieve_pc": r.tolist(),
        "correct_len_pc": correct_len,
    }
    print(json.dumps(results))
