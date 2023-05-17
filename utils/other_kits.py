# -*- coding: utf-8 -*-



# import torch
# import numpy as np

def compute_f1(gold, predicted, target_label_id=1, details=False):
    c_predict = 0
    c_correct = 0
    c_gold = 0
    false_neg = 0
    false_pos = 0
    all_pos = 0
    all_neg = 0
    for g, p in zip(gold, predicted):
        if g == target_label_id:
            c_gold += 1
            all_pos += 1
            if p != target_label_id:
                false_neg += 1
        else:
            all_neg += 1
        if p == target_label_id:
            c_predict += 1
            if g != target_label_id:
                false_pos += 1
        if g == target_label_id and p == target_label_id:
            c_correct += 1

    p = c_correct / c_predict if c_predict != 0 else 0
    r = c_correct / c_gold if c_gold != 0 else 0
    f = 2 * p * r / (p + r) if p + r != 0 else 0
    if details == "full":
        return p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg

    if details:
        return p, r, f, c_correct, c_predict, c_gold
    return p, r, f
