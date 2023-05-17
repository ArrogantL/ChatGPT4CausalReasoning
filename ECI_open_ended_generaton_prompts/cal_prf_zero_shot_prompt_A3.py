# -*- coding: utf-8 -*-

import json
import re
from argparse import ArgumentParser

"""
python cal_prf_zero_shot_prompt_A3.py \
--output_dir < output_dir output by chatgpt_ECI_openA123.py with ("prompt_style" == 3) >
"""
parser = ArgumentParser()
parser.add_argument('--output_dir', type=str)

args = parser.parse_args()
file_name = args.output_dir
file_name += "/response.json"
with open(file_name, 'r', encoding="utf-8") as fin:
    lines = fin.readlines()

items = [json.loads(line) for line in lines]
c_correct = 0
c_gold = 0
c_predict = 0
# format check
for itemid, item in enumerate(items):
    gpt_response = item["gpt_response"]
    gold_pairs = []
    predicted_pairs = []
    print(gpt_response)
    gpt_response = re.sub("^(cause->effect|effect->cause):?|[()]|cause -> effect|cause->effec", "", gpt_response)
    gpt_response = re.sub("\s+", " ", gpt_response).strip()
    gpt_response_parts = re.split("AND|,", gpt_response)
    for part in gpt_response_parts:
        if "->" in part:
            event_line = part.split("->")
            for i in range(len(event_line) - 1):
                cause_e = event_line[i].strip()
                effect_e = event_line[i + 1].strip()
                predicted_pairs.append((cause_e, effect_e))

    # print("#" * 10, itemid, "#" * 10)
    # print(gpt_response)
    # for c, e in predicted_pairs:
    #     print("> ", c, " -> ", e, " <")
    words = item["words"]
    causal_triples = item["causal_triples"]

    for (l1, r1), (l2, r2) in causal_triples:
        e1 = " ".join(words[l1:r1])
        e2 = " ".join(words[l2:r2])
        gold_pairs.append((e1, e2))

    visited_gold = set()
    for c, e in predicted_pairs:
        for g_c, g_e in gold_pairs:
            if (g_c, g_e) in visited_gold:
                continue
            if len(set(re.split("\s", c.strip())).intersection(set(re.split("\s", g_c.strip())))) != 0 and len(
                    set(re.split("\s", e.strip())).intersection(set(re.split("\s", g_e.strip())))) != 0:
                c_correct += 1
                visited_gold.add((g_c, g_e))
                break

    c_gold += len(gold_pairs)
    c_predict += len(predicted_pairs)

p = c_correct / c_predict if c_predict != 0 else 0
r = c_correct / c_gold if c_gold != 0 else 0
f = 2 * p * r / (p + r) if p + r != 0 else 0
print("P: %.4f, R: %.4f, F: %.4f" % (p, r, f))
print("Correct: %d, Predict: %d, Gold: %d" % (c_correct, c_predict, c_gold))
print("& %.4f & %.4f & %.4f" % (p, r, f))
print("len(items)", len(items))
