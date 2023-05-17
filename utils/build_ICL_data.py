# -*- coding: utf-8 -*-

import json
import random
from collections import defaultdict


def get_e_CARE_task3_demonstrations(pos, neg):
    with open("data/e-CARE/e-CARE-main-code/dataset/Causal_Reasoning/train.jsonl", 'r', encoding="utf-8") as fin:
        pos_prompts = []
        neg_prompts = []
        for line in fin:
            item = json.loads(line)
            res1, res2 = "Yes", "No"
            if item["label"] == 1:
                res1, res2 = "No", "Yes"
            prompt1 = """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer: %s""" % (item["premise"], item["hypothesis1"], res1)
            prompt2 = """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer: %s""" % (item["premise"], item["hypothesis2"], res2)
            if item["label"] == 1:
                prompt1, prompt2 = prompt2, prompt1
            pos_prompts.append(prompt1)
            neg_prompts.append(prompt2)
        random.seed(0)
        random.shuffle(pos_prompts)
        random.shuffle(neg_prompts)
        out_ICLs = pos_prompts[:pos] + neg_prompts[:neg]
        random.shuffle(out_ICLs)

        return out_ICLs


def get_task3_COT(pos, neg):
    ecare_task3_COT_file = "e-CARE_tast3_COT.txt"

    ICL_list = []
    cur_icl = ""
    for line in open(ecare_task3_COT_file, 'r', encoding="utf-8"):
        if line == "\n":
            ICL_list.append(cur_icl)
            cur_icl = ""
        cur_icl += line
    ICL_list.append(cur_icl)
    cur_icl = ""
    assert len(ICL_list) == 12
    label2icl = defaultdict(list)
    for t in ICL_list:
        label = int(t.endswith("Yes\n"))
        # print(label,t[-10:])
        label2icl[label].append(t)
    random.seed(0)
    random.shuffle(label2icl[0])
    random.shuffle(label2icl[1])

    out_ICLs = label2icl[1][:pos] + label2icl[0][:neg]
    random.shuffle(out_ICLs)

    return out_ICLs


def get_ESC_demonstrations(pos, neg):
    with open("data/ESC/intra_sent_causality.json", 'r', encoding="utf-8") as fin:
        pos_prompts = []
        neg_prompts = []
        for line in fin:
            item = json.loads(line)
            words = item["words"]
            sent = " ".join(words)
            events = item["events"]
            event1 = " ".join([words[t] for t in events[0]])
            event2 = " ".join([words[t] for t in events[1]])
            bi_causal_label = item["bi_causal_label"]
            prompt = """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer: %s""" % (sent, event1, event2, ("No", "Yes")[bi_causal_label])

            if bi_causal_label == 1:
                pos_prompts.append(prompt)
            else:
                neg_prompts.append(prompt)
        random.seed(0)
        random.shuffle(pos_prompts)
        random.shuffle(neg_prompts)
        out_ICLs = pos_prompts[:pos] + neg_prompts[:neg]
        random.shuffle(out_ICLs)

        return out_ICLs


def get_esc_COT(pos, neg):
    esc_COT_file = "COT_4pos_8neg.txt"

    ICL_list = []
    cur_icl = ""
    for line in open(esc_COT_file, 'r', encoding="utf-8"):
        if line == "\n":
            ICL_list.append(cur_icl)
            cur_icl = ""
        cur_icl += line
    ICL_list.append(cur_icl)
    cur_icl = ""
    assert len(ICL_list) == 12
    label2icl = defaultdict(list)
    for t in ICL_list:
        label = int(t.endswith("Yes.\n"))
        # print(label,t[-10:])
        label2icl[label].append(t)
    random.seed(0)
    random.shuffle(label2icl[0])
    random.shuffle(label2icl[1])
    assert len(label2icl[1])==4

    out_ICLs = label2icl[1][:pos] + label2icl[0][:neg]
    random.shuffle(out_ICLs)
    return out_ICLs

if __name__ == '__main__':
    # out_ICLs=get_e_CARE_task3_demonstrations(pos=4, neg=8)
    # for t in out_ICLs:
    #     print(t)
    #     print()
    print(len(get_task3_COT(2, 3)))
