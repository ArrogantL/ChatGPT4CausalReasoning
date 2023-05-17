# -*- coding: utf-8 -*-

"""
需要控制这些因素：

因果比例

事件间距

事件密度

显隐式因果的数量

主题

"""
import json
import random

from gpt3_ECI4_analyze_isexplicit import get_id2explicit


def cal_pos_neg(item_ls):
    pos=0
    neg=0
    for item in item_ls:
        if item["bi_causal_label"]==1:
            pos+=1
        elif item["bi_causal_label"]==0:
            neg+=1
        else:
            assert False
    total=len(item_ls)
    return pos/total,neg/total,pos,neg

def cal_ask_for(item_ls):
    pos=0
    neg=0
    for item in item_ls:
        if item["ask-for"]=="cause":
            pos+=1
        elif item["ask-for"]=="effect":
            neg+=1
        else:
            assert False
    total=len(item_ls)
    return pos/total,neg/total,pos,neg

def cal_ex_im(item_ls):
    pos=0
    neg=0
    for item in item_ls:
        if id2exp[item["item_id"]]:
            pos+=1
        else:
            neg+=1
    total=len(item_ls)
    return pos/total,neg/total,pos,neg

def cal_topic(item_ls):
    all_topic=['1', '3', '4', '5', '7', '8', '12', '13', '14', '16', '18', '19', '20', '22', '23', '24', '30', '32', '33', '35', '37', '41']
    res=[0]*len(all_topic)
    for item in item_ls:
        res[all_topic.index(item["topic"])]+=1
    total=len(item_ls)
    res=[x/total for x in res]
    return res

id2exp=get_id2explicit()


# items=[json.loads(line) for line in open("/Users/XXX/Desktop/projects/LLM/GPT3/data/ESC/intra_sent_causality.two_direction.train.json",'r',encoding="utf-8").readlines()]
# random.seed(1)
# random.shuffle(items)
# sample_items=items[:1000]
# # with open("ESC_ECI.sample1000.jsonl",'w+',encoding="utf-") as fout:
# #     for item in sample_items:
# #         fout.write(json.dumps(item,ensure_ascii=False)+"\n")
#
# print("### POS NEG ###")
# print(cal_pos_neg(items))
# print(cal_pos_neg(sample_items))
#
# print("### EX IM ###")
# print(cal_ex_im(items))
# print(cal_ex_im(sample_items))
#
# print("### TOPIC ###")
# format_str="%.3f\t"*22
# print(format_str%tuple(cal_topic(items)))
# print(format_str%tuple(cal_topic(sample_items)))

# 明显，在主题、正负例比例、显隐式，三个方面，都称得上是均衡。

# items=[json.loads(line) for line in open("/Users/XXX/Desktop/projects/LLM/GPT3/data/CTB/causal_time_bank.with_trigger.json",'r',encoding="utf-8").readlines()]
# pos_items=[item for item in items if item["bi_causal_label"]==1]
# neg_items=[item for item in items if item["bi_causal_label"]==0]
# random.seed(2)
# sample_rate=1000/len(items)
# sample_pos_num=int(sample_rate*len(pos_items))
# sample_neg_num=int(sample_rate*len(neg_items))
#
# random.shuffle(pos_items)
# random.shuffle(neg_items)
# sample_pos=pos_items[:sample_pos_num+1]
# sample_neg=neg_items[:sample_neg_num]
#
# sample_items=sample_pos+sample_neg
# print(len(sample_items))
# random.shuffle(sample_items)
# with open("CTB_ECI.sample1000.jsonl",'w+',encoding="utf-") as fout:
#     for item in sample_items:
#         fout.write(json.dumps(item,ensure_ascii=False)+"\n")
#
# print("### POS NEG ###")
# print(cal_pos_neg(items))
# print(cal_pos_neg(sample_items))

# items=[json.loads(line) for line in open("/Users/XXX/Desktop/projects/LLM/GPT3/data/MAVEN_ERE/all.with_neg.ECI.with_docname_sentid.valid.json",'r',encoding="utf-8").readlines()]
# pos_items=[item for item in items if item["bi_causal_label"]==1]
# neg_items=[item for item in items if item["bi_causal_label"]==0]
# random.seed(2)
# sample_rate=1000/len(items)
# sample_pos_num=int(sample_rate*len(pos_items))
# sample_neg_num=int(sample_rate*len(neg_items))
#
# random.shuffle(pos_items)
# random.shuffle(neg_items)
# sample_pos=pos_items[:sample_pos_num+1]
# sample_neg=neg_items[:sample_neg_num]
#
# sample_items=sample_pos+sample_neg
# print(len(sample_items))
# random.shuffle(sample_items)
# with open("MAVEN_ECI.sample1000.jsonl",'w+',encoding="utf-") as fout:
#     for item in sample_items:
#         fout.write(json.dumps(item,ensure_ascii=False)+"\n")
#
# print("### POS NEG ###")
# print(cal_pos_neg(items))
# print(cal_pos_neg(sample_items))

# items=[json.loads(line) for line in open("/Users/XXX/Desktop/projects/LLM/GPT3/data/e-CARE/Causal_Reasoning/test.jsonl",'r',encoding="utf-8").readlines()]
# pos_items=[item for item in items if item["ask-for"]=="cause"]
# neg_items=[item for item in items if item["ask-for"]=="effect"]
# random.seed(2)
# sample_rate=1000/len(items)
# sample_pos_num=int(sample_rate*len(pos_items))
# sample_neg_num=int(sample_rate*len(neg_items))
#
# random.shuffle(pos_items)
# random.shuffle(neg_items)
# sample_pos=pos_items[:sample_pos_num+1]
# sample_neg=neg_items[:sample_neg_num]
#
# sample_items=sample_pos+sample_neg
# print(len(sample_items))
# random.shuffle(sample_items)
# with open("e-CARE.task1.sample1000.jsonl",'w+',encoding="utf-") as fout:
#     for item in sample_items:
#         fout.write(json.dumps(item,ensure_ascii=False)+"\n")
#
# print("### ask for ###")
# print(cal_ask_for(items))
# print(cal_ask_for(sample_items))


items=[json.loads(line) for line in open("/Users/XXX/Desktop/projects/LLM/GPT3/data/e-CARE/Explanation_Generation/test_gen.jsonl",'r',encoding="utf-8").readlines()]
random.seed(2)
random.shuffle(items)
sample_items=items[:1000]
with open("e-CARE.task2.sample1000.jsonl",'w+',encoding="utf-") as fout:
    for item in sample_items:
        fout.write(json.dumps(item,ensure_ascii=False)+"\n")





