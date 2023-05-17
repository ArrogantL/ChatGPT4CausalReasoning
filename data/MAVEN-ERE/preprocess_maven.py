import json
import logging
import os
import pickle
import random
from collections import defaultdict






def process_origin_data_with_uncausal_pair():
    corpus = []
    # for file in ["train.jsonl", "valid.jsonl"]:
    # for file in ["valid.jsonl"]:
    for file in ["train.jsonl"]:
        corpus.extend([json.loads(line) for line in open(file, 'r', encoding="utf-8").readlines()])
    count_dict = defaultdict(int)
    total_id = 0
    fout = open("all.with_neg.with_docname_sentid.train.json", "w+", encoding="utf-8")
    for itemid, item in enumerate(corpus):
        doc_id = item['id']
        tokens = item["tokens"]
        sentences = item["sentences"]
        causality_rels = item["causal_relations"]
        cause_rels = causality_rels["CAUSE"]
        prediction_rels = causality_rels["PRECONDITION"]
        sentid2rels = {}
        events = {t["id"]: t for t in item["events"]}

        cause_dict = {}
        for c, e in cause_rels + prediction_rels:
            if c not in cause_dict:
                cause_dict[c] = set()
            cause_dict[c].add(e)
            mc_ls = events[c]["mention"]
            me_ls = events[e]["mention"]
            for mc in mc_ls:
                for me in me_ls:
                    if mc["sent_id"] == me["sent_id"] and len(tokens[mc["sent_id"]]) <= 70:
                        if mc['sent_id'] not in sentid2rels:
                            sentid2rels[mc['sent_id']] = []
                        assert " ".join(tokens[mc['sent_id']][mc["offset"][0]:mc["offset"][1]]) == mc["trigger_word"]
                        assert " ".join(tokens[me['sent_id']][me["offset"][0]:me["offset"][1]]) == me["trigger_word"]
                        assert (mc["offset"], me["offset"]) not in sentid2rels[mc['sent_id']]
                        sentid2rels[mc['sent_id']].append((mc["offset"], me["offset"]))
                        # count_dict["cause"]+=1
                        print(sentences[mc["sent_id"]])
                        print(mc["trigger_word"], "->", me["trigger_word"])
        uncausal_pair_list = []
        uncausal_pair_list_hash = []
        event_id_sort_list = list(sorted(events))
        for eid1 in range(len(event_id_sort_list) - 1):
            for eid2 in range(eid1, len(event_id_sort_list)):
                tmp_eid1, tmp_eid2 = event_id_sort_list[eid1], event_id_sort_list[eid2]

                if random.randint(0, 1) == 1:
                    tmp_eid1, tmp_eid2 = tmp_eid2, tmp_eid1

                if ((tmp_eid1 in cause_dict) and (tmp_eid2 in cause_dict[tmp_eid1])) or \
                        ((tmp_eid2 in cause_dict) and (tmp_eid1 in cause_dict[tmp_eid2])):
                    # 跳过有因果的pair
                    continue
                uncausal_pair_list.append([tmp_eid1, tmp_eid2])
                uncausal_pair_list_hash.append("%s####%s" % (tmp_eid1, tmp_eid2))
        assert len(set(event_id_sort_list)) == len(event_id_sort_list)
        # 记录

        neg_sentid2rels = {}
        for c, e in uncausal_pair_list:
            mc_ls = events[c]["mention"]
            me_ls = events[e]["mention"]
            for mc in mc_ls:
                for me in me_ls:
                    if mc["sent_id"] == me["sent_id"] and len(tokens[mc["sent_id"]]) <= 70:
                        if mc["offset"][0] == me["offset"][0] and mc["offset"][1] == me["offset"][1]:
                            # 跳过自指
                            continue
                        if mc['sent_id'] not in neg_sentid2rels:
                            neg_sentid2rels[mc['sent_id']] = []
                        assert " ".join(tokens[mc['sent_id']][mc["offset"][0]:mc["offset"][1]]) == mc["trigger_word"]
                        assert " ".join(tokens[me['sent_id']][me["offset"][0]:me["offset"][1]]) == me["trigger_word"]
                        assert (mc["offset"], me["offset"]) not in neg_sentid2rels[mc['sent_id']]
                        neg_sentid2rels[mc['sent_id']].append((mc["offset"], me["offset"]))
                        # count_dict["cause"]+=1
                        print(sentences[mc["sent_id"]])
                        print(mc["trigger_word"], "->", me["trigger_word"])

        for sentid in sentid2rels:
            fout.write(
                json.dumps(
                    {"id": total_id, "words": tokens[sentid], "causal_triples": sentid2rels[sentid], "topic": 1, "doc_name": doc_id, "sent_in_doc_id": sentid},
                    ensure_ascii=False) + "\n")
            total_id += 1

        for sentid in neg_sentid2rels:
            fout.write(
                json.dumps(
                    {"id": total_id, "words": tokens[sentid], "un_causal_triples": neg_sentid2rels[sentid], "topic": 1, "doc_name": doc_id,
                     "sent_in_doc_id": sentid},
                    ensure_ascii=False) + "\n")
            total_id += 1
    fout.close()






if __name__ == '__main__':
    process_origin_data_with_uncausal_pair()
    ...
