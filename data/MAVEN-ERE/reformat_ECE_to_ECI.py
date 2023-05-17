# -*- coding: utf-8 -*-

import json
from collections import defaultdict
from random import random, seed, randint

seed(1234)
counter=defaultdict(int)
fout = open("all.with_neg.ECI.with_docname_sentid.train.json", 'w+', encoding="utf-8")
with open("all.with_neg.with_docname_sentid.train.json", 'r', encoding="utf-8") as fin:
    for line in fin:
        item = json.loads(line)
        # {"id": 1, "words": ["The", "sea"ls", "."], "causal_triples": [[[11, 12], [2, 3]], [[11, 12], [7, 8]]], "topic": 1}
        item_id = item["id"]
        words = item["words"]
        topic = -1
        if "causal_triples" in item:
            triples = item["causal_triples"]
            bi_causal_label=tri_causal_label=1
        elif "un_causal_triples" in item:
            triples = item["un_causal_triples"]
            bi_causal_label = tri_causal_label = 0
        # un_causal_triples
        for (l1, r1), (l2, r2) in triples:
            events = [list(range(l1, r1)), list(range(l2, r2))]
            cur = {"item_id": item_id, "words": words, "topic": -1, "events": events, "bi_causal_label": bi_causal_label,
                   "tri_causal_label": tri_causal_label,"doc_name": item["doc_name"], "sent_in_doc_id": item["sent_in_doc_id"]}
            counter[bi_causal_label]+=1
            fout.write(json.dumps(cur, ensure_ascii=False) + "\n")

fout.close()
print(counter)


