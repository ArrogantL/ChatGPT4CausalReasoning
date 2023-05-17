# -*- coding: utf-8 -*-

import json
import os
import re


def task1(file_name, test_file=None):
    items = [json.loads(line) for line in open(file_name, 'r', encoding="utf-8").readlines()]
    output_dict = {}
    correct_num = 0
    for item in items:

        item_index = item["index"]
        print(item_index)
        gpt_response = item["gpt_response"]
        gpt_response = gpt_response.strip()
        hypothesis1 = item["hypothesis1"]
        hypothesis2 = item["hypothesis2"]

        if gpt_response.startswith("Option 1:"):
            label = 0
        elif gpt_response.startswith("Option 2:"):
            label = 1
        elif gpt_response == hypothesis1:
            label = 0
        elif gpt_response == hypothesis2:
            label = 1
        elif "The correct answer is Option 1." in gpt_response:
            label = 0
        elif "The correct answer is Option 2." in gpt_response:
            label = 1
        elif gpt_response == "Option 1":
            label = 0
        elif gpt_response == "Option 2":
            label = 1
        elif gpt_response == "Option 2.":
            label = 1
        else:
            print(gpt_response)
            assert False
        output_dict[item_index] = label
        if label == item["label"]:
            correct_num += 1
    output_file_name = file_name[:-5] + ".format.json"
    json.dump(output_dict, open(output_file_name, 'w+', encoding="utf-8"), ensure_ascii=False)

    if test_file is None:
        test_file = "data/e-CARE/Causal_Reasoning/test.jsonl"
    os.system("python data/e-CARE/e-CARE-main-code/causal_reasoning.py %s %s" % (output_file_name, test_file))
    print("our acc:", correct_num / len(items))


def task2(file_name, test_file=None):
    items = [json.loads(line) for line in open(file_name, 'r', encoding="utf-8").readlines()]
    output_dict = {}
    for item in items:
        # {"index": "test-105", "premise": "The vet have cut the throat of the goat in front of his students.", "ask-for": "effect", "hypothesis1": "The vet was fainted.", "hypothesis2": "He can't use the operation knife any more.", "label": 1, "gpt_response": " Option 2: He can't use the operation knife any more."}
        item_index = item["index"]
        gpt_response = item["gpt_response"]
        output_dict[item_index] = gpt_response
    output_file_name = file_name[:-5] + ".format.json"
    json.dump(output_dict, open(output_file_name, 'w+', encoding="utf-8"), ensure_ascii=False)

    if test_file is None:
        test_file = "data/e-CARE/Explanation_Generation/test_gen.jsonl"
    os.system("python data/e-CARE/e-CARE-main-code/conceptual_explanation_generation.py %s %s" % (output_file_name, test_file))


def eval_task1_in_pos_neg(file_name, is_COT=False):
    print(file_name)
    if not is_COT:
        assert "cot" not in file_name.lower() or "cot-zero-shot" in file_name.lower()

    def response_parse(origin_response):
        if is_COT:
            assert "Answer:" in origin_response, origin_response
            origin_response = origin_response.split("Answer:")[-1][:10]
        response_str = origin_response.lower().strip()
        response_str = re.split("[^a-z]", response_str)
        if origin_response == " It is possible that there is a causal relationship between" \
                or origin_response == " It is possible that shaking your foot could have caused" \
                or origin_response == "\n\nIt is possible that there is a causal" \
                or origin_response == " It is possible that Event A caused Event B," \
                or origin_response == " It is possible that the compacted soil caused the" \
                or origin_response == "\n\nThere is a causal relationship between Event A" \
                or origin_response == " It is possible that Lucy's work as a chemist" \
                or origin_response.startswith(" It is possible that") \
                or origin_response.startswith("It is possible that Event A") \
                or origin_response.startswith(" Possibly") \
                or origin_response.startswith(" It is possible,") \
                or origin_response.startswith("There is a possible causal") \
                or origin_response.startswith("There might be a possible causal") \
                or origin_response.startswith("It is implied that there may be a causal relationship") \
                or origin_response.startswith("It depends on the order of the events. If") \
                or origin_response.startswith("It is possible,") \
                or origin_response.startswith("It is possible, but not definitive.") \
                or origin_response.startswith("There might be an indirect") \
                or origin_response.startswith("Potentially, there could be a causal relationship between") \
                or origin_response.startswith("Potentially, yes.") \
                or origin_response.startswith("There may be a causal") \
                or origin_response.startswith(" There is a causal relationship") \
                or origin_response.startswith("Possibly") \
                or origin_response.startswith("There might be a causal") \
                or origin_response.startswith("\n\nThere could be a causal relationship") \
                or origin_response.startswith("There is a potential causal") \
                or origin_response.startswith("It is possible that there is a causal relationship") \
                or origin_response.startswith("It is possible that Event B caused Event A") \
                or origin_response.startswith("It is possible that Event B") \
                or origin_response.startswith("Event A and Event B suggest a causal relationship") \
                or origin_response.startswith("It is likely that there is a causal") \
                or origin_response.startswith("There may be a temporal relationship between Event A and") \
                or origin_response.startswith("It is possible that there could be a causal") \
                or origin_response.startswith("Event A is a possible precursor to Event B") \
                or origin_response.startswith("There could be a causal relationship between Event A and") \
                or origin_response.startswith("Possibly.") \
                or origin_response.startswith("Possibly, but not definitively. Infertility") \
                or origin_response.startswith("Possibly, but more information is needed to determine") \
                or origin_response.startswith("There is a possibility of a causal relationship between Event") \
                or origin_response.startswith("\n\nThere is a possible causal relationship") \
                or origin_response.endswith("Therefore, there is a causal relationship between Event A and Event B.") \
                or origin_response == " It is possible that washing her hands helped to reduce" \
                or origin_response == " Potential" \
                or origin_response == " Maybe":
            return 1
        if origin_response.startswith(" Not necessarily.") or origin_response == " It is not possible to determine if there is a" \
                or origin_response.startswith(" It depends. ") \
                or origin_response == " It is not possible to answer this question without more" \
                or origin_response == " It is difficult to determine if there is a causal" \
                or origin_response == " It is difficult to say definitively whether there is a" \
                or origin_response == "\n\nThere is no known causal relationship between Event" \
                or origin_response == "\n\nThere is no definitive answer to this question" \
                or origin_response == "The given events do not provide enough information to determine" \
                or origin_response == "\n\nThere is no causal relationship between Event A" \
                or origin_response == "Insufficient information" \
                or origin_response == "There may be a correlation between Event A and Event" \
                or origin_response == "There might be a correlation between Event A and Event" \
                or origin_response == "\n\nThere is not enough information to determine whether" \
                or origin_response == " It is unclear from the information given whether there is" \
                or origin_response == "Uncertain" \
                or origin_response == " It is impossible to determine whether or not there is" \
                or origin_response == " It is not certain." \
                or origin_response.startswith(" It is not possible to answer this question ") \
                or origin_response.startswith("The question appears to be") \
                or origin_response.startswith("It is not necessarily") \
                or origin_response.startswith("It is possible that Event B influenced Event A,") \
                or origin_response.startswith("There may or may not be a causal relationship betw") \
                or origin_response.startswith(" It is possible, but not necessarily.") \
                or origin_response.startswith("There is no") \
                or origin_response.startswith("There isn't necessarily a causal") \
                or origin_response.startswith("Uncertain.") \
                or origin_response.startswith("Not directly.") \
                or origin_response.startswith("Not necessarily") \
                or origin_response.startswith("There may be a correlation between") \
                or origin_response.startswith("Unclear") \
                or origin_response.startswith(" It is not possible to determine") \
                or origin_response.startswith("It is not possible to definitively") \
                or origin_response.startswith("There may not be a direct causal") \
                or origin_response.startswith("It is possible that there is no causal relationship") \
                or origin_response.startswith("There is a correlation between") \
                or origin_response.startswith("It is possible, but not necessarily") \
                or origin_response.startswith("Without additional information") \
                or origin_response.startswith(" It is difficult to determine") \
                or origin_response.startswith("It is impossible to determine") \
                or origin_response.startswith(" It is difficult to say") \
                or origin_response.startswith(" It is unclear ") \
                or origin_response.startswith("It is not possible to determine") \
                or origin_response.startswith("It depends on the context") \
                or origin_response.startswith("It is unlikely that there is a causal") \
                or origin_response.startswith("It is unclear whether there is a causal") \
                or origin_response.startswith("It is unclear") \
                or origin_response.startswith("There does not appear to be a causal") \
                or origin_response.startswith("As an AI language model") \
                or origin_response.startswith("Insufficient information is provided to determine if there is") \
                or origin_response.startswith("There is no direct causal") \
                or origin_response.startswith("Event A and Event B are related, but there") \
                or origin_response.startswith("There is no necessary causal") \
                or origin_response.startswith("There is no causal") \
                or origin_response.startswith(" It is not clear") \
                or origin_response.startswith("As an AI language model, I cannot determine") \
                or origin_response.startswith("There is not necessarily a causal") \
                or origin_response.startswith(" It is not known for certain") \
                or origin_response.startswith(" It is impossible to determine") \
                or origin_response.startswith(" It is not certain") \
                or origin_response.startswith("\n\nThere is no clear causal relationship") \
                or origin_response.startswith(" It is impossible to say") \
                or origin_response.startswith("There is no clear causal") \
                or origin_response.startswith("There is no apparent causal") \
                or origin_response.startswith("\n\nThere is no") \
                or origin_response.startswith(" It is unknown") \
                or origin_response.startswith(" There is no") \
                or origin_response.startswith("It is unlikely that there is a direct causal") \
                or origin_response.startswith("\n\nThere is not necessarily") \
                or origin_response.startswith("\n\nThere is no way to") \
                or origin_response.startswith("\n\nIt is difficult to say") \
                or origin_response.startswith(" It depends on ") \
                or origin_response.startswith("It is not clear") \
                or origin_response.startswith("It is difficult to determine") \
                or origin_response.startswith("There is not enough information to determine "):
            return 0
        for t in response_str:
            if t == "yes":
                return 1
            elif t == "no":
                return 0
            else:
                # return 1
                print([origin_response])
                assert False

    corpus = [json.loads(line) for line in open(file_name, 'r', encoding="utf-8")]
    pred = []
    gold = []
    for item in corpus:
        response1 = response_parse(item['gpt_response_1'])
        response2 = response_parse(item['gpt_response_2'])
        pred.extend([response1, response2])
        if item["label"] == 0:
            cur_answer = [1, 0]
        else:
            cur_answer = [0, 1]
        gold.extend(cur_answer)

    correct = 0
    total = 0
    assert len(gold) == len(pred)
    for g, p in zip(gold, pred):
        total += 1
        if g == p:
            correct += 1
    print("acc:", correct / total, total)
    acc = correct / total

    correct = 0
    total = 0
    assert len(gold) == len(pred)
    for g, p in zip(gold, pred):
        if g == 0:
            continue
        total += 1
        if g == p:
            correct += 1
            assert p == 1
    print("pos acc:", correct / total, total)
    pos_acc = correct / total

    correct = 0
    total = 0
    for g, p in zip(gold, pred):
        if g == 1:
            continue
        total += 1
        if g == p:
            correct += 1
    print("neg acc:", correct / total, total)
    neg_acc = correct / total

    print("& %.1f & %.1f & %.1f" % (pos_acc * 100, neg_acc * 100, acc * 100))


if __name__ == '__main__':
    """
    Please note that our prompt does not restrict the output format of the model.
    While in most cases, the model's output is structured and organized for the CD task,
    there are still a few instances where the output format may be less controlled.
    We do not include instructions about the output format in the prompt
    (e.g., "Please output the answer in the format of Option1/Option2.") 
    because doing so would slightly decrease the performance of ChatGPT.
    As a result, we manually annotated labels for different output formats,
    which are reflected in the if-else matching statements in the above code.
    

    For e-care, this process was performed on the test set.
    Therefore, Therefore, when conducting experiments on the training or development set,
    it is possible that the formats of a small portion of examples have not been covered by our code.
    If you are conducting experiments using the training or development set, you will need to insert addition if-else matching statements yourself.
    
    In short, it's no trouble at all.
    Even if you directly skip uncovered examples, the impact on the experimental results is minimal.
    """

    # multi-choice CD task
    # task1("output/e-CARE/task1_davinci003_std1/response.json")
    # task1("output/e-CARE/task1_davinci002_std1/response.json")
    # task1("output/e-CARE/task1_gpt-3.5-turbo_std1/response.json")
    # task1("output/gpt4/e-CARE_task1_sample1000/response.json")

    # task1("output/COPA/task1_davinci003_std1/response.json",
    #       test_file="data/COPA/copa.all.jsonl")
    # task1("output/COPA/task1_davinci002_std1/response.json",
    #   test_file="data/COPA/copa.all.jsonl")
    # task1("output/gpt4/copa_all/response.json",
    #   test_file="data/COPA/copa.all.jsonl")

    # task1("output/COPA/task1_gpt-3.5-turbo_std1/response.json",
    #   test_file="data/COPA/copa.all.jsonl")

    # binary classification CD task
    # eval_task1_in_pos_neg("output/COPA/task3_davinci003_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci002_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_gpt-3.5-turbo_std1/response.json")
    # eval_task1_in_pos_neg("output/COPA/task3_davinci002_std1/response.json")
    # eval_task1_in_pos_neg("output/COPA/task3_gpt-3.5-turbo_std1/response.json")
    # eval_task1_in_pos_neg("output/gpt4/e-CARE_task3_sample1000/response.json")
    # eval_task1_in_pos_neg("output/gpt4/copa_all_task3/response.json")

    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-1-2_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-2-2_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-1-4_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-2-4_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-4-2_std1/response.json")
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-COT-4-2_std1/response.json",is_COT=True)
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-COT-4-4_std1/response.json",is_COT=True)
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-COT-4-8_std1/response.json",is_COT=True)
    # eval_task1_in_pos_neg("output/e-CARE/task3_davinci003_ICL-COT-zero-shot_std1/response.json",is_COT=False)

    # causal explanation generation task
    # task2("output/e-CARE/task2_davinci003_std1/response.json")
    # task2("output/e-CARE/task2_gpt-3.5-turbo_std1/response.json")
    # task2("output/gpt4/e-CARE_task2_sample1000/response.json")

    ...
