# -*- coding: utf-8 -*-

import json
import os
import random
import traceback
from argparse import ArgumentParser

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from utils.log_kits import get_logger


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def completion_with_backoff_davinci(args, logger, **kwargs):
    try:
        response = openai.Completion.create(**kwargs)
        response_ls = [response["choices"][i]["text"] for i in range(len(kwargs["prompt"]))]
        return response_ls
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def completion_with_backoff_chat(args, logger, **kwargs):
    try:
        response = openai.ChatCompletion.create(**kwargs)
        return response
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e


def get_ESC_sent2events():
    sent2events = {}
    all_lines = list(
        open("ChatGPT4CausalReasoning/data/ESC/intra_sent_causality.json", 'r', encoding="utf-8").readlines())

    for line in all_lines:
        item = json.loads(line)
        words = item["words"]
        events = item["events"]
        if "#".join(words) not in sent2events:
            sent2events["#".join(words)] = []
        sent2events["#".join(words)].append(" ".join([words[t] for t in events[0]]))
        sent2events["#".join(words)].append(" ".join([words[t] for t in events[1]]))
    for key in sent2events:
        sent2events[key] = set(sent2events[key])
    return sent2events


def process(args):
    random.seed(args.seed)
    openai.api_key = args.openai_api_key

    # logger
    logger = get_logger(os.path.join(args.output_dir, "log"), is_console=True)
    logger.info(str(args))
    for key, value in args.__dict__.items():
        logger.info("%s :\t%s" % (key, str(value)))

    esc_sent2event_set = get_ESC_sent2events()
    with open(args.input_file, 'r', encoding="utf-8") as fin:
        with open(os.path.join(args.output_dir, "response.json"), 'a+', encoding="utf-8") as fout:
            all_item_list = [json.loads(line) for line in fin.readlines()]

            if args.debug_max_input_items != -1:
                all_item_list = debug_sample(all_item_list, args.debug_max_input_items)
                assert len(all_item_list) == args.debug_max_input_items
            uninfered_items = all_item_list
            random.shuffle(uninfered_items)

            for item_no, item in enumerate(uninfered_items):
                if args.debug and item_no == 2:
                    break
                logger.info("#" * 10 + "item " + str(item_no) + "#" * 10)

                words = item["words"]
                causal_triples = item["causal_triples"]
                sent = " ".join(words)
                if "#".join(words) in esc_sent2event_set:
                    all_cur_e = esc_sent2event_set["#".join(words)]
                else:
                    continue

                item["gpt_response"] = {}
                for cur_e in all_cur_e:
                    ICL_prompt = """Input: %s
Question: List the events in the input sentence that are causally related to the event "%s".
Answer:
1.""" % (sent, cur_e)

                    if args.gpt_type in ("text-davinci-002", "text-davinci-003"):
                        response = completion_with_backoff_davinci(args, logger,
                                                                   model=args.gpt_type,
                                                                   prompt=[ICL_prompt],
                                                                   temperature=0,
                                                                   max_tokens=1024,
                                                                   )[0]
                    elif args.gpt_type in ("gpt-3.5-turbo", "gpt-4"):
                        response = completion_with_backoff_chat(args, logger,
                                                                model=args.gpt_type,
                                                                messages=[
                                                                    {"role": "user", "content": ICL_prompt}
                                                                ],
                                                                temperature=0,
                                                                max_tokens=1024,
                                                                )["choices"][0]["message"]["content"]
                    else:
                        assert False
                    item["gpt_response"][cur_e] = response

                    logger.info("\n\n" + ICL_prompt + response)

                # save response
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def debug_sample(items, max_num, dtype="item"):
    pos, neg = [], []
    for item in items:
        if dtype == "item":
            causal_triples = item["causal_triples"]
        else:
            causal_triples = json.loads(item)["causal_triples"]
        if len(causal_triples) == 0:
            neg.append(item)
        else:
            pos.append(item)
    rate = max_num / len(items)
    pos_asked = int(len(pos) * rate)
    neg_asked = int(len(neg) * rate)
    pos_asked += max_num - pos_asked - neg_asked
    assert pos_asked + neg_asked == max_num
    random.shuffle(pos)
    random.shuffle(neg)
    output_items = pos[:pos_asked] + neg[:neg_asked]
    random.shuffle(output_items)
    return output_items


if __name__ == '__main__':
    """

    set "gpt_type" to select the version of ChatGPT

    set "openai_api_key" as your OpenAI API key

    """
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int)
    parser.add_argument('--input_file', type=str,default="ChatGPT4CausalReasoning/ECI_open_ended_generaton_prompts/data/ESC/intra_sent_causality.ECE.json")
    parser.add_argument('--gpt_type', type=str, choices=["gpt-3.5-turbo", "text-davinci-002", "text-davinci-003"], default="gpt-3.5-turbo")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_max_input_items', type=int, default=-1)
    args = parser.parse_args()
    process(args)
