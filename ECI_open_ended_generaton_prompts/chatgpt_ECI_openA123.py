# -*- coding: utf-8 -*-

import json
import os
import random
import shutil
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


def process(args):

    random.seed(args.seed)
    openai.api_key = args.openai_api_key

    # logger
    logger = get_logger(os.path.join(args.output_dir, "log"), is_console=True)
    logger.info(str(args))
    for key, value in args.__dict__.items():
        logger.info("%s :\t%s" % (key, str(value)))



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

                cur_all_items = [item]

                prompts = []
                for tid, t in enumerate(cur_all_items):
                    words = t["words"]
                    causal_triples = t["causal_triples"]
                    sent = " ".join(words)

                    answer_part = ""
                    if tid != len(cur_all_items) - 1:
                        if len(causal_triples) == 0:
                            answer_part = "None."
                        else:
                            for ctid, ((l1, r1), (l2, r2)) in enumerate(causal_triples):
                                e1 = " ".join(words[l1:r1])
                                e2 = " ".join(words[l2:r2])
                                if args.prompt_style == 1:
                                    answer_part += "\nCause: %s\nEffect: %s" % (e1, e2)
                                elif args.prompt_style == 2:
                                    answer_part += "(%s) cause (%s)" % (e1, e2)
                                    if ctid != len(causal_triples) - 1:
                                        answer_part += " AND "
                                elif args.prompt_style == 3:
                                    answer_part += "%s -> %s" % (e1, e2)
                                    if ctid != len(causal_triples) - 1:
                                        answer_part += " AND "
                    if args.prompt_style == 1:
                        prompts.append("Input: %s\nQuestion: List the cause-effect pairs in the input sentence.\nAnswer:%s" % (sent, answer_part))
                    elif args.prompt_style == 2:
                        prompts.append("""Input:%s
Question: If there is a causal relationship between two events in the input sentence, extract the causal pair at the word level. If there are multiple causal pairs, add AND between them, otherwise answer None. For example: (accuse of) cause (death) AND (kill) cause (death)
Answer:%s""" % (sent, answer_part))
                    elif args.prompt_style == 3:
                        prompts.append("""Input:%s
Question: Is there a token-level causal relationship in the sentence? If so, please extract it into this form: cause->effect. If there are multiple causal relationships, add AND between causal pairs, and display No if there is no causal relationship.
Answer:%s""" % (sent, answer_part))
                    else:
                        assert False
                ICL_prompt = "\n\n".join(prompts)


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

                # save response
                item["gpt_response"] = response
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info("\n\n" + ICL_prompt + response)


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
    set "prompt_style" to select open-ended prompt A.1/2/3
    
    set "gpt_type" to select the version of ChatGPT
    
    set "openai_api_key" as your OpenAI API key
    
    """
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int)
    parser.add_argument('--prompt_style', type=int, choices=[1, 2, 3])
    parser.add_argument('--input_file', type=str,default="ChatGPT4CausalReasoning/ECI_open_ended_generaton_prompts/data/ESC/intra_sent_causality.ECE.json")

    parser.add_argument('--gpt_type', type=str, choices=["gpt-3.5-turbo", "gpt-4", "text-davinci-002", "text-davinci-003"])

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--openai_api_key', type=str)


    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_max_input_items', type=int, default=-1)





    args = parser.parse_args()
    process(args)

