This folder contains the outputs of zero-shot ChatGPT on the ECI, CD and CEG tasks over five datasets.

For more flexible experimental settings, you can execute the provided code to easily explore the performance of ChatGPT:
-- ICL, COT: Please refer to CD_binary_classification.py and CD_multi_choice.py
-- More prompts: ECI_differ_causal_prompts.py, ECI_open_ended_generaton_prompts/chatgpt_ECI_openA123.py, ECI_open_ended_generaton_prompts/chatgpt_ECI_openB.py

We encourage further exploration of prompts for causal reasoning.
You can submit your results to us, and after code review, we will add them to this repository.


Due to access rate limitations imposed by OpenAI,
text-davinci-002 and text-davinci-003 are 10 times faster,
but the cost of gpt-3.5-turbo is lower.