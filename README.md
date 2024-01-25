# ChatGPT Causal Reasoning Evaluation



This project contains the code of paper:

Is ChatGPT a Good Causal Reasoner? A Comprehensive Evaluation. [[Paper-ArXiv]](https://arxiv.org/abs/2305.07375)

Accepted by the Findings of EMNLP 2023.




## 1. Install



The code relies primarily on Python and the OpenAI API.

You need to execute the following command:

```python
# first, create conda environment
conda env create -f conda_environment.yml

conda activate <name of the env you created>

# second, install other python package
pip install -r pip_requirements.txt

# Finally, before starting the code, you will need to prepare an OpenAI API key.
```



## 2. Predict and Evaluation for ChatGPT

<font color=red>**For the following code files, we provide detailed usage instructions within each code file. They are easy to understand and can be reused to explore more experimental settings that interest you., <u>but require you to provide your own openAI API key</u></u>.**</font>

### 2.1 Zero-shot ChatGPT

```python
ECI:
    predict: ECI.py
    evaluate: ECI_compute_score.py
multi-choice CD:
    predict: CD_multi_choice.py
    evaluate: CD_and_CEG_compute_score.py
binary-classification CD:
    predict: CD_binary_classification.py
    evaluate: CD_and_CEG_compute_score.py
CEG:
    predict: CEG.py
    automic evaluate: CD_and_CEG_compute_score.py
    human evaluate: CEG_human_evaluation.xlsx
```

### 2.2 ChatGPT with ICL or CoT

```python
ECI:
    predict: ECI.py
    evaluate: ECI_compute_score.py
binary-classification CD:
    predict: CD_binary_classification.py
    evaluate: CD_and_CEG_compute_score.py
```

### 2.3 ChatGPT Using Prompts That Express the Causality in Different Ways

```python
predict: ECI_differ_causal_prompts.py
evaluate: ECI_differ_causal_prompts_compute_score.py
```

### 2.4 ChatGPT Using Prompts in the Form of Open-Ended Generation

```python
# To facilitate coding, we conducted this experiment in a dependent directory, with all the code and data located in the "ECI_open_ended_generation_prompts" folder.

predict:
    ECI_open_ended_generaton_prompts/chatgpt_ECI_openA123.py
   	ECI_open_ended_generaton_prompts/chatgpt_ECI_openB.py
evaluate: 
	ECI_open_ended_generaton_prompts/cal_prf_zero_shot_prompt_A1.py
    ECI_open_ended_generaton_prompts/cal_prf_zero_shot_prompt_A2.py
    ECI_open_ended_generaton_prompts/cal_prf_zero_shot_prompt_A3.py
    ECI_open_ended_generaton_prompts/cal_prf_zero_shot_prompt_B.py
```



## 3. Other Directories

```python
-- data # five datasets used in our experiments
-- output_by_ChatGPT # output of 4 versions of ChatGPT in ECI, CD and CEG task
-- ECI_open_ended_generaton_prompts # code and ChatGPT's output with the open-ended generation prompts in the ECI task
-- utils # other tool code
```





### Citation
If you find our reports benifit your research, please cite the following paper:

```ruby
@inproceedings{gao-etal-2023-chatgpt,
    title = "Is {C}hat{GPT} a Good Causal Reasoner? A Comprehensive Evaluation",
    author = "Gao, Jinglong  and
      Ding, Xiao  and
      Qin, Bing  and
      Liu, Ting",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.743",
    doi = "10.18653/v1/2023.findings-emnlp.743",
    pages = "11111--11126",
    abstract = "Causal reasoning ability is crucial for numerous NLP applications. Despite the impressive emerging ability of ChatGPT in various NLP tasks, it is unclear how well ChatGPT performs in causal reasoning. In this paper, we conduct the first comprehensive evaluation of the ChatGPT{'}s causal reasoning capabilities. Experiments show that ChatGPT is not a good causal reasoner, but a good causal interpreter. Besides, ChatGPT has a serious hallucination on causal reasoning, possibly due to the reporting biases between causal and non-causal relationships in natural language, as well as ChatGPT{'}s upgrading processes, such as RLHF. The In-Context Learning (ICL) and Chain-of-Thought (CoT) techniques can further exacerbate such causal hallucination. Additionally, the causal reasoning ability of ChatGPT is sensitive to the words used to express the causal concept in prompts, and close-ended prompts perform better than open-ended prompts. For events in sentences, ChatGPT excels at capturing explicit causality rather than implicit causality, and performs better in sentences with lower event density and smaller lexical distance between events.",
}
```
