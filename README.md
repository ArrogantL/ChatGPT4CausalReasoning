# ChatGPT Causal Reasoning Evaluation

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

### 2.2 ChatGPT with ICL or COT

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

