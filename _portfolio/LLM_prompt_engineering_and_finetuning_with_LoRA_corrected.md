<a href="https://githubtocolab.com/Hbvsa/LLMs/blob/main/LLM_prompt_finetuning_RLHF_langchain_huggingface/LLM_prompt_engineering_and_finetuning_with_LoRA_corrected.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Prompt engineering for the summarization of dialogues using the FLAN-T5 model


# Table of Contents

- [ 1 - Summarize Dialogue without Prompt Engineering](#1)
- [ 2 - Summarize Dialogue with an Instruction Prompt](#2)
- [ 3 - Summarize Dialogue with One Shot and Few Shot Inference](#3)
  - [ 3.1 - One Shot Inference](#3.1)
  - [ 3.2 - Few Shot Inference](#3.2)
- [ 4 - Generative Configuration Parameters for Inference](#4)
- [ 5 - Finetuning the LLM](#5)
  - [ 5.1 - Tokenize the train, test and validation datasets with the instruction prompt](#5.1)
  - [ 5.2 - Full Finetuning](#5.2)
  - [ 5.3 - LoRA Finetuning](#5.3)
- [ 6 - Evalute the LoRA model versus baseline](#6)
  - [ 6.1 - Qualitatively](#6.1)
  - [ 6.2 - Quantitatively with ROGUE scores](#6.2)


```python
!pip install datasets
```

    Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (2.20.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets) (3.15.4)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.26.4)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (17.0.0)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets) (0.6)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.1.4)
    Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.32.3)
    Requirement already satisfied: tqdm>=4.66.3 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.5)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.4.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.16)
    Requirement already satisfied: fsspec<=2024.5.0,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from fsspec[http]<=2024.5.0,>=2023.1.0->datasets) (2024.5.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.10.1)
    Requirement already satisfied: huggingface-hub>=0.21.2 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.23.5)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (2.3.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (24.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.21.2->datasets) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (2024.7.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.1)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)



```python
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
```


```python
dataset = load_dataset("knkarthick/dialogsum")
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(


<a name='1'></a>
## 1 - Summarization without Prompt Engineering

Generating a summary of a dialogue with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face with the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) Hugging Face dataset. The models available in the Hugging Face `transformers` package can be found [here](https://huggingface.co/docs/transformers/index).

Explore the dataset examples


```python
dataset.shape
```




    {'train': (12460, 4), 'validation': (500, 4), 'test': (1500, 4)}




```python
dataset['train'][0]
```




    {'id': 'train_0',
     'dialogue': "#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?\n#Person2#: I found it would be a good idea to get a check-up.\n#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.\n#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?\n#Person1#: Well, the best way to avoid serious illnesses is to find out about them early. So try to come at least once a year for your own good.\n#Person2#: Ok.\n#Person1#: Let me see here. Your eyes and ears look fine. Take a deep breath, please. Do you smoke, Mr. Smith?\n#Person2#: Yes.\n#Person1#: Smoking is the leading cause of lung cancer and heart disease, you know. You really should quit.\n#Person2#: I've tried hundreds of times, but I just can't seem to kick the habit.\n#Person1#: Well, we have classes and some medications that might help. I'll give you more information before you leave.\n#Person2#: Ok, thanks doctor.",
     'summary': "Mr. Smith's getting a check-up, and Doctor Hawkins advises him to have one every year. Hawkins'll give some information about their classes and medications to help Mr. Smith quit smoking.",
     'topic': 'get a check-up'}




```python
dash_line = '-'.join('' for x in range(100))
for i, sample in enumerate(dataset['test']):
  print("Example",i)
  print(dash_line)
  print("Dialogue")
  print(dash_line)
  print(sample['dialogue'])
  print(dash_line)
  print("Summary")
  print(dash_line)
  print(sample['summary'])
  break
```

    Example 0
    ---------------------------------------------------------------------------------------------------
    Dialogue
    ---------------------------------------------------------------------------------------------------
    #Person1#: Ms. Dawson, I need you to take a dictation for me.
    #Person2#: Yes, sir...
    #Person1#: This should go out as an intra-office memorandum to all employees by this afternoon. Are you ready?
    #Person2#: Yes, sir. Go ahead.
    #Person1#: Attention all staff... Effective immediately, all office communications are restricted to email correspondence and official memos. The use of Instant Message programs by employees during working hours is strictly prohibited.
    #Person2#: Sir, does this apply to intra-office communications only? Or will it also restrict external communications?
    #Person1#: It should apply to all communications, not only in this office between employees, but also any outside communications.
    #Person2#: But sir, many employees use Instant Messaging to communicate with their clients.
    #Person1#: They will just have to change their communication methods. I don't want any - one using Instant Messaging in this office. It wastes too much time! Now, please continue with the memo. Where were we?
    #Person2#: This applies to internal and external communications.
    #Person1#: Yes. Any employee who persists in using Instant Messaging will first receive a warning and be placed on probation. At second offense, the employee will face termination. Any questions regarding this new policy may be directed to department heads.
    #Person2#: Is that all?
    #Person1#: Yes. Please get this memo typed up and distributed to all employees before 4 pm.
    ---------------------------------------------------------------------------------------------------
    Summary
    ---------------------------------------------------------------------------------------------------
    Ms. Dawson helps #Person1# to write a memo to inform every employee that they have to change the communication method and should not use Instant Messaging anymore.


Load the [FLAN-T5 model](https://huggingface.co/docs/transformers/model_doc/flan-t5)


```python
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```


    config.json:   0%|          | 0.00/1.40k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/990M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]


To perform encoding and decoding, you need to work with text in a tokenized form. Download the tokenizer for the FLAN-T5 model using `AutoTokenizer.from_pretrained()` method.


```python
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```


    tokenizer_config.json:   0%|          | 0.00/2.54k [00:00<?, ?B/s]



    spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.42M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/2.20k [00:00<?, ?B/s]


Test the tokenizer encoding and decoding a simple sentence:


```python
sentence = "Is skarner jungle good in this meta?"

sentence_encoded = tokenizer(sentence, return_tensors='pt')

sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0],
        skip_special_tokens=True
    )

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)
```

    ENCODED SENTENCE:
    tensor([   27,     7,     3,     7,  4031,   687, 19126,   207,    16,    48,
            10531,    58,     1])
    
    DECODED SENTENCE:
    Is skarner jungle good in this meta?


Without prompt engineering the models does not understand the task very well.


```python
for i, sample in enumerate(dataset['test']):

    dialogue = sample['dialogue']
    summary = sample['summary']

    inputs = tokenizer(dialogue, return_tensors='pt')
    summary_generated = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0]

    output = tokenizer.decode(summary_generated,skip_special_tokens=True)

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')

    if i ==0:#change according to how many examples you want
      break
```

    ---------------------------------------------------------------------------------------------------
    Example  1
    ---------------------------------------------------------------------------------------------------
    INPUT PROMPT:
    #Person1#: Ms. Dawson, I need you to take a dictation for me.
    #Person2#: Yes, sir...
    #Person1#: This should go out as an intra-office memorandum to all employees by this afternoon. Are you ready?
    #Person2#: Yes, sir. Go ahead.
    #Person1#: Attention all staff... Effective immediately, all office communications are restricted to email correspondence and official memos. The use of Instant Message programs by employees during working hours is strictly prohibited.
    #Person2#: Sir, does this apply to intra-office communications only? Or will it also restrict external communications?
    #Person1#: It should apply to all communications, not only in this office between employees, but also any outside communications.
    #Person2#: But sir, many employees use Instant Messaging to communicate with their clients.
    #Person1#: They will just have to change their communication methods. I don't want any - one using Instant Messaging in this office. It wastes too much time! Now, please continue with the memo. Where were we?
    #Person2#: This applies to internal and external communications.
    #Person1#: Yes. Any employee who persists in using Instant Messaging will first receive a warning and be placed on probation. At second offense, the employee will face termination. Any questions regarding this new policy may be directed to department heads.
    #Person2#: Is that all?
    #Person1#: Yes. Please get this memo typed up and distributed to all employees before 4 pm.
    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    Ms. Dawson helps #Person1# to write a memo to inform every employee that they have to change the communication method and should not use Instant Messaging anymore.
    ---------------------------------------------------------------------------------------------------
    MODEL GENERATION - WITHOUT PROMPT ENGINEERING:
    #Person1#: Ms. Dawson, I need you to take a dictation for me.
    


<a name='2'></a>
## 2 - Summarize Dialogue with an Instruction Prompt
Inject an instruction prompt to help the model understand the required task. We can see compared to the first example that the model did improve.




```python
for i, sample in enumerate(dataset['test']):

    dialogue = sample['dialogue']
    summary = sample['summary']

    prompt = f"""
Summarize the following dialogue.
{dialogue}
Summary:
"""

    inputs = tokenizer(prompt, return_tensors='pt')
    summary_generated = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0]

    output = tokenizer.decode(summary_generated,skip_special_tokens=True)

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')

    if i ==0:#change according to how many examples you want
      break
```

    ---------------------------------------------------------------------------------------------------
    Example  1
    ---------------------------------------------------------------------------------------------------
    INPUT PROMPT:
    
    Summarize the following dialogue.
    #Person1#: Ms. Dawson, I need you to take a dictation for me.
    #Person2#: Yes, sir...
    #Person1#: This should go out as an intra-office memorandum to all employees by this afternoon. Are you ready?
    #Person2#: Yes, sir. Go ahead.
    #Person1#: Attention all staff... Effective immediately, all office communications are restricted to email correspondence and official memos. The use of Instant Message programs by employees during working hours is strictly prohibited.
    #Person2#: Sir, does this apply to intra-office communications only? Or will it also restrict external communications?
    #Person1#: It should apply to all communications, not only in this office between employees, but also any outside communications.
    #Person2#: But sir, many employees use Instant Messaging to communicate with their clients.
    #Person1#: They will just have to change their communication methods. I don't want any - one using Instant Messaging in this office. It wastes too much time! Now, please continue with the memo. Where were we?
    #Person2#: This applies to internal and external communications.
    #Person1#: Yes. Any employee who persists in using Instant Messaging will first receive a warning and be placed on probation. At second offense, the employee will face termination. Any questions regarding this new policy may be directed to department heads.
    #Person2#: Is that all?
    #Person1#: Yes. Please get this memo typed up and distributed to all employees before 4 pm.
    Summary:
    
    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    Ms. Dawson helps #Person1# to write a memo to inform every employee that they have to change the communication method and should not use Instant Messaging anymore.
    ---------------------------------------------------------------------------------------------------
    MODEL GENERATION - ZERO SHOT:
    The memo will go out to all employees by this afternoon.
    


<a name='3'></a>
## 3 - Summarize Dialogue with One Shot and Few Shot Inference
**One shot and few shot inference** is a method used to provide the LLM with examples of the task we require it to perform. This is also called "in-context learning" which gives the model the context to understand the specific task.


<a name='3.1'></a>
### 3.1 - One Shot Inference

Function which takes `example_samples` and generates a prompt with those completed examples. At the end of the examples adds the dialogue you want to summarize from `sample_to_summarize`.


```python
def make_prompt(example_samples, sample_to_summarize):



    #Initialize prompt
    prompt = ''

    #Add examples
    for index in example_samples:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt += f"""
Dialogue:
{dialogue}
Summarize the dialogue.
{summary}
"""
    #Add the dialogue of the sample you want to summarize and the instruction
    dialogue = dataset['test'][sample_to_summarize]['dialogue']

    prompt += f"""
Dialogue:
{dialogue}
Summarize the dialogue.
"""
    # return all the examples plus the dialogue you want to summarize
    return prompt
```

Construct the prompt to perform one shot inference:


```python
example_samples = [10]
sample_to_summarize = 120
one_shot_prompt = make_prompt(example_samples, sample_to_summarize)
print(one_shot_prompt)
```

    
    Dialogue:
    #Person1#: Happy Birthday, this is for you, Brian.
    #Person2#: I'm so happy you remember, please come in and enjoy the party. Everyone's here, I'm sure you have a good time.
    #Person1#: Brian, may I have a pleasure to have a dance with you?
    #Person2#: Ok.
    #Person1#: This is really wonderful party.
    #Person2#: Yes, you are always popular with everyone. and you look very pretty today.
    #Person1#: Thanks, that's very kind of you to say. I hope my necklace goes with my dress, and they both make me look good I feel.
    #Person2#: You look great, you are absolutely glowing.
    #Person1#: Thanks, this is a fine party. We should have a drink together to celebrate your birthday
    Summarize the dialogue.
    #Person1# attends Brian's birthday party. Brian thinks #Person1# looks great and charming.
    
    Dialogue:
    #Person1#: Hello, I bought the pendant in your shop, just before. 
    #Person2#: Yes. Thank you very much. 
    #Person1#: Now I come back to the hotel and try to show it to my friend, the pendant is broken, I'm afraid. 
    #Person2#: Oh, is it? 
    #Person1#: Would you change it to a new one? 
    #Person2#: Yes, certainly. You have the receipt? 
    #Person1#: Yes, I do. 
    #Person2#: Then would you kindly come to our shop with the receipt by 10 o'clock? We will replace it. 
    #Person1#: Thank you so much. 
    Summarize the dialogue.
    


Now pass this prompt to perform the one shot inference:


```python
summary = dataset['test'][sample_to_summarize]['summary']
inputs = tokenizer(one_shot_prompt, return_tensors='pt')
generated_summary = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0]

output = tokenizer.decode(generated_summary, skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')
```

    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    #Person1# wants to change the broken pendant in #Person2#'s shop.
    
    ---------------------------------------------------------------------------------------------------
    MODEL GENERATION - ONE SHOT:
    Person1 bought a pendant in your shop. The pendant is broken. Then Person2 will replace it.


<a name='3.2'></a>
### 3.2 - Few Shot Inference

The performance of the model by including extra examples in this case did not have an effect although in general is a good practice.


```python
example_samples = [10, 20,30]
sample_to_summarize = 120
few_shot_prompt = make_prompt(example_samples, sample_to_summarize)
print(few_shot_prompt)
```

    
    Dialogue:
    #Person1#: Happy Birthday, this is for you, Brian.
    #Person2#: I'm so happy you remember, please come in and enjoy the party. Everyone's here, I'm sure you have a good time.
    #Person1#: Brian, may I have a pleasure to have a dance with you?
    #Person2#: Ok.
    #Person1#: This is really wonderful party.
    #Person2#: Yes, you are always popular with everyone. and you look very pretty today.
    #Person1#: Thanks, that's very kind of you to say. I hope my necklace goes with my dress, and they both make me look good I feel.
    #Person2#: You look great, you are absolutely glowing.
    #Person1#: Thanks, this is a fine party. We should have a drink together to celebrate your birthday
    Summarize the dialogue.
    #Person1# attends Brian's birthday party. Brian thinks #Person1# looks great and charming.
    
    Dialogue:
    #Person1#: What's wrong with you? Why are you scratching so much?
    #Person2#: I feel itchy! I can't stand it anymore! I think I may be coming down with something. I feel lightheaded and weak.
    #Person1#: Let me have a look. Whoa! Get away from me!
    #Person2#: What's wrong?
    #Person1#: I think you have chicken pox! You are contagious! Get away! Don't breathe on me!
    #Person2#: Maybe it's just a rash or an allergy! We can't be sure until I see a doctor.
    #Person1#: Well in the meantime you are a biohazard! I didn't get it when I was a kid and I've heard that you can even die if you get it as an adult!
    #Person2#: Are you serious? You always blow things out of proportion. In any case, I think I'll go take an oatmeal bath.
    Summarize the dialogue.
    #Person1# thinks #Person2# has chicken pox and warns #Person2# about the possible hazards but #Person2# thinks it will be fine.
    
    Dialogue:
    #Person1#: Where are you going for your trip?
    #Person2#: I think Hebei is a good place.
    #Person1#: But I heard the north of China are experiencing severe sandstorms!
    #Person2#: Really?
    #Person1#: Yes, it's said that Hebes was experiencing six degree strong winds.
    #Person2#: How do these storms affect the people who live in these areas?
    #Person1#: The report said the number of people with respiratory tract infections tended to rise after sandstorms. The sand gets into people's noses and throats and creates irritation.
    #Person2#: It sounds that sandstorms are trouble for everybody!
    #Person1#: You are quite right.
    Summarize the dialogue.
    #Person2# plans to have a trip in Hebei but #Person1# says there are sandstorms in there.
    
    Dialogue:
    #Person1#: Hello, I bought the pendant in your shop, just before. 
    #Person2#: Yes. Thank you very much. 
    #Person1#: Now I come back to the hotel and try to show it to my friend, the pendant is broken, I'm afraid. 
    #Person2#: Oh, is it? 
    #Person1#: Would you change it to a new one? 
    #Person2#: Yes, certainly. You have the receipt? 
    #Person1#: Yes, I do. 
    #Person2#: Then would you kindly come to our shop with the receipt by 10 o'clock? We will replace it. 
    #Person1#: Thank you so much. 
    Summarize the dialogue.
    


Now pass this prompt to perform a few shot inference:


```python
summary = dataset['test'][sample_to_summarize]['summary']
inputs = tokenizer(few_shot_prompt, return_tensors='pt')
generated_summary = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0]

output = tokenizer.decode(generated_summary,skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
```

    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    #Person1# wants to change the broken pendant in #Person2#'s shop.
    
    ---------------------------------------------------------------------------------------------------
    MODEL GENERATION - FEW SHOT:
    Person1 bought a pendant in your shop. The pendant is broken. Then Person2 will replace it.


<a name='4'></a>
## 4 - Generation parameters

Changing the generation parameters. The temperature controls how the probability distribution for the generation of tokens is being distributed. A higher temperature increases lower probability tokens for more creativity but also hallucinations.


```python
#generation_config = GenerationConfig(max_new_tokens=50)
# generation_config = GenerationConfig(max_new_tokens=10)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
model_generation = model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
    )[0]

output = tokenizer.decode(model_generation,skip_special_tokens=True)

print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
```

    ---------------------------------------------------------------------------------------------------
    MODEL GENERATION - FEW SHOT:
    Person1 wants to return the pendant she bought in your shop.
    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    #Person1# wants to change the broken pendant in #Person2#'s shop.
    


<a name='3'></a>
## 5 - Finetuning the LLM


```python
!pip install evaluate
```

    Requirement already satisfied: evaluate in /usr/local/lib/python3.10/dist-packages (0.4.2)
    Requirement already satisfied: datasets>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from evaluate) (2.20.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from evaluate) (1.26.4)
    Requirement already satisfied: dill in /usr/local/lib/python3.10/dist-packages (from evaluate) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from evaluate) (2.1.4)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from evaluate) (2.32.3)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.10/dist-packages (from evaluate) (4.66.5)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from evaluate) (3.4.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from evaluate) (0.70.16)
    Requirement already satisfied: fsspec>=2021.05.0 in /usr/local/lib/python3.10/dist-packages (from fsspec[http]>=2021.05.0->evaluate) (2024.5.0)
    Requirement already satisfied: huggingface-hub>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from evaluate) (0.23.5)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from evaluate) (24.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets>=2.0.0->evaluate) (3.15.4)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=2.0.0->evaluate) (17.0.0)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets>=2.0.0->evaluate) (0.6)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets>=2.0.0->evaluate) (3.10.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets>=2.0.0->evaluate) (6.0.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.7.0->evaluate) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->evaluate) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->evaluate) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->evaluate) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->evaluate) (2024.7.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->evaluate) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->evaluate) (2024.1)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->evaluate) (2024.1)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (2.3.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (24.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.0.0->evaluate) (4.0.3)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->evaluate) (1.16.0)



```python
from transformers import TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
```


```python
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map = 'cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
```


```python
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))
```

    trainable model parameters: 247577856
    all model parameters: 247577856
    percentage of trainable model parameters: 100.00%


Test the model with the zero shot inferencing. You can see that the model struggles to summarize the dialogue compared to the baseline summary, but it does pull out some important information from the text which indicates the model can be fine-tuned to the task at hand.


```python
index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"].to('cuda'),
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')
```

    ---------------------------------------------------------------------------------------------------
    INPUT PROMPT:
    
    Summarize the following conversation.
    
    #Person1#: Have you considered upgrading your system?
    #Person2#: Yes, but I'm not sure what exactly I would need.
    #Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
    #Person2#: That would be a definite bonus.
    #Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
    #Person2#: How can we do that?
    #Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
    #Person2#: No.
    #Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
    #Person2#: That sounds great. Thanks.
    
    Summary:
    
    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    #Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
    
    ---------------------------------------------------------------------------------------------------
    MODEL GENERATION - ZERO SHOT:
    #Person1#: I'm thinking of upgrading my computer.


<a name='5.1'></a>
###5.1 -Tokenize the train, test and validation datasets with the instruction prompt


```python
def tokenize_function(sample):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    #Add the instruction prompts
    prompt = [start_prompt + dialogue + end_prompt for dialogue in sample["dialogue"]]
    #Tokenize the inputs and labels/responses
    sample['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    sample['labels'] = tokenizer(sample["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return sample

#the map function distributes the function across all samples across all splits
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
```


    Map:   0%|          | 0/500 [00:00<?, ? examples/s]


To save time for this tutorial demonstration filter the dataset to a smaller one


```python
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
```


    Filter:   0%|          | 0/12460 [00:00<?, ? examples/s]



    Filter:   0%|          | 0/500 [00:00<?, ? examples/s]



    Filter:   0%|          | 0/1500 [00:00<?, ? examples/s]



```python
print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)
```

    Shapes of the datasets:
    Training: (125, 2)
    Validation: (5, 2)
    Test: (15, 2)
    DatasetDict({
        train: Dataset({
            features: ['input_ids', 'labels'],
            num_rows: 125
        })
        validation: Dataset({
            features: ['input_ids', 'labels'],
            num_rows: 5
        })
        test: Dataset({
            features: ['input_ids', 'labels'],
            num_rows: 15
        })
    })


<a name='5.2'></a>
###5.2 - Full finetuning


```python
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)
```

    max_steps is given, it will override any value given in num_train_epochs



```python
trainer.train()
```



    <div>

      <progress value='1' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1/1 00:11, Epoch 0/1]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>49.000000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=1, training_loss=49.0, metrics={'train_runtime': 15.3607, 'train_samples_per_second': 0.521, 'train_steps_per_second': 0.065, 'total_flos': 5478058819584.0, 'train_loss': 49.0, 'epoch': 0.0625})



Since doing a full training session takes way too much time, instead load the trained model directly with the following next lines of code. We shall call the fully finetuned model the instruct model.


```python
instruct_model_name='truocpham/flan-dialogue-summary-checkpoint'
```


```python
instruct_model = AutoModelForSeq2SeqLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map='cuda')
```

<a name='5.3'></a>
###5.3 - LoRA finetuning

Now, let's perform **Parameter Efficient Fine-Tuning (PEFT)** fine-tuning as opposed to "full fine-tuning" as you did above. PEFT is a form of instruction fine-tuning that is much more efficient than full fine-tuning - with comparable evaluation results as you will see soon.

PEFT is a generic term that includes **Low-Rank Adaptation (LoRA)** and prompt tuning (which is NOT THE SAME as prompt engineering!). In most cases, when someone says PEFT, they typically mean LoRA. LoRA, at a very high level, allows the user to fine-tune their model using fewer compute resources (in some cases, a single GPU). After fine-tuning for a specific task, use case, or tenant with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges. This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).  

That said, at inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request.  The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.

Using LoRa is also important to prevent Catastrophic Forgetting which makes the original LLM to lose its previous knowledge.


```python
!pip install peft
```

    Requirement already satisfied: peft in /usr/local/lib/python3.10/dist-packages (0.12.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from peft) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from peft) (24.1)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from peft) (5.9.5)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from peft) (6.0.1)
    Requirement already satisfied: torch>=1.13.0 in /usr/local/lib/python3.10/dist-packages (from peft) (2.3.1+cu121)
    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (from peft) (4.42.4)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from peft) (4.66.5)
    Requirement already satisfied: accelerate>=0.21.0 in /usr/local/lib/python3.10/dist-packages (from peft) (0.32.1)
    Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from peft) (0.4.4)
    Requirement already satisfied: huggingface-hub>=0.17.0 in /usr/local/lib/python3.10/dist-packages (from peft) (0.23.5)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.17.0->peft) (3.15.4)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.17.0->peft) (2024.5.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.17.0->peft) (2.32.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.17.0->peft) (4.12.2)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (1.13.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (3.1.4)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (12.1.105)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (2.20.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (12.1.105)
    Requirement already satisfied: triton==2.3.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.0->peft) (2.3.1)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.13.0->peft) (12.6.20)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers->peft) (2024.5.15)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers->peft) (0.19.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.13.0->peft) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.17.0->peft) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.17.0->peft) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.17.0->peft) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.17.0->peft) (2024.7.4)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.13.0->peft) (1.3.0)



```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)
```


```python
peft_model = get_peft_model(original_model,
                            lora_config)
print(print_number_of_trainable_model_parameters(peft_model))
```

    trainable model parameters: 3538944
    all model parameters: 251116800
    percentage of trainable model parameters: 1.41%



```python
output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)
```


```python
peft_trainer.train()
```



    <div>

      <progress value='16' max='16' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [16/16 00:36, Epoch 1/1]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>





    TrainOutput(global_step=16, training_loss=30.0859375, metrics={'train_runtime': 38.7296, 'train_samples_per_second': 3.228, 'train_steps_per_second': 0.413, 'total_flos': 86953623552000.0, 'train_loss': 30.0859375, 'epoch': 1.0})




```python
peft_model_path=r'Desktop/GithubLLMs/LLMs/peft-dialogue-summary-checkpoint-local'

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
```




    ('Desktop/GithubLLMs/LLMs/peft-dialogue-summary-checkpoint-local\\tokenizer_config.json',
     'Desktop/GithubLLMs/LLMs/peft-dialogue-summary-checkpoint-local\\special_tokens_map.json',
     'Desktop/GithubLLMs/LLMs/peft-dialogue-summary-checkpoint-local\\tokenizer.json')



That training was performed on a subset of data. So let's load the fully trained LoRa model from a previous saved model.


```python
from peft import PeftModel, PeftConfig

# huggin face alternative
peft_dialogue_summary_checkpoint = 'intotheverse/peft-dialogue-summary-checkpoint'

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(peft_model_base,
                                       peft_dialogue_summary_checkpoint, #'./peft-dialogue-summary-checkpoint-from-s3/',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)
```

<a name='6'></a>
##6 - Evaluting the finetuned models

<a name='6.1'></a>
###6.1 - Qualitatively


```python
index = 200
dialogue = dataset['test'][index]['dialogue']
baseline_human_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{baseline_human_summary}')
print(dash_line)
print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')
print(dash_line)
print(f'PEFT MODEL: {peft_model_text_output}')

```

    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    #Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
    ---------------------------------------------------------------------------------------------------
    ORIGINAL MODEL:
    #Person1#: I'm thinking of upgrading my computer.
    ---------------------------------------------------------------------------------------------------
    INSTRUCT MODEL:
    #Person1# suggests #Person2# upgrading #Person2#'s system, hardware, and CD-ROM drive. #Person2# thinks it's great.
    ---------------------------------------------------------------------------------------------------
    PEFT MODEL: #Person1# recommends adding a painting program to #Person2#'s software and upgrading hardware. #Person2# also wants to upgrade the hardware because it's outdated now.


<a name='6.2'></a>
###6.2 - Quantitavely with ROGUE scores

Using 10 examples to save time


```python
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

    human_baseline_text_output = human_baseline_summaries[idx]

    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)
    instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))

df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries', 'peft_model_summaries'])
df
```





  <div id="df-d17e8db8-b068-4564-82fa-b334abb75ee5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>human_baseline_summaries</th>
      <th>original_model_summaries</th>
      <th>instruct_model_summaries</th>
      <th>peft_model_summaries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ms. Dawson helps #Person1# to write a memo to ...</td>
      <td>#Person1#: I need to take a dictation for you.</td>
      <td>#Person1# asks Ms. Dawson to take a dictation ...</td>
      <td>#Person1# asks Ms. Dawson to take a dictation ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>In order to prevent employees from wasting tim...</td>
      <td>#Person1#: I need to take a dictation for you.</td>
      <td>#Person1# asks Ms. Dawson to take a dictation ...</td>
      <td>#Person1# asks Ms. Dawson to take a dictation ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ms. Dawson takes a dictation for #Person1# abo...</td>
      <td>#Person1#: I need to take a dictation for you.</td>
      <td>#Person1# asks Ms. Dawson to take a dictation ...</td>
      <td>#Person1# asks Ms. Dawson to take a dictation ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#Person2# arrives late because of traffic jam....</td>
      <td>The traffic jam at the Carrefour intersection ...</td>
      <td>#Person2# got stuck in traffic again. #Person1...</td>
      <td>#Person2# got stuck in traffic and #Person1# s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#Person2# decides to follow #Person1#'s sugges...</td>
      <td>The traffic jam at the Carrefour intersection ...</td>
      <td>#Person2# got stuck in traffic again. #Person1...</td>
      <td>#Person2# got stuck in traffic and #Person1# s...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>#Person2# complains to #Person1# about the tra...</td>
      <td>The traffic jam at the Carrefour intersection ...</td>
      <td>#Person2# got stuck in traffic again. #Person1...</td>
      <td>#Person2# got stuck in traffic and #Person1# s...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>#Person1# tells Kate that Masha and Hero get d...</td>
      <td>Masha and Hero are getting divorced.</td>
      <td>Masha and Hero are getting divorced. Kate can'...</td>
      <td>Kate tells #Person2# Masha and Hero are gettin...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>#Person1# tells Kate that Masha and Hero are g...</td>
      <td>Masha and Hero are getting divorced.</td>
      <td>Masha and Hero are getting divorced. Kate can'...</td>
      <td>Kate tells #Person2# Masha and Hero are gettin...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>#Person1# and Kate talk about the divorce betw...</td>
      <td>Masha and Hero are getting divorced.</td>
      <td>Masha and Hero are getting divorced. Kate can'...</td>
      <td>Kate tells #Person2# Masha and Hero are gettin...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>#Person1# and Brian are at the birthday party ...</td>
      <td>#Person1#: Happy birthday, Brian. #Person2#: I...</td>
      <td>Brian's birthday is coming. #Person1# invites ...</td>
      <td>Brian remembers his birthday and invites #Pers...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d17e8db8-b068-4564-82fa-b334abb75ee5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d17e8db8-b068-4564-82fa-b334abb75ee5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d17e8db8-b068-4564-82fa-b334abb75ee5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-256529f0-e486-4993-b0a4-998b9130ee7e">
  <button class="colab-df-quickchart" onclick="quickchart('df-256529f0-e486-4993-b0a4-998b9130ee7e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-256529f0-e486-4993-b0a4-998b9130ee7e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_cd66ab7d-aead-4e85-8130-d8e534c771bf">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_cd66ab7d-aead-4e85-8130-d8e534c771bf button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
!pip install rouge_score
```

    Collecting rouge_score
      Downloading rouge_score-0.1.2.tar.gz (17 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from rouge_score) (1.4.0)
    Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (from rouge_score) (3.8.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from rouge_score) (1.26.4)
    Requirement already satisfied: six>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from rouge_score) (1.16.0)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk->rouge_score) (8.1.7)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk->rouge_score) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk->rouge_score) (2024.5.15)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk->rouge_score) (4.66.5)
    Building wheels for collected packages: rouge_score
      Building wheel for rouge_score (setup.py) ... [?25l[?25hdone
      Created wheel for rouge_score: filename=rouge_score-0.1.2-py3-none-any.whl size=24935 sha256=9908596001ef07101e1e0872cb7c28e92be4ee7f8025ec9184df226954d0364f
      Stored in directory: /root/.cache/pip/wheels/5f/dd/89/461065a73be61a532ff8599a28e9beef17985c9e9c31e541b4
    Successfully built rouge_score
    Installing collected packages: rouge_score
    Successfully installed rouge_score-0.1.2



```python
rouge = evaluate.load('rouge')

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)
print('PEFT MODEL:')
print(peft_model_results)
```

    ORIGINAL MODEL:
    {'rouge1': 0.23884559093833285, 'rouge2': 0.11535720375106562, 'rougeL': 0.21714203657752046, 'rougeLsum': 0.2175800707655546}
    INSTRUCT MODEL:
    {'rouge1': 0.41026607717457186, 'rouge2': 0.17840645241958838, 'rougeL': 0.2977022096267017, 'rougeLsum': 0.2987374187518165}
    PEFT MODEL:
    {'rouge1': 0.3725351062275605, 'rouge2': 0.12138811933618107, 'rougeL': 0.27620639623170606, 'rougeLsum': 0.2758134870822362}



```python
print("Absolute percentage improvement of PEFT MODEL over HUMAN BASELINE")

improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
```

    Absolute percentage improvement of PEFT MODEL over HUMAN BASELINE
    rouge1: 13.37%
    rouge2: 0.60%
    rougeL: 5.91%
    rougeLsum: 5.82%



```python
print("Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL")

improvement = (np.array(list(peft_model_results.values())) - np.array(list(instruct_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
```

    Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL
    rouge1: -3.77%
    rouge2: -5.70%
    rougeL: -2.15%
    rougeLsum: -2.29%


#### Results using all samples


```python
results = pd.read_csv("https://raw.githubusercontent.com/Hbvsa/LLMs/main/LLM_prompt_finetuning_RLHF_langchain_huggingface/dialogue-summary-training-results.csv")
```


```python
human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values
peft_model_summaries     = results['peft_model_summaries'].values

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)
print('PEFT MODEL:')
print(peft_model_results)
```

    ORIGINAL MODEL:
    {'rouge1': 0.2334158581572823, 'rouge2': 0.07603964187010573, 'rougeL': 0.20145520923859048, 'rougeLsum': 0.20145899339006135}
    INSTRUCT MODEL:
    {'rouge1': 0.42161291557556113, 'rouge2': 0.18035380596301792, 'rougeL': 0.3384439349963909, 'rougeLsum': 0.33835653595561666}
    PEFT MODEL:
    {'rouge1': 0.40810631575616746, 'rouge2': 0.1633255794568712, 'rougeL': 0.32507074586565354, 'rougeLsum': 0.3248950182867091}



```python
print("Absolute percentage improvement of PEFT MODEL over HUMAN BASELINE")

improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
```

    Absolute percentage improvement of PEFT MODEL over HUMAN BASELINE
    rouge1: 17.47%
    rouge2: 8.73%
    rougeL: 12.36%
    rougeLsum: 12.34%



```python
print("Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL")

improvement = (np.array(list(peft_model_results.values())) - np.array(list(instruct_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
```

    Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL
    rouge1: -1.35%
    rouge2: -1.70%
    rougeL: -1.34%
    rougeLsum: -1.35%


Here you see a small percentage decrease in the ROUGE metrics vs. full fine-tuned. However, the training requires much less computing and memory resources (often just a single GPU).
