<a href="https://githubtocolab.com/Hbvsa/LLMs/blob/main/Prompting_examples_and_simple_chat_bot/prompt_tactics_examples.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Table of Contents

- [ 1 - Guidelines For Prompting](#1)
- [ 2 - Summarizing](#2)
- [ 3 - Inferring](#3)
- [ 4 - Transforming](#4)



## Setup
#### Load the API key and relevant Python libraries.


```python
!pip install openai
```

    Collecting openai
      Downloading openai-1.40.1-py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3/dist-packages (from openai) (1.7.0)
    Collecting httpx<1,>=0.23.0 (from openai)
      Downloading httpx-0.27.0-py3-none-any.whl.metadata (7.2 kB)
    Collecting jiter<1,>=0.4.0 (from openai)
      Downloading jiter-0.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from openai) (2.8.2)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.4)
    Requirement already satisfied: typing-extensions<5,>=4.11 in /usr/local/lib/python3.10/dist-packages (from openai) (4.12.2)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (3.7)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (1.2.2)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (2024.7.4)
    Collecting httpcore==1.* (from httpx<1,>=0.23.0->openai)
      Downloading httpcore-1.0.5-py3-none-any.whl.metadata (20 kB)
    Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx<1,>=0.23.0->openai)
      Downloading h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)
    Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1.9.0->openai) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1.9.0->openai) (2.20.1)
    Downloading openai-1.40.1-py3-none-any.whl (360 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m360.4/360.4 kB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading httpx-0.27.0-py3-none-any.whl (75 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m4.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading httpcore-1.0.5-py3-none-any.whl (77 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m4.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jiter-0.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (318 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m318.9/318.9 kB[0m [31m11.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading h11-0.14.0-py3-none-any.whl (58 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: jiter, h11, httpcore, httpx, openai
    Successfully installed h11-0.14.0 httpcore-1.0.5 httpx-0.27.0 jiter-0.5.0 openai-1.40.1



```python
import openai
import os
os.environ['OPENAI_API_KEY'] = 'your_key'
```


```python
client = openai.OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content
```

<a name='1'></a>
# Guidelines for Prompting


## Prompting Principles
- **Principle 1: Write clear and specific instructions**
- **Principle 2: Give the model time to “think”**

### Tactics

#### Tactic 1: Use delimiters to clearly indicate distinct parts of the input
- Delimiters can be anything like: \`\`\`, """, < >, `<tag> </tag>`, `:`


```python
text = f"""
You should express what you want a model to do by \
providing instructions that are as clear and \
specific as you can possibly make them. \
This will guide the model towards the desired output, \
and reduce the chances of receiving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short prompt. \
In many cases, longer prompts provide more clarity \
and context for the model, which can lead to \
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple angle brackets\
into a single sentence.
<<<{text}>>>
"""
response = get_completion(prompt)
print(response)
```

    Clear and specific instructions are essential for guiding a model towards the desired output and reducing the chances of receiving irrelevant or incorrect responses, with longer prompts often providing more clarity and context for more detailed and relevant outputs.


#### Tactic 2: Ask for a structured output
- JSON, HTML


```python
prompt = f"""
Generate a list of three made-up book titles along \
with their authors and genres.
Provide them in JSON format with the following keys:
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)
```

    [
        {
            "book_id": 1,
            "title": "The Midnight Garden",
            "author": "Elena Rivers",
            "genre": "Fantasy"
        },
        {
            "book_id": 2,
            "title": "Echoes of the Past",
            "author": "Nathan Black",
            "genre": "Mystery"
        },
        {
            "book_id": 3,
            "title": "Whispers in the Wind",
            "author": "Samantha Reed",
            "genre": "Romance"
        }
    ]


#### Tactic 3: Ask the model to check whether conditions are satisfied


```python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \
water boiling. While that's happening, \
grab a cup and put a tea bag in it. Once the water is \
hot enough, just pour it over the tea bag. \
Let it sit for a bit so the tea can steep. After a \
few minutes, take out the tea bag. If you \
like, you can add some sugar or milk to taste. \
And that's it! You've got yourself a delicious \
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes.
If it contains a sequence of instructions, \
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"

"""
response = get_completion(prompt)
print("Completion for Text 1:")
print(response)
```

    Completion for Text 1:
    
    Step 1 - Get some water boiling.
    Step 2 - Grab a cup and put a tea bag in it.
    Step 3 - Pour the hot water over the tea bag.
    Step 4 - Let the tea steep for a few minutes.
    Step 5 - Remove the tea bag.
    Step 6 - Add sugar or milk to taste.
    Step 7 - Enjoy your delicious cup of tea.



```python
text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \
walk in the park. The flowers are blooming, and the \
trees are swaying gently in the breeze. People \
are out and about, enjoying the lovely weather. \
Some are having picnics, while others are playing \
games or simply relaxing on the grass. It's a \
perfect day to spend time outdoors and appreciate the \
beauty of nature.
"""
prompt = f"""
You will be provided with text delimited by triple quotes.
If it contains a sequence of instructions, \
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 2:")
print(response)
```

    Completion for Text 2:
    No steps provided.


#### Tactic 4: "Few-shot" prompting


```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \
valley flows from a modest spring; the \
grandest symphony originates from a single note; \
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
```

    <grandparent>: Just as the tallest tree withstands the strongest winds, and the brightest star shines through the darkest night, resilience is the ability to bounce back from adversity and keep moving forward. It is the strength to endure and overcome challenges with grace and determination.


### Principle 2: Give the model time to “think”

#### Tactic 1: Specify the steps required to complete a task


```python
text = f"""
In a charming village, siblings Jack and Jill set out on \
a quest to fetch water from a hilltop \
well. As they climbed, singing joyfully, misfortune \
struck—Jack tripped on a stone and tumbled \
down the hill, with Jill following suit. \
Though slightly battered, the pair returned home to \
comforting embraces. Despite the mishap, \
their adventurous spirits remained undimmed, and they \
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions:
1 - Summarize the following text delimited by triple \
quotes with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
\"\"\"{text}\"\"\"
"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)
```

    Completion for prompt 1:
    1 - Jack and Jill go on a quest to fetch water from a well, but encounter misfortune on the way back home.
    
    2 - Jack et Jill partent en quête d'eau d'un puits, mais rencontrent un malheur sur le chemin du retour.
    
    3 - Jack, Jill
    
    4 - {
        "french_summary": "Jack et Jill partent en quête d'eau d'un puits, mais rencontrent un malheur sur le chemin du retour.",
        "num_names": 2
    }


#### Ask for output in a specified format


```python
prompt_2 = f"""
Your task is to perform the following actions:
1 - Summarize the following text delimited by
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in summary>
Output JSON: <json with summary and num_names>

Text: \"\"\"{text}\"\"\"
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)
```

    
    Completion for prompt 2:
    Summary: Jack and Jill, two siblings, go on a quest to fetch water from a hilltop well but encounter misfortune along the way, yet their adventurous spirits remain undimmed.
    Translation: Jack et Jill, deux frères et sœurs, partent en quête d'eau d'un puits au sommet d'une colline mais rencontrent des malheurs en chemin, pourtant leurs esprits aventureux restent intacts.
    Names: Jack, Jill
    Output JSON: {"french_summary": "Jack et Jill, deux frères et sœurs, partent en quête d'eau d'un puits au sommet d'une colline mais rencontrent des malheurs en chemin, pourtant leurs esprits aventureux restent intacts.", "num_names": 2}


#### Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion


```python
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
 help working out the financials.
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)
```

    The student's solution is correct. The total cost for the first year of operations as a function of the number of square feet is indeed 450x + 100,000.


#### Note that the student's solution is actually not correct.
#### We can fix this by instructing the model to work out its own solution first.


```python
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem including the final total.
- Then compare your solution to the student's solution \
and evaluate if the student's solution is correct or not.
Don't decide if the student's solution is correct until
you have done the problem yourself.

Use the following format:
Question:
---
question here
---
Student's solution:
---
student's solution here
---
Actual solution:
---
steps to work out the solution and your solution here
---
Is the student's solution the same as actual solution \
just calculated:
---
yes or no
---
Student grade:
---
correct or incorrect
---

Question:
---
I'm building a solar power installation and I need help \
working out the financials.
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
---
Student's solution:
---
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
---
Actual solution:
"""
response = get_completion(prompt)
print(response)
```

    ---
    Let x be the size of the installation in square feet.
    Costs:
    1. Land cost: $100 * x
    2. Solar panel cost: $250 * x
    3. Maintenance cost: $100,000 + $10 * x
    
    Total cost: $100 * x + $250 * x + $100,000 + $10 * x = $350 * x + $100,000
    ---
    
    Is the student's solution the same as actual solution just calculated:
    ---
    No
    ---
    
    Student grade:
    ---
    Incorrect


## Model Limitations: Hallucinations
- Boie is a real company, the product name is not real.


```python
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)
```

    The AeroGlide UltraSlim Smart Toothbrush by Boie is a high-tech toothbrush designed to provide a superior cleaning experience. It features a slim and sleek design that is easy to hold and maneuver, making it perfect for those who want a comfortable and efficient brushing experience.
    
    The toothbrush is equipped with smart technology that tracks your brushing habits and provides real-time feedback to help you improve your oral hygiene routine. It also has a built-in timer and pressure sensor to ensure that you are brushing for the recommended amount of time and with the right amount of pressure.
    
    The bristles of the AeroGlide UltraSlim Smart Toothbrush are made from a durable and hygienic material that is gentle on your teeth and gums. They are also designed to effectively remove plaque and debris from hard-to-reach areas, leaving your mouth feeling clean and fresh.
    
    Overall, the AeroGlide UltraSlim Smart Toothbrush by Boie is a cutting-edge dental tool that combines style, functionality, and technology to help you achieve a healthier smile.


<a name='2'></a>
# Summarizing
Summarize text with a focus on specific topics.


## Text to summarize


```python
prod_review = """
Got this panda plush toy for my daughter's birthday, \
who loves it and takes it everywhere. It's soft and \
super cute, and its face has a friendly look. It's \
a bit small for what I paid though. I think there \
might be other options that are bigger for the \
same price. It arrived a day earlier than expected, \
so I got to play with it myself before I gave it \
to her.
"""
```

## Summarize with a word/sentence/character limit


```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site.

Summarize the review below, delimited by triple
quotes, in at most 30 words.

Review: \"\"\"{prod_review}\"\"\"
"""

response = get_completion(prompt)
print(response)
```

    ```
    Cute and soft panda plush toy loved by daughter, but smaller than expected for the price. Arrived early, allowing for personal enjoyment before gifting.
    ```


## Summarize with a focus on shipping and delivery


```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
Shipping deparmtment.

Summarize the review below, delimited by triple
quotes, in at most 30 words, and focusing on any aspects \
that mention shipping and delivery of the product.

Review: \"\"\"{prod_review}\"\"\"
"""

response = get_completion(prompt)
print(response)

```

    ```
    Product arrived a day early, allowing for personal inspection before gifting. Customer suggests larger options for the price paid.
    ```


## Summarize with a focus on price and value


```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
pricing deparmtment, responsible for determining the \
price of the product.

Summarize the review below, delimited by triple
quotes, in at most 30 words, and focusing on any aspects \
that are relevant to the price and perceived value.

Review: \"\"\"{prod_review}\"\"\"
"""

response = get_completion(prompt)
print(response)

```

    The panda plush toy is loved for its softness and cuteness, but some customers feel it's a bit small for the price, suggesting larger options at the same price point.


#### Comment
- Summaries include topics that are not related to the topic of focus.

## Try "extract" instead of "summarize"


```python
prompt = f"""
Your task is to extract relevant information from \
a product review from an ecommerce site to give \
feedback to the Shipping department.

From the review below, delimited by triple quotes \
extract the information relevant to shipping and \
delivery. Limit to 30 words.

Review: \"\"\"{prod_review}\"\"\"
"""

response = get_completion(prompt)
print(response)
```

    Shipping feedback: Product arrived a day earlier than expected, allowing customer to play with it before giving it as a gift.


## Summarize multiple product reviews


```python

review_1 = prod_review

# review for a standing lamp
review_2 = """
Needed a nice lamp for my bedroom, and this one \
had additional storage and not too high of a price \
point. Got it fast - arrived in 2 days. The string \
to the lamp broke during the transit and the company \
happily sent over a new one. Came within a few days \
as well. It was easy to put together. Then I had a \
missing part, so I contacted their support and they \
very quickly got me the missing piece! Seems to me \
to be a great company that cares about their customers \
and products.
"""

# review for an electric toothbrush
review_3 = """
My dental hygienist recommended an electric toothbrush, \
which is why I got this. The battery life seems to be \
pretty impressive so far. After initial charging and \
leaving the charger plugged in for the first week to \
condition the battery, I've unplugged the charger and \
been using it for twice daily brushing for the last \
3 weeks all on the same charge. But the toothbrush head \
is too small. I’ve seen baby toothbrushes bigger than \
this one. I wish the head was bigger with different \
length bristles to get between teeth better because \
this one doesn’t.  Overall if you can get this one \
around the $50 mark, it's a good deal. The manufactuer's \
replacements heads are pretty expensive, but you can \
get generic ones that're more reasonably priced. This \
toothbrush makes me feel like I've been to the dentist \
every day. My teeth feel sparkly clean!
"""

# review for a blender
review_4 = """
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

reviews = [review_1, review_2, review_3, review_4]

```


```python
for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary of a product \
    review from an ecommerce site.

    Summarize the review below, delimited by triple \
    quotes in at most 20 words.

    Review: '''{reviews[i]}'''
    """

    response = get_completion(prompt)
    print(i, response, "\n")

```

    0 Summary: 
    Cute panda plush toy loved by daughter, soft and friendly, but smaller than expected for the price. Arrived early. 
    
    1 Summary: 
    Lamp with storage, affordable, fast delivery. Excellent customer service - replaced broken parts and missing piece promptly. 
    
    2 Impressive battery life, small head, good deal for $50, generic replacement heads available, leaves teeth feeling clean. 
    
    3 Summary: Price increased post-sale, quality decline noted, motor issue after a year, but efficient for various food prep tasks. 
    




<a name='3'></a>
#Inferring
Infer sentiment and topics from product reviews and news articles

## Product review text


```python
lamp_review = """
Needed a nice lamp for my bedroom, and this one had \
additional storage and not too high of a price point. \
Got it fast.  The string to our lamp broke during the \
transit and the company happily sent over a new one. \
Came within a few days as well. It was easy to put \
together.  I had a missing part, so I contacted their \
support and they very quickly got me the missing piece! \
Lumina seems to me to be a great company that cares \
about their customers and products!!
"""
```

## Sentiment (positive/negative)


```python
prompt = f"""
What is the sentiment of the following product review,
which is delimited with triple quotes?

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

    The sentiment of the review is positive. The reviewer is satisfied with the lamp they purchased, appreciates the additional storage and reasonable price point, and is happy with the customer service provided by the company. They describe the company as caring about their customers and products.



```python
prompt = f"""
What is the sentiment of the following product review,
which is delimited with triple quotes?

Give your answer as a single word, either "positive" \
or "negative".

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

    Positive


## Identify types of emotions


```python
prompt = f"""
Identify a list of emotions that the writer of the \
following review is expressing. Include no more than \
five items in the list. Format your answer as a list of \
lower-case words separated by commas.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

    happy, satisfied, grateful, impressed, content


## Identify anger


```python
prompt = f"""
Is the writer of the following review expressing anger?\
The review is delimited with triple quotes. \
Give your answer as either yes or no.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

    No


## Extract product and company name from customer reviews


```python
prompt = f"""
Identify the following items from the review text:
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple quotes. \
Format your response as a JSON object with \
"Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

    {
        "Item": "lamp",
        "Brand": "Lumina"
    }


## Doing multiple tasks at once


```python
prompt = f"""
Identify the following items from the review text:
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple quotes. \
Format your response as a JSON object with \
"Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

    {
        "Sentiment": "positive",
        "Anger": false,
        "Item": "lamp",
        "Brand": "Lumina"
    }


## Inferring topics


```python
story = """
In a recent survey conducted by the government,
public sector employees were asked to rate their level
of satisfaction with the department they work at.
The results revealed that NASA was the most popular
department with a satisfaction rating of 95%.

One NASA employee, John Smith, commented on the findings,
stating, "I'm not surprised that NASA came out on top.
It's a great place to work with amazing people and
incredible opportunities. I'm proud to be a part of
such an innovative organization."

The results were also welcomed by NASA's management team,
with Director Tom Johnson stating, "We are thrilled to
hear that our employees are satisfied with their work at NASA.
We have a talented and dedicated team who work tirelessly
to achieve our goals, and it's fantastic to see that their
hard work is paying off."

The survey also revealed that the
Social Security Administration had the lowest satisfaction
rating, with only 45% of employees indicating they were
satisfied with their job. The government has pledged to
address the concerns raised by employees in the survey and
work towards improving job satisfaction across all departments."""
```

## Infer 5 topics


```python
prompt = f"""
Determine five topics that are being discussed in the \
following text, which is delimited by triple quotes.

Make each item one or two words long.

Format your response as a list of items separated by commas such as topic1, topic2, ...

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)
```

    government, survey, NASA, job satisfaction, Social Security Administration



```python
topic_list = [
    "nasa", "local government", "engineering",
    "employee satisfaction", "federal government"
]
```

## Make a news alert for certain topics


```python
prompt = f"""
Determine whether each item in the following list of \
topics is a topic in the text below, which
is delimited with triple quotes.

Give your answer as list with 0 or 1 for each topic in the following format 1,0,1...

List of topics: {", ".join(topic_list)}

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)
```

    1,0,0,1,1



```python
topic_dict = {key:int(i) for key, i in zip(topic_list, response.split(sep=','))}
if topic_dict['nasa'] == 1:
    print("ALERT: New NASA story!")
```

    ALERT: New NASA story!


<a name='4'></a>
#Transforming
Transformation tasks such as language translation, spelling and grammar checking, tone adjustment, and format conversion.

## Translation

ChatGPT is trained with sources in many languages. This gives the model the ability to do translation. Here are some examples of how to use this capability.


```python
prompt = f"""
Translate the following English text to Spanish: \
'''Hi, I would like to order a blender'''
"""
response = get_completion(prompt)
print(response)
```

    Hola, me gustaría ordenar una licuadora.



```python
prompt = f"""
Tell me which language this is:
'''Combien coûte le lampadaire?'''
"""
response = get_completion(prompt)
print(response)
```

    This is French.



```python
prompt = f"""
Translate the following  text to French and Spanish
and English pirate: \
'''I want to order a basketball'''
"""
response = get_completion(prompt)
print(response)
```

    French: "Je veux commander un ballon de basket"
    
    Spanish: "Quiero ordenar un balón de baloncesto"
    
    English: "I want to order a basketball"



```python
prompt = f"""
Translate the following text to Spanish in both the \
formal and informal forms:
'Would you like to order a pillow?'
"""
response = get_completion(prompt)
print(response)
```

    Formal: ¿Le gustaría ordenar una almohada?
    Informal: ¿Te gustaría ordenar una almohada?


### Universal Translator
Imagine you are in charge of IT at a large multinational e-commerce company. Users are messaging you with IT issues in all their native languages. Your staff is from all over the world and speaks only their native languages. You need a universal translator!


```python
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                               # My screen is flashing
]
```


```python
for issue in user_messages:
    prompt = f"Tell me what language this is: '''{issue}'''"
    lang = get_completion(prompt)
    print(f"Original message ({lang}): {issue}")

    prompt = f"""
    Translate the following  text to English \
    and Korean: '''{issue}'''
    """
    response = get_completion(prompt)
    print(response, "\n")
```

    Original message (French): La performance du système est plus lente que d'habitude.
    English: "The system performance is slower than usual."
    Korean: "시스템 성능이 평소보다 느립니다." 
    
    Original message (Spanish): Mi monitor tiene píxeles que no se iluminan.
    English: "My monitor has pixels that do not light up."
    Korean: "내 모니터에는 빛나지 않는 픽셀이 있습니다." 
    
    Original message (Italian): Il mio mouse non funziona
    English: My mouse is not working
    Korean: 내 마우스가 작동하지 않습니다 
    
    Original message (This is Polish.): Mój klawisz Ctrl jest zepsuty
    English: My Ctrl key is broken
    Korean: 나의 Ctrl 키가 고장 났어요 
    
    Original message (This is Chinese.): 我的屏幕在闪烁
    English: "My screen is flickering."
    Korean: "내 화면이 깜박거립니다." 
    


## Format Conversion
ChatGPT can translate between formats. The prompt should describe the input and output formats.


```python
data_json = { "resturant employees" :[
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""
response = get_completion(prompt)
print(response)
```

    <html>
    <head>
      <title>Restaurant Employees</title>
    </head>
    <body>
      <table>
        <tr>
          <th>Name</th>
          <th>Email</th>
        </tr>
        <tr>
          <td>Shyam</td>
          <td>shyamjaiswal@gmail.com</td>
        </tr>
        <tr>
          <td>Bob</td>
          <td>bob32@gmail.com</td>
        </tr>
        <tr>
          <td>Jai</td>
          <td>jai87@gmail.com</td>
        </tr>
      </table>
    </body>
    </html>



```python
from IPython.display import display, Markdown, Latex, HTML, JSON
display(HTML(response))
```


<html>
<head>
  <title>Restaurant Employees</title>
</head>
<body>
  <table>
    <tr>
      <th>Name</th>
      <th>Email</th>
    </tr>
    <tr>
      <td>Shyam</td>
      <td>shyamjaiswal@gmail.com</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>bob32@gmail.com</td>
    </tr>
    <tr>
      <td>Jai</td>
      <td>jai87@gmail.com</td>
    </tr>
  </table>
</body>
</html>


## Spellcheck/Grammar check.

Here are some examples of common grammar and spelling problems and the LLM's response.

To signal to the LLM that you want it to proofread your text, you instruct the model to 'proofread' or 'proofread and correct'.


```python
text = [
  "The girl with the black and white puppies have a ball.",  # The girl has a ball.
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
  "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
  "Your going to need you’re notebook.",  # Homonyms
  "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
  "This phrase is to cherck chatGPT for speling abilitty"  # spelling
]
for t in text:
    prompt = f"""Proofread and correct the following text
    and rewrite the corrected version. If you don't find
    and errors, just say "No errors found". Don't use
    any punctuation around the text:
    '''{t}'''"""
    response = get_completion(prompt)
    print(response)
```

    The girl with the black and white puppies has a ball.
    No errors found
    No errors found.
    No errors found.
    You're going to need your notebook.
    No errors found.
    No errors found



```python
text = f"""
Got this for my daughter for her birthday cuz she keeps taking \
mine from my room.  Yes, adults also like pandas too.  She takes \
it everywhere with her, and it's super soft and cute.  One of the \
ears is a bit lower than the other, and I don't think that was \
designed to be asymmetrical. It's a bit small for what I paid for it \
though. I think there might be other options that are bigger for \
the same price.  It arrived a day earlier than expected, so I got \
to play with it myself before I gave it to my daughter.
"""
prompt = f"proofread and correct this review: '''{text}'''"
response = get_completion(prompt)
print(response)
```

    Got this for my daughter for her birthday because she keeps taking mine from my room. Yes, adults also like pandas too. She takes it everywhere with her, and it's super soft and cute. One of the ears is a bit lower than the other, and I don't think that was designed to be asymmetrical. It's a bit small for what I paid for it though. I think there might be other options that are bigger for the same price. It arrived a day earlier than expected, so I got to play with it myself before I gave it to my daughter.



```python
from redlines import Redlines

diff = Redlines(text,response)
display(Markdown(diff.output_markdown))
```


```python
prompt = f"""
proofread and correct this review. Make it more compelling.
Ensure it follows APA style guide and targets an advanced reader.
Output in markdown format.
Text: '''{text}'''
"""
response = get_completion(prompt)
display(Markdown(response))
```


# Review of Panda Plush Toy

I purchased this adorable panda plush toy for my daughter's birthday, as she kept taking mine from my room. Despite being marketed towards children, adults can also appreciate the charm of this cuddly companion. The plush is incredibly soft and undeniably cute, making it a hit with my daughter who now takes it everywhere with her.

However, I did notice a slight flaw in the design - one of the ears is slightly lower than the other, giving it an unintentional asymmetrical look. Additionally, I found the size of the plush to be a bit smaller than expected given the price point. I believe there are larger options available for the same cost.

On a positive note, the plush arrived a day earlier than anticipated, allowing me to enjoy its company before gifting it to my daughter. Overall, while there are some minor drawbacks, the quality and charm of this panda plush make it a worthwhile purchase for any panda enthusiast.

---
APA Style:
Author. (Year). Review of Panda Plush Toy. Markdown Review Journal, Volume(Issue), Page range.



