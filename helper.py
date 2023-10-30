import openai
from tqdm import tqdm
import os
import pandas as pd
from itertools import chain
from time import sleep, time

def extract_keywords(dataset, prompt, context, resume=0):

    input_path = os.path.join('Data/Processed', dataset, 'TTM', dataset + '.txt')
    output_path = os.path.join('Data/Processed', dataset, 'LLM', dataset + '.txt')

    openai.api_key = ''
    sleep(2) # Rate Limit Buffer
    
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = f.read().split('\n')
        
    with open(output_path, 'a', encoding='utf-8') as f:
        for text in tqdm(documents[resume:], desc="Processing files"):
            modified_prompt = prompt.format(context=context, document=text)
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": modified_prompt}
                ]
            )

            try:
                f.write(completion.choices[0].message['content'] + '\n')  # Append the message with a newline
            except IndexError:
                print(f'API Error')
                break

def sample_csv(dataset, num_samples, columns_to_retain):
    
    input_path = os.path.join('Data/Raw', dataset, dataset + '.csv')
    output_path = os.path.join('Data/Processed', dataset, 'TTM', dataset + '.txt')
    
    df = pd.read_csv(input_path)
    df = df.sample(n=num_samples)
    abstracts = df[columns_to_retain].values.tolist()
    abstracts = list(chain.from_iterable(abstracts))
    abstracts = [abstract.replace('\n', ' ') for abstract in abstracts]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(abstracts))

def dataset_stats(string_list):
    total_words = 0
    string_count = 0

    for string in string_list:
        words = string.split()
        total_words += len(words)
        string_count += 1

    average_word_count = total_words / string_count
    return int(average_word_count), string_count, total_words


def postprocess_topics(input_folder, output_folder, prompt):
    
    openai.api_key = ''
    
    for file in os.listdir(input_folder):
    
        print('Processing ' + file)
    
        file_path = os.path.join(input_folder, file)
        with open(file_path, 'r') as f:
            topics = f.readlines()
    
        topics = [topic.split(':')[-1].strip() for topic in topics]
        context = file.split('-')[0].replace('_', ' ')
    
        save_path = os.path.join(output_folder, file.replace('.txt', '.csv'))
        with open(save_path, 'w', encoding='utf-8') as f:
            for topic in tqdm(topics, desc="Processing topics"):
                modified_prompt = prompt.format(context=context, words=topic)
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert annotator for topic modeling results."},
                        {"role": "user", "content": modified_prompt}
                    ]
                )
    
                try:
                    f.write(completion.choices[0].message['content'] + '\n')
                    sleep(2) # Rate Limit Buffer
                except IndexError:
                    print(f'API Error')
                    break 