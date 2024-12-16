import json
import pandas as pd
import random
from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizerFast

def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

filename = 'output2 (1).json'

intents = load_json_file(filename)

def create_df():
    df = pd.DataFrame({
        'Pattern' : [],
        'Tag' : []
    })

    return df
df = create_df()

def extract_json_info(json_file, df):

    for intent in json_file:

        for pattern in intent['patterns']:

            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag

    return df

df = extract_json_info(intents, df)

labels = df['Tag'].unique().tolist()
labels = [s.strip() for s in labels]

num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}



model_path = "F:\AI LEVEL 4\AI SEMESTER (1)\Graduation Project\Final ChatBot version (2)\chatbot"


model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def chat(chatbot):

    print("Chatbot: Hi! I am your virtual assistance,Feel free to ask, and I'll do my best to provide you with answers and assistance..")
    print("Type 'quit' to exit the chat\n\n")

    text = input("User: ").strip().lower()

    while(text != 'quit'):

        label = label2id[chatbot(text)[0]['label']]

        score = chatbot(text)[0]['score']
        if label >=0 and label <= 23:
          response = random.choice(intents[label]['responses'])
          print(f"Chatbot: {response}\n")
          print(f"The probability of accuracy of this diagnosis is {score}\n\n")
          text = input("User: ").strip().lower()
          continue

        if score < 0.5:
            print("Chatbot: Sorry I can't answer that\n\n")
            text = input("User: ").strip().lower()
            continue

        # label = label2id[chatbot(text)[0]['label']]
        response = random.choice(intents[label]['responses'])

        print(f"Chatbot: {response}\n\n")

        text = input("User: ").strip().lower()
        
        
chat(chatbot)