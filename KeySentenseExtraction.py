import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import nltk
from sklearn.model_selection import train_test_split
import os
import pdb
import torch.nn as nn

# Load News and Get the articles and labels
def load_news_data(train_file,test_file):
    train_texts = []
    train_labels = []
    test_texts = []
    with open(train_file,'r',encoding='utf-8') as f:
        train_datas = json.load(f)
    with open(test_file,'r',encoding='utf-8') as s:
        test_datas = json.load(s)
    for data in train_datas:
        train_texts.append(train_datas[data]['article'])
        train_labels.append(int(train_datas[data]['label']))

    for data in test_datas:
        test_texts.append(test_datas[data]['article'])

    return train_texts, train_labels, test_texts, train_datas, test_datas

# tokenize the data
def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    return dataset

# Finetune the Pre-trained model
def train_model(train_dataset):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None
    )
    trainer.train()
    return model

# Get the classification probability of the original news and the masked news.
def classify_news(news_text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(news_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # calculate the probability
    probability = torch.softmax(logits, dim=-1)  # logits -> probability
    # pdb.set_trace()
    return probability[0][1].item()

# split the news into sentences
def split_into_sentences(news_text):
    return nltk.tokenize.sent_tokenize(news_text)

# Mask the news one sentence by sentence
def get_sentence_importance(news_text, model, tokenizer, device):
    sentences = split_into_sentences(news_text)
    anchor_classification = classify_news(news_text, model, tokenizer, device)

    max_change = 0
    key_sentence = None
    key_index = -1

    for i in range(len(sentences)):
        # mask sentence i in the news
        masked_news = ' '.join([sent for j, sent in enumerate(sentences) if j != i])
        # get the classification of masked news
        new_classification = classify_news(masked_news, model, tokenizer, device)
        # calculate the change of masked news and original news
        change = abs(anchor_classification - new_classification)
        # record the biggest change
        if change >= max_change:
            max_change = change
            key_sentence = sentences[i]
            key_index = i

    return key_sentence, key_index

def main():
    # load the train datas and test datas
    train_file = 'train_gossip.json'
    test_file = 'test_gossip.json'

    train_output_file = 'Gossipcop//train_gossip_important_sentences.json'
    test_output_file = 'Gossipcop//test_gossip_important_sentences.json'

    train_texts, train_labels, test_texts, train_datas, test_datas = load_news_data(train_file,test_file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = tokenize_data(train_texts, train_labels, tokenizer)


    if os.path.exists("Gossip_model.pt"):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.load_state_dict(torch.load("Gossip_model.pt"))
        model.eval()
    else:
        model = train_model(train_dataset)
        torch.save(model.state_dict(), "Gossip_model.pt")


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    train_important_sentences = {}
    train_idx = 0
    test_important_sentences = {}
    test_idx = 0

    for id, (key, value) in enumerate(train_datas.items()):

        news_text = value['article']
        news_id = value['id']
        key_sentence, key_index = get_sentence_importance(news_text, model, tokenizer, device)

        train_important_sentences[train_idx] = {
            'id': news_id,
            'sentence': key_sentence,
            'index': key_index
        }
        train_idx += 1


    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_important_sentences, f, ensure_ascii=False, indent=4)

    print(f"Train_datasets Completion! Result saved in  {train_output_file}")


    for id, (key, value) in enumerate(test_datas.items()):
        news_text = value['article']
        news_id = value['id']
        key_sentence, key_index = get_sentence_importance(news_text, model, tokenizer, device)

        test_important_sentences[test_idx] = {
            'id': news_id,
            'sentence': key_sentence,
            'index': key_index
        }
        test_idx += 1

    with open(test_output_file, 'w', encoding='utf-8') as f:
        json.dump(test_important_sentences, f, ensure_ascii=False, indent=4)

    print(f"Test_datasets Completion! Result saved in {test_output_file}")


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
