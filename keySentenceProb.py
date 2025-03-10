import nltk
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
import pdb
import random
from tqdm import tqdm
import torch.nn.functional as F
import re

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_key_sentence_start(input_text, key_sentence, tokenizer):
    """
    Find the start index of the key sentence in the tokenized input.
    This approach uses character matching to avoid tokenization mismatches.
    """

    # Convert the input text and key sentence back to strings for character position matching

    # Find the position
    char_start_pos = input_text.find(key_sentence)

    if char_start_pos == -1:
        pdb.set_trace()
        raise ValueError("Key sentence not found in the input text.")

    try:
        # Map character position to token position
        token_start_pos = tokenizer(input_text[:char_start_pos], return_tensors="pt")["input_ids"].squeeze().shape[0]
    except Exception as e:
        # Print detailed error message and raise error
        pdb.set_trace()
        raise ValueError(
            f"Error occurred while mapping character position to token position.\n"
            f"Details: {str(e)}\n"
            f"Input text: {input_text}\n"
            f"Character start position: {char_start_pos}"
        )
    return token_start_pos

def compute_key_sentence_losses(model, tokenizer, text, key_sentence, max_tokens=20, device='cuda'):
    """
    Compute the loss for each word in the key sentence with a maximum length of `max_tokens`.
    """
    # Tokenize the entire text (context + key sentence) with truncation and padding
    inputs = tokenizer(text, return_tensors="pt",max_length=4096, truncation=True).to(device)
    input_ids = inputs["input_ids"]

    # Tokenize the key sentence separately to match against the text
    key_sentence_encoding = tokenizer(key_sentence, return_tensors="pt").to(device)
    key_sentence_ids = key_sentence_encoding["input_ids"]  # Access the input_ids

    # Find the position of the key sentence in the input_ids
    key_sentence_start = find_key_sentence_start(text, key_sentence, tokenizer) - 1
    if key_sentence_start == -1:
        return None

    # Perform the forward pass to get the logits (prediction scores for each token)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Convert logits to probabilities (this step is memory intensive)
    probs = F.softmax(logits, dim=-1)

    # Calculate the loss for each token using CrossEntropyLoss
    losses = []
    for i in range(min(len(key_sentence_ids[0])-1, max_tokens)):  # Ensure we don't go out of bounds
        # Get the predicted token probabilities for the i-th token
        try:
            predicted_token_probs = probs[0, key_sentence_start + i]
        except Exception as e:
            pdb.set_trace()
            raise ValueError("index is out of bounds.")
        # Get the true token (target) for the i-th token
        true_token_id = input_ids[0, key_sentence_start + i].item()

        # Compute the cross-entropy loss for the true token
        loss = -torch.log(predicted_token_probs[true_token_id])
        losses.append(loss.item())

    if len(losses) < max_tokens:
        losses.extend([0] * (max_tokens - len(losses)))  # Padding with 0s

    # Free up memory for tensors not needed after this point
    del inputs, key_sentence_encoding, outputs, logits, probs
    torch.cuda.empty_cache()

    return losses

def regenerate_important_sentence(news_id, important_sentences_list):
    # Find the corresponding key sentence
    key_sentence = None
    for key, value in important_sentences_list.items():
        if value['id'] == news_id:
            key_sentence = value['sentence']
            break

    return key_sentence


if __name__ == '__main__':
    # Read key sentence numbering document
    with open('key_sentence_extraction//Gossipcop//train_gossip_important_sentences.json') as f:
        train_important_sentences_list = json.load(f)

    with open('key_sentence_extraction//Gossipcop//test_gossip_important_sentences.json') as f:
        test_important_sentences_list = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(2024)

    model_path = "/home/wangchi/LLaMA"
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Read the original data and retrieve the key sentence JSON file corresponding to the ID
    with open('train_gossip.json') as f:
        train_data = json.load(f)

    with open('test_gossip.json') as f:
        test_data = json.load(f)

    max_words = 20
    train_all_word_probs = []
    train_all_labels = []
    train_all_news_id = []
    # Processing the train_data dataset
    for data in tqdm(train_data.values(), desc="Processing train_data"):
        news_id = data['id']
        news_article = data['article']
        news_title = data['title']
        news_label = data['label']
        text = f"Please pretend to be a fake news author. Based on the following Title and Article, continue writing " \
               f"the content to make the article align with the characteristics of fake news.\n\n" \
               f"Title:{news_title}. Article:{news_article}"

        key_sentence = regenerate_important_sentence(news_id, important_sentences_list=train_important_sentences_list)
        # pdb.set_trace()

        word_probs = compute_key_sentence_losses(model,tokenizer, text, key_sentence, max_words, device)
        if word_probs is not None:
            train_all_news_id.append(news_id)
            train_all_word_probs.append(word_probs)
            train_all_labels.append(news_label)

    all_news_id = torch.tensor(train_all_news_id)
    all_word_probs_tensor = torch.tensor(train_all_word_probs)
    all_labels_tensor = torch.tensor(train_all_labels)
    torch.save({'id': all_news_id, 'probs': all_word_probs_tensor, 'labels': all_labels_tensor},
               "Gossipcop//train_gossip_data_probs_llama.pt")
    print("Gossipcop//train_gossip_data_probs_llama.pt")


    test_all_word_probs = []
    test_all_labels = []
    test_all_news_id = []
    # Processing the test_data dataset
    for data in tqdm(test_data.values(), desc="Processing test_data"):
        news_id = data['id']
        news_article = data['article']
        news_title = data['title']
        news_label = data['label']
        text = f"Please pretend to be a fake news author. Based on the following Title and Article, continue writing " \
               f"the content to make the article align with the characteristics of fake news.\n\n" \
               f"Title:{news_title}. Article:{news_article}"
        key_sentence = regenerate_important_sentence(news_id, important_sentences_list=test_important_sentences_list)
        word_probs = compute_key_sentence_losses(model,tokenizer, text, key_sentence,max_words, device)
        if word_probs is not None:
            test_all_news_id.append(news_id)
            test_all_word_probs.append(word_probs)
            test_all_labels.append(news_label)

    all_news_id = torch.tensor(test_all_news_id)
    all_word_probs_tensor = torch.tensor(test_all_word_probs)
    all_labels_tensor = torch.tensor(test_all_labels)
    # Save probability distribution and labels
    torch.save({'id': all_news_id, 'probs': all_word_probs_tensor, 'labels': all_labels_tensor},
               "Gossipcop//test_gossip_data_probs_llama.pt")
    print("Gossipcop//test_gossip_data_probs_llama.pt")