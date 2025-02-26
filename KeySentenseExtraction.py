import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import nltk
from sklearn.model_selection import train_test_split
import os
import pdb
import torch.nn as nn
# nltk.download('punkt')
# 加载新闻数据


def load_news_data(mf_file, mr_file):
    with open(mf_file, 'r', encoding='utf-8') as mf:
        mf_data = json.load(mf)

    with open(mr_file, 'r', encoding='utf-8') as mr:
        mr_data = json.load(mr)

    return mf_data, mr_data


def load_Gptnews_data(file):
    texts = []
    labels = []
    with open(file,'r',encoding='utf-8') as f:
        datas = json.load(f)
    for data in datas:
        texts.append(datas[data]['article'])
        labels.append(int(datas[data]['label']))
    return texts, labels, datas

# 将新闻数据转换为文本和标签
def prepare_data(mf_data, mr_data):
    texts = []
    labels = []

    for key, value in mf_data.items():
        texts.append(value['text'])
        labels.append(1)  # 虚假新闻标记为1

    for key, value in mr_data.items():
        texts.append(value['text'])
        labels.append(0)  # 真实新闻标记为0

    return texts, labels


# 预处理数据
def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    return dataset


# 训练模型
def train_model(train_dataset, val_dataset):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',  # 输出路径
        evaluation_strategy='epoch',  # 每个epoch进行评估
        per_device_train_batch_size=8,  # 训练时的batch size
        per_device_eval_batch_size=8,  # 评估时的batch size
        num_train_epochs=10,  # 训练轮数
        weight_decay=0.01,  # 权重衰减
        logging_dir='./logs',  # 日志路径
        report_to="none"  # 禁用wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    return model


# 对新闻文本进行分类
def classify_news(news_text, model, tokenizer, device, news_label):
    model.eval()
    inputs = tokenizer(news_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    news_label = torch.tensor([news_label], dtype=torch.long).to(device)  # 将标签转换为 Tensor 类型，并移到设备
    loss_fn = nn.CrossEntropyLoss()  # Using CrossEntropyLoss to compute the loss
    loss = loss_fn(logits, news_label)  # logits should be raw outputs, not probabilities

    return loss.item()  # Return the loss value and the probabilities

# 将新闻分割为句子
def split_into_sentences(news_text):
    return nltk.tokenize.sent_tokenize(news_text)

# 逐个删除句子并检测分类变化，获取关键句子
def get_sentence_importance(news_text, model, tokenizer, device, news_label):
    sentences = split_into_sentences(news_text)
    original_loss = classify_news(news_text, model, tokenizer, device, news_label)

    max_change = 0
    key_sentence = None
    key_index = -1

    for i in range(len(sentences)):
        # 删除句子 i
        modified_text = ' '.join([sent for j, sent in enumerate(sentences) if j != i])

        # 对删除后的新闻进行分类
        new_loss = classify_news(modified_text, model, tokenizer, device,news_label)

        # 计算删除前后的分类结果变化
        change = abs(original_loss - new_loss)

        # 如果当前句子导致的分类概率上升最大，记录它
        if change >= max_change:
            max_change = change
            key_sentence = sentences[i]
            key_index = i

    return key_sentence, key_index


# 主函数：加载数据、训练模型、提取关键句子并保存为JSON文件
def main():
    # 文件路径
    # mf_file = '/home/wangchi/MIN-FNS/Experiment/Dataset_unpack/PolitiFact++/MF_politifact.json'
    # mr_file = '/home/wangchi/MIN-FNS/Experiment/Dataset_unpack/PolitiFact++/MR_politifact.json'
    # mf_data, mr_data = load_news_data(mf_file, mr_file)
    # texts, labels = prepare_data(mf_data, mr_data)
    gpt_file = '/home/wangchi/MIN-FNS/Experiment/Dataset_FakeLLM/ChatGPT3.5-Politifact/merged_data.json'
    output_file = 'GPT_important_sentences.json'
    texts, labels, gpt_datas = load_Gptnews_data(gpt_file)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.9, random_state=42)

    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 创建训练和验证数据集
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_data(val_texts, val_labels, tokenizer)

    # 训练模型
    if os.path.exists("GPTmodel.pt"):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.load_state_dict(torch.load("GPTmodel.pt"))
        model.eval()  # 切换到评估模式
    else:
        model = train_model(train_dataset, val_dataset)
        torch.save(model.state_dict(), "GPTmodel.pt")

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 处理所有新闻，提取最重要的句子
    important_sentences = {}
    idx = 0  # 全局索引计数器
    # pdb.set_trace()
    # 处理 GPT 文件中的新闻
    for id, (key, value) in enumerate(gpt_datas.items()):
        # pdb.set_trace()
        news_text = value['article']  # 直接从字典中获取
        news_id = value['id']  # 获取 id
        news_label = int(value['label'])
        key_sentence, key_index = get_sentence_importance(news_text, model, tokenizer, device, news_label)

        important_sentences[idx] = {
            'id': news_id,
            'sentence': key_sentence,
            'index': key_index
        }
        idx += 1

    # # 处理 MR 文件中的新闻
    # for id, (key, value) in enumerate(mr_data.items()):
    #     # pdb.set_trace()
    #     news_text = value['article']  # 直接从字典中获取
    #     news_id = value['id']  # 获取 id
    #     news_label = 0
    #     key_sentence, key_index = get_sentence_importance(news_text, model, tokenizer, device, news_label)
    #
    #     important_sentences[idx] = {
    #         'id': news_id,
    #         'sentence': key_sentence,
    #         'index': key_index
    #     }
    #     idx += 1

    # 保存结果到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(important_sentences, f, ensure_ascii=False, indent=4)

    print(f"关键句子提取完成，结果已保存至 {output_file}")


if __name__ == "__main__":
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
