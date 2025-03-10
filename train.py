import pdb
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np
import random
import os
from sklearn import metrics
from typing import List, Tuple


# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu = nn.LeakyReLU()

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x):

        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)

        return x

# training
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()

    loss_avg = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_avg.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
    return np.mean(loss_avg)

# testing
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

# evaluate the distribution of vector
def evaluate_model1(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            all_preds.append(outputs)
            all_labels.append(labels)

    return torch.cat(all_preds).cpu(), torch.cat(all_labels).cpu()

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


if __name__ =='__main__':
    # Load datas
    seed_torch(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    train_data = torch.load("Gossipcop//train_gossip_data_probs_llama.pt")
    test_data = torch.load("Gossipcop//test_gossip_data_probs_llama.pt")

    train_word_probs = train_data['probs'].float()
    train_labels = train_data['labels'].long()

    test_word_probs = test_data['probs'].float()
    test_labels = test_data['labels'].long()

    X_train, y_train = (train_word_probs, train_labels)
    X_test, y_test = (test_word_probs, test_labels)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = 20
    hidden_dim = 128
    output_dim = 2

    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    print("*"*80)
    print("Training...")
    # training model
    for epoch in range(100):
        avg_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{100} avg loss:", avg_loss)

    # testing model
    print("*"*80)
    print("Testing...")
    preds, test_labels = evaluate_model(model, test_loader, device)

    result_pre = precision_score(test_labels, preds,zero_division=0,average="weighted")
    result_f1 = f1_score(test_labels, preds,average="weighted")
    result_recall = recall_score(test_labels, preds,average="weighted")
    result_acc = accuracy_score(test_labels, preds)
    print(metrics.classification_report(test_labels, preds, digits=4))
    print(
        f"result：f1分数：{result_f1}，recall分数：{result_recall}，Accuracy分数：{result_acc}，precision分数：{result_pre}")

