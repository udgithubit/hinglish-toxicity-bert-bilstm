import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

# Configuration
DATASET_PATH = Path(__file__).parent / "data" / "hinglish_toxicity_sample_500.csv"
EPOCHS = 4
BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data Preparation
def load_data():
    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    df = df.rename(columns={"Text": "text", "Label": "label"})
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\u0900-\u097F\s।?!,]', '', str(x)))
    df = df[df['label'].isin([0, 1])]
    return df


class HindiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


# Model Definition: plain BERT, no BiLSTM
class BertClassifier(nn.Module):
    """BERT alone, using the pooled output straight into a linear classifier"""

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return torch.sigmoid(self.classifier(pooled)).squeeze()


# Training and Evaluation
def train_model(model, train_loader, epochs):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())


def evaluate_model(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label']
            outputs = model(**inputs)
            preds.extend(torch.round(outputs).cpu().numpy())
            truths.extend(labels.numpy())
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)
    prec = precision_score(truths, preds, zero_division=0)
    recall = recall_score(truths, preds)
    return acc, prec, recall, f1, truths, preds


def main():
    df = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_loader = DataLoader(
        HindiDataset(X_train.tolist(), y_train.tolist(), tokenizer),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        HindiDataset(X_val.tolist(), y_val.tolist(), tokenizer),
        batch_size=BATCH_SIZE
    )

    print(f"\n{'=' * 50}\nTraining Baseline: BERT only (no BiLSTM)\n{'=' * 50}")
    model = BertClassifier()
    train_model(model, train_loader, EPOCHS)
    acc, prec, recall, f1, truths, preds = evaluate_model(model, val_loader)

    print("\n" + "=" * 80)
    print(f"BASELINE RESULTS ON {DATASET_PATH} (Epochs={EPOCHS})")
    print("=" * 80)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(truths, preds, zero_division=0))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()