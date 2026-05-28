# =========================================
# 1. INSTALL
# =========================================
!pip install -q transformers sentence-transformers torch pandas scikit-learn requests

# =========================================
# 2. IMPORTS
# =========================================
import pandas as pd
import numpy as np
import re
import torch
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# =========================================
# 3. LOAD DATA
# =========================================

from google.colab import drive
drive.mount('/content/drive',force_remount=True)

FAKE_PATH = '/content/drive/MyDrive/Capstone/ISOT_Fake_News_detection_dataset/Fake.csv'
TRUE_PATH = '/content/drive/MyDrive/Capstone/ISOT_Fake_News_detection_dataset/True.csv'

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df]).reset_index(drop=True)

# =========================================
# 4. PREPROCESS
# =========================================
df['content'] = df['title']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text

df['content'] = df['content'].apply(clean_text)

# balance
df_real = df[df.label == 1]
df_fake = df[df.label == 0]

min_size = min(len(df_real), len(df_fake))
df = pd.concat([
    df_real.sample(min_size, random_state=42),
    df_fake.sample(min_size, random_state=42)
]).sample(frac=1).reset_index(drop=True)

# reduce size
df = df.sample(20000, random_state=42)

# =========================================
# 5. SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'], test_size=0.2, stratify=df['label']
)

# =========================================
# 6. TOKENIZER
# =========================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =========================================
# 7. DATASET
# =========================================
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_loader = torch.utils.data.DataLoader(
    NewsDataset(X_train, y_train), batch_size=16, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    NewsDataset(X_test, y_test), batch_size=16
)

# =========================================
# 8. MODEL
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# =========================================
# 9. TRAIN
# =========================================
for epoch in range(5):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

# =========================================
# 10. EVALUATION (FIXED)
# =========================================
model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in test_loader:
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        ).logits

        probs = torch.sigmoid(logits).cpu().numpy()
        predictions = (probs > 0.5).astype(int).flatten()

        preds.extend(predictions)
        true.extend(batch['labels'].numpy())

print(classification_report(true, preds))

#New code ..
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(true, preds)

# Display labels
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "REAL"]
)

# Plot
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(true, preds)

print("="*30)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("="*30)

# =========================================
# 11. SAVE
# =========================================
model.save_pretrained("/content/bert_model")
tokenizer.save_pretrained("/content/bert_model")

# =========================================
# 12. VERIFICATION SYSTEM
# =========================================
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_query(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return " ".join(text.split()[:5])

def is_suspicious(text):
    keywords = [
        "hidden", "secret", "unknown", "shocking",
        "they don't want you", "underground civilization",
        "ancient aliens", "conspiracy"
    ]
    return any(k in text.lower() for k in keywords)

def fetch_news(query):
    try:
        url = f"https://news.google.com/rss/search?q={query}"
        res = requests.get(url)
        titles = re.findall(r"<title>(.*?)</title>", res.text)

        good_sources = [
            "reuters", "bbc", "cnn", "forbes",
            "ap news", "bloomberg", "cnbc",
            "the guardian", "nytimes", "washington post"
        ]

        clean_titles = []
        for t in titles[1:10]:
            t_lower = t.lower()

            if "google news" in t_lower:
                continue

            if len(t.split()) < 5:
                continue

            if not any(src in t_lower for src in good_sources):
                continue

            clean_titles.append(t)

        return clean_titles[:5]

    except:
        return []

def verify_claim(claim, evidence_list):
    if not evidence_list:
        return 0, 0

    claim_emb = sim_model.encode(claim, convert_to_tensor=True)

    scores = []
    for ev in evidence_list:
        ev_emb = sim_model.encode(ev, convert_to_tensor=True)
        scores.append(util.cos_sim(claim_emb, ev_emb).item())

    scores.sort(reverse=True)

    best_score = max(scores)
    coverage = sum(1 for s in scores if s > 0.58)

    return best_score, coverage

# =========================================
# 13. FINAL PREDICTION
# =========================================
def predict_news(text):
    text = clean_text(text)

    if len(text.split()) < 5:
        return "FAKE ❌"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        style_score = torch.sigmoid(logits)[0].item()

    # penalty for fake-like phrases
    if is_suspicious(text):
        style_score -= 0.15

    evidence = fetch_news(build_query(text))
    evidence_score, coverage = verify_claim(text, evidence)

    print("\n🧠 Style:", round(style_score, 4))
    print("🌍 Evidence:", round(evidence_score, 4))
    print("📊 Coverage:", coverage)
    for e in evidence:
        print("-", e)

    if evidence_score >= 0.60:
        label = "REAL"
    elif evidence_score >= 0.50 and coverage >= 1:
        label = "REAL"
    elif evidence_score <= 0.40:
        label = "FAKE"
    else:
        label = "REAL" if style_score > 0.55 else "FAKE"

    return label + (" ✅" if label == "REAL" else " ❌")

# =========================================
# 14. TEST
# =========================================
tests = [
    "NASA announced a new mission to explore Jupiter’s moons.",
    "Scientists discovered humans can live without oxygen permanently.",
    "Apple announced a new iPhone with improved battery life.",
    "A hidden underground civilization was found using unknown technology.",
    "The Federal Reserve raised interest rates to control inflation."
]

for t in tests:
    print("\n======================")
    print("Input:", t)
    print("Prediction:", predict_news(t))