# -*- coding: utf-8 -*-
"""MITA_Capstone_Project.ipynb

### Step 1 — Environment setup & data loading

Dataset link - https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Mount Drive & install libraries
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install any missing packages (most are pre-installed in Colab)
!pip install -q tensorflow keras numpy pandas matplotlib seaborn scikit-learn nltk

"""Imports

Import all necessary libraries for data processing, visualization, NLP, and deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

nltk.download('stopwords')
from nltk.corpus import stopwords

print("TF version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

"""Load the data
Load fake and real news datasets, assign labels, and combine into a single shuffled dataset
"""

# Adjust these paths to match where you uploaded the files
FAKE_PATH = '/content/drive/MyDrive/Capstone/ISOT_Fake_News_detection_dataset/Fake.csv'
TRUE_PATH = '/content/drive/MyDrive/Capstone/ISOT_Fake_News_detection_dataset/True.csv'

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

# Add labels: 0 = fake, 1 = real
fake_df['label'] = 0
true_df['label'] = 1

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

df

"""### Step 2 — Exploratory Data Analysis (EDA)

Basic stats

Explore dataset structure, columns, missing values, and class distribution
"""

# Shape and column overview
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nLabel distribution:\n", df['label'].value_counts())
print("\nSample fake article title:", df[df['label']==0]['title'].iloc[10])
print("Sample real article title:", df[df['label']==1]['title'].iloc[10])

"""Combine title + text, measure length

Combine title and text into a single content column and compute word count for each article
"""

# Merge title and body into one field — more signal for the model
df['content'] = df['title'] + ' ' + df['text']

# Word count per article
df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))

print(df.groupby('label')['word_count'].describe().round(1))

"""Class balance bar chart

Visualize class balance and distribution of article lengths for fake vs real news
"""

'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Class balance
label_counts = df['label'].value_counts()
axes[0].bar(['Fake (0)', 'Real (1)'], label_counts.values, color=['#2563EB','#38BDF8'], edgecolor='white')
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Number of Articles')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')

# Word count distribution
df[df['label']==0]['word_count'].plot(kind='hist', bins=60, alpha=0.6, color='#2563EB', label='Fake', ax=axes[1])
df[df['label']==1]['word_count'].plot(kind='hist', bins=60, alpha=0.6, color='#38BDF8', label='Real', ax=axes[1])
axes[1].set_title('Article Length Distribution (Word Count)')
axes[1].set_xlabel('Words per Article')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].set_xlim(0, 2000)

plt.tight_layout()
plt.show()
'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#EAF2FF')

# Background color for each chart
for ax in axes:
    ax.set_facecolor('#F8FAFC')

# =========================
# CLASS DISTRIBUTION
# =========================
label_counts = df['label'].value_counts()

axes[0].bar(
    ['Fake (0)', 'Real (1)'],
    label_counts.values,
    color=['#2563EB', '#38BDF8'],
    edgecolor='white',
    linewidth=1.5
)

axes[0].set_title(
    'Class Distribution',
    fontsize=14,
    fontweight='bold',
    color='#0F172A'
)

axes[0].set_ylabel(
    'Number of Articles',
    color='#0F172A'
)

axes[0].tick_params(colors='#0F172A')

for i, v in enumerate(label_counts.values):
    axes[0].text(
        i,
        v + 100,
        str(v),
        ha='center',
        fontweight='bold',
        color='#0F172A'
    )

# =========================
# WORD COUNT DISTRIBUTION
# =========================
df[df['label']==0]['word_count'].plot(
    kind='hist',
    bins=60,
    alpha=0.7,
    color='#2563EB',
    label='Fake',
    ax=axes[1]
)

df[df['label']==1]['word_count'].plot(
    kind='hist',
    bins=60,
    alpha=0.7,
    color='#38BDF8',
    label='Real',
    ax=axes[1]
)

axes[1].set_title(
    'Article Length Distribution',
    fontsize=14,
    fontweight='bold',
    color='#0F172A'
)

axes[1].set_xlabel(
    'Words per Article',
    color='#0F172A'
)

axes[1].set_ylabel(
    'Frequency',
    color='#0F172A'
)

axes[1].tick_params(colors='#0F172A')

axes[1].legend(
    facecolor='#F8FAFC',
    edgecolor='white'
)

axes[1].set_xlim(0, 2000)

# Remove top/right borders
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

""" Top keywords — fake vs real"""

'''
from collections import Counter

stop_words = set(stopwords.words('english'))

def get_top_words(subset, n=20):
    words = ' '.join(subset['content'].astype(str)).lower().split()
    words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
    return Counter(words).most_common(n)

fake_words = get_top_words(df[df['label']==0])
real_words = get_top_words(df[df['label']==1])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, words, color, title in zip(
    axes,
    [fake_words, real_words],
    ['#D85A30', '#1D9E75'],
    ['Top 20 Words — Fake News', 'Top 20 Words — Real News']
):
    labels, counts = zip(*words)
    ax.barh(labels[::-1], counts[::-1], color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel('Frequency')

plt.tight_layout()
plt.show()
'''
from collections import Counter

stop_words = set(stopwords.words('english'))

# =========================
# TOP WORDS FUNCTION
# =========================
def get_top_words(subset, n=20):
    words = ' '.join(subset['content'].astype(str)).lower().split()

    words = [
        w for w in words
        if w.isalpha()
        and w not in stop_words
        and len(w) > 2
    ]

    return Counter(words).most_common(n)

# Get words
fake_words = get_top_words(df[df['label']==0])
real_words = get_top_words(df[df['label']==1])

# =========================
# CREATE FIGURE
# =========================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
    facecolor='#EAF2FF'
)

# Chart background
for ax in axes:
    ax.set_facecolor('#F8FAFC')

# =========================
# PLOT CHARTS
# =========================
for ax, words, color, title in zip(
    axes,
    [fake_words, real_words],
    ['#2563EB', '#38BDF8'],
    ['Top 20 Words — Fake News', 'Top 20 Words — Real News']
):

    labels, counts = zip(*words)

    ax.barh(
        labels[::-1],
        counts[::-1],
        color=color,
        alpha=0.9
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight='bold',
        color='#0F172A'
    )

    ax.set_xlabel(
        'Frequency',
        color='#0F172A'
    )

    ax.tick_params(
        colors='#0F172A'
    )

    # Remove extra borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

"""Subject category breakdown

"""

'''
if 'subject' in df.columns:
    subject_counts = df.groupby(['subject','label']).size().unstack(fill_value=0)
    subject_counts.plot(kind='barh', figsize=(10, 6), color=['#D85A30','#1D9E75'])
    plt.title('Article Count by Subject Category')
    plt.xlabel('Number of Articles')
    plt.legend(['Fake', 'Real'])
    plt.tight_layout()
    plt.show()
'''
if 'subject' in df.columns:

    # =========================
    # SUBJECT COUNTS
    # =========================
    subject_counts = df.groupby(
        ['subject', 'label']
    ).size().unstack(fill_value=0)

    # =========================
    # CREATE FIGURE
    # =========================
    fig, ax = plt.subplots(
        figsize=(10, 6),
        facecolor='#EAF2FF'
    )

    ax.set_facecolor('#F8FAFC')

    # =========================
    # PLOT
    # =========================
    subject_counts.plot(
        kind='barh',
        color=['#2563EB', '#38BDF8'],
        edgecolor='white',
        linewidth=1.2,
        ax=ax
    )

    # =========================
    # TITLES & LABELS
    # =========================
    ax.set_title(
        'Article Count by Subject Category',
        fontsize=15,
        fontweight='bold',
        color='#0F172A'
    )

    ax.set_xlabel(
        'Number of Articles',
        fontsize=12,
        color='#0F172A'
    )

    ax.set_ylabel(
        'Subject',
        fontsize=12,
        color='#0F172A'
    )

    # =========================
    # LEGEND
    # =========================
    legend = ax.legend(
        ['Fake', 'Real'],
        facecolor='#F8FAFC',
        edgecolor='white'
    )

    for text in legend.get_texts():
        text.set_color('#0F172A')

    # =========================
    # AXIS STYLING
    # =========================
    ax.tick_params(
        colors='#0F172A'
    )

    # Remove extra borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

from wordcloud import WordCloud

text = " ".join(df['text'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

"""### Step 3 — Text preprocessing pipeline

Text cleaning function

 Define function to clean text by removing noise such as punctuation, URLs, HTML tags, and extra spaces
 Apply text cleaning function to all articles to create a clean_text column
"""

stop_words = set(stopwords.words('english'))                           #Loads common words like:the, is, in, and

def clean_text(text):                                                     #function to clean one article at a time
    text = str(text).lower()                                            # lowercase
    text = re.sub(r'\[.*?\]', '', text)                                    # remove brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)                            # remove URLs
    text = re.sub(r'<.*?>+', '', text)                                  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)                     # remove punctuation
    text = re.sub(r'\n', ' ', text)                                           # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)                                  # remove words with numbers
    text = re.sub(r'\s+', ' ', text).strip()                               # collapse whitespace
    text = re.sub(r'^.*?\(Reuters\)\s?-', '', text)                     # Removes "CITY (Reuters) -"
    return text

# Applies function to every row ,Creates new column
df['clean_text'] = df['content'].apply(clean_text)

# Quick check
print(df['clean_text'].iloc[0][:300])

"""Train/test split

Split dataset into training and testing sets with balanced class distribution
"""

X = df['clean_text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")
print(f"Train label balance: {y_train.mean():.3f}")
print(f"Test label balance:  {y_test.mean():.3f}")

"""Tokenizer — fit on training data only

Initialize tokenizer to convert text into numerical sequences with a fixed vocabulary size

Fit tokenizer only on training data to build word-to-index mapping and avoid data leakage
"""

VOCAB_SIZE  = 30000   # top N words to keep
MAX_LEN     = 500     # max tokens per article (covers ~85% of articles)
OOV_TOKEN   = "<OOV>" # out-of-vocabulary placeholder

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)   #  only fit on training set, never test

word_index = tokenizer.word_index  #Maps words → numbers
print(f"Unique tokens in vocabulary: {len(word_index):,}")
print(f"Sample entries: { {k: word_index[k] for k in list(word_index)[:8]} }")

"""Convert text to padded sequences

Convert cleaned text into sequences of numerical tokens for both training and testing data

Pad or truncate sequences to ensure all inputs have the same fixed length
"""

# Convert text → integer sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

# Pad / truncate to MAX_LEN (post-padding is standard for LSTM)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

print("X_train_pad shape:", X_train_pad.shape)  # should be (35918, 500)
print("X_test_pad shape: ", X_test_pad.shape)   # should be (8980, 500)
print("\nSample sequence (first 20 tokens):", X_train_pad[0][:20])

""" Verify coverage at MAX_LEN=500

Analyze how many articles fall within the chosen maximum sequence length (MAX_LEN)

Visualize distribution of token sequence lengths and highlight MAX_LEN threshold
"""

train_lengths = [len(s) for s in X_train_seq]
coverage = sum(1 for l in train_lengths if l <= MAX_LEN) / len(train_lengths)
print(f"Articles fully covered at MAX_LEN={MAX_LEN}: {coverage:.1%}")

plt.figure(figsize=(8, 3))
plt.hist(train_lengths, bins=60, color='#7F77DD', alpha=0.8, edgecolor='white')
plt.axvline(MAX_LEN, color='#D85A30', linewidth=2, linestyle='--', label=f'MAX_LEN={MAX_LEN}')
plt.xlabel('Sequence length (tokens)')
plt.ylabel('Frequency')
plt.title('Token sequence lengths after cleaning')
plt.legend()
plt.tight_layout()
plt.show()

"""```
# This is formatted as code
```

### Step 4 — Word embeddings with GloVe

Download GloVe 100d
"""

!wget -q --show-progress http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip -d glove/
!ls glove/

"""Load GloVe vectors into a dictionary

Load GloVe embeddings into a dictionary mapping words to vector representations
"""

EMBEDDING_DIM = 100
GLOVE_PATH = 'glove/glove.6B.100d.txt'

glove_embeddings = {}  #Empty dictionary to store:

with open(GLOVE_PATH, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32') #Extract vector
        glove_embeddings[word] = vector #Store in dictionary

print(f"GloVe vectors loaded: {len(glove_embeddings):,}")
print(f"Vector dimension:     {next(iter(glove_embeddings.values())).shape}")

# Quick sanity check
print(f"\n'trump' in GloVe: {'trump' in glove_embeddings}")
print(f"'fake'  in GloVe: {'fake'  in glove_embeddings}")
print(f"'news'  in GloVe: {'news'  in glove_embeddings}")

"""Build the embedding matrix

Create embedding matrix to map tokenizer word indices to corresponding GloVe vectors
Evaluate how many words in vocabulary are covered by GloVe embeddings
"""

# How many words we'll actually embed (capped at VOCAB_SIZE)
num_words = min(VOCAB_SIZE, len(word_index) + 1)

# Initialize with zeros — words not in GloVe stay as zero vectors
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

covered = 0
missed  = 0

for word, idx in word_index.items():
    if idx >= num_words:
        continue
    vector = glove_embeddings.get(word)
    if vector is not None:
        embedding_matrix[idx] = vector
        covered += 1
    else:
        missed += 1

coverage_pct = covered / (covered + missed) * 100
print(f"Embedding matrix shape:  {embedding_matrix.shape}")
print(f"Words covered by GloVe:  {covered:,}  ({coverage_pct:.1f}%)")
print(f"Words missing (OOV):     {missed:,}  ({100-coverage_pct:.1f}%)")

"""Visualize a few word vectors

Reduce high-dimensional word vectors to 2D using PCA for visualization of word relationships
"""

from sklearn.decomposition import PCA #Converts high-dimensional data → lower dimensions , 100D → 2D

# Pick words we know matter from EDA
probe_words = ['trump', 'clinton', 'obama', 'said', 'government',
               'fake', 'news', 'republican', 'house', 'washington',
               'hillary', 'president', 'people', 'state', 'donald']

# Filter to words in GloVe
vectors = []
labels  = []
for w in probe_words:
    if w in glove_embeddings:
        vectors.append(glove_embeddings[w])
        labels.append(w)

# Reduce to 2D with PCA
pca = PCA(n_components=2, random_state=42)  #100 dimensions → 2 dimensions
coords = pca.fit_transform(np.array(vectors))  #Converts vectors into 2D points

plt.figure(figsize=(9, 6))
plt.scatter(coords[:, 0], coords[:, 1], color='#7F77DD', s=60, alpha=0.8)
for i, label in enumerate(labels):
    plt.annotate(label, (coords[i, 0]+0.01, coords[i, 1]+0.01), fontsize=10)
plt.title('PCA projection of GloVe vectors (probe words)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show() #Words with similar meaning appear close

"""### Step 5 — Building and Training the LSTM Model

Define the Architecture

Build a Bidirectional LSTM model using pre-trained GloVe embeddings for text classification

Compile the model with optimizer, loss function, and evaluation metrics
"""

model = Sequential([                                                          #Layers are stacked one after another (simple pipeline)

    tf.keras.Input(shape=(MAX_LEN,)), # Explicitly define input shape
    # Embedding layer initialized with GloVe weights
    Embedding(input_dim=num_words,
              output_dim=EMBEDDING_DIM,
              weights=[embedding_matrix],
              #input_length=MAX_LEN,
              trainable=False),

    # Bidirectional LSTM to capture context from both directions
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),

    # Second LSTM layer to extract higher-level patterns
    Bidirectional(LSTM(32)),
    Dropout(0.3),

    # Dense layers for classification
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification (0 or 1)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

"""Set Callbacks & Train

Define training parameters and early stopping to prevent overfitting

Train the LSTM model on training data with validation split
"""

# Stop training if validation loss doesn't improve for 3 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Use 10% of training data for validation
    callbacks=[early_stop],
    verbose=1
)

"""### Step 6 — Model Evaluation

Evaluate on Test Data
"""

# Evaluate the model on the unseen test set
test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

"""Classification Report & Confusion Matrix

Generate detailed classification metrics including precision, recall, and F1-score

Visualize model predictions using confusion matrix
"""

# Get predictions
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Print the full report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""Plot Training History"""

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

plot_history(history)

"""Final Step: The "Manual" Test"""

def predict_news(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded)[0][0]

    label = "REAL" if prediction > 0.5 else "FAKE"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

# Test it out!
predict_news("Breaking: The moon is actually made of green cheese, NASA confirms in secret document.")
predict_news(" Time Magazine Humiliates Trump After He Lies About Award")
predict_news("UNITED NATIONS (Reuters) - Two North Korean shipments to a Syrian government agency responsible for the country s chemical weapons program were intercepted in the past six months, according to a confidential United Nations report on North Korea sanctions violations. The report by a panel of independent U.N. experts, which was submitted to the U.N. Security Council earlier this month and seen by Reuters on Monday, gave no details on when or where the interdictions occurred or what the shipments contained.   The panel is investigating reported prohibited chemical, ballistic missile and conventional arms cooperation between Syria and the DPRK (North Korea),  the experts wrote in the 37-page report.   Two member states interdicted shipments destined for Syria. Another Member state informed the panel that it had reasons to believe that the goods were part of a KOMID contract with Syria,  according to the report. KOMID is the Korea Mining Development Trading Corporation. It was blacklisted by the Security Council in 2009 and described as Pyongyang s key arms dealer and exporter of equipment related to ballistic missiles and conventional weapons. In March 2016 the council also blacklisted two KOMID representatives in Syria.   The consignees were Syrian entities designated by the European Union and the United States as front companies for Syria s Scientific Studies and Research Centre (SSRC), a Syrian entity identified by the Panel as cooperating with KOMID in previous prohibited item transfers,  the U.N. experts wrote.  SSRC has overseen the country s chemical weapons program since the 1970s. The U.N. experts said activities between Syria and North Korea they were investigating included cooperation on Syrian Scud missile programs and maintenance and repair of Syrian surface-to-air missiles air defense systems. The North Korean and Syrian missions to the United Nations did not immediately respond to a request for comment.  The experts said they were also investigating the use of the VX nerve agent in Malaysia to kill the estranged half-brother of North Korea s leader Kim Jong Un in February.  North Korea has been under U.N. sanctions since 2006 over its ballistic missile and nuclear programs and the Security Council has ratcheted up the measures in response to five nuclear weapons tests and four long-range missile launches. Syria agreed to destroy its chemical weapons in 2013 under a deal brokered by Russia and the United States. However, diplomats and weapons inspectors suspect Syria may have secretly maintained or developed a new chemical weapons capability. During the country s more than six-year long civil war the Organisation for the Prohibition of Chemical Weapons has said the banned nerve agent sarin has been used at least twice, while the use of chlorine as a weapon has been widespread. The Syrian government has repeatedly denied using chemical weapons")
predict_news("U.S. House approves $81 billion for disaster aid")
predict_news("Google announced new AI tools for developers at its annual conference.")
predict_news("NASA announced a new mission to explore Jupiter’s moons.")

# Save the model
model.save('/content/drive/MyDrive/Capstone/fake_news_lstm_model.keras')

# Save the tokenizer (essential for processing new text later)
import pickle
with open('/content/drive/MyDrive/Capstone/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and Tokenizer saved to Google Drive!")