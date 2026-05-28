# Fake News Detection AI

An end-to-end AI-powered fake news detection system using BERT, NLP, FastAPI, and Spring Boot to classify news articles as REAL or FAKE using writing style analysis, semantic similarity, and live evidence retrieval from trusted news sources.

---

## Features

* Fine-tuned BERT model for fake news classification
* Sentence-BERT semantic similarity verification
* Live evidence retrieval from trusted news sources
* Confidence-based REAL/FAKE prediction engine
* Interactive frontend visualizations using Chart.js
* Full-stack architecture using FastAPI + Spring Boot
* Adversarial and conspiracy-language detection
* Real-world news validation workflow

---

## Technologies Used

### AI / Machine Learning

* Python
* PyTorch
* HuggingFace Transformers
* BERT
* Sentence-BERT
* NLP
* Scikit-learn

### Backend

* FastAPI
* REST APIs

### Frontend

* Spring Boot
* Thymeleaf
* HTML
* CSS
* JavaScript
* Chart.js

### Data Sources

* Google News RSS
* Reuters
* BBC
* AP News
* Bloomberg
* CNBC
* New York Times

---

## Repository Structure

```text
fake-news-detection-ai/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── tokenizer.pickle
│
├── frontend/
│   ├── NewsController.java
│   └── index.html
│
├── notebooks/
│   ├── fake_news_lstm_training.py
│   └── fake_news_bert_training.py
│
├── screenshots/
│
└── README.md
```

---

## System Architecture

User Input
↓
Text Preprocessing
↓
BERT Classification
↓
Evidence Retrieval (Google News RSS)
↓
Sentence-BERT Similarity Analysis
↓
REAL / FAKE Prediction Engine
↓
Interactive Visualization Dashboard

---

## Machine Learning Workflow

### LSTM Pipeline

* Data preprocessing and cleaning
* Tokenization and sequence padding
* GloVe embeddings
* Bidirectional LSTM model training
* Evaluation using confusion matrix and classification metrics

### BERT Pipeline

* Fine-tuning `bert-base-uncased`
* Semantic similarity verification
* Evidence-based misinformation validation
* Confidence scoring and prediction logic

---

## Results

* Achieved strong real-world fake news classification performance
* Successfully identified misleading and conspiracy-based content
* Implemented evidence-based verification for improved reliability
* Integrated semantic similarity scoring using Sentence-BERT

---

## Screenshots

Add screenshots here:

* Homepage UI
* REAL prediction example
* FAKE prediction example
* Confidence charts
* Analysis dashboard

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/fake-news-detection-ai.git
cd fake-news-detection-ai
```

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Run FastAPI Backend

```bash
uvicorn backend.app:app --reload
```

### Run Spring Boot Frontend

Run the Spring Boot application and open:

```text
http://localhost:8080
```

---

## Model Files

The trained BERT model (`model.safetensors`) is not included in this repository due to GitHub file size limitations.

To run the project locally:

1. Train the model using:

   * `notebooks/fake_news_bert_training.py`

2. Place trained model files inside:

   * `backend/bert_model/`

Required model files:

* model.safetensors
* config.json
* tokenizer.json
* tokenizer_config.json

---

## Future Improvements

* Docker deployment
* Azure cloud deployment
* Real-time fact-check API integration
* Multilingual fake news detection
* Attention visualization for explainability
* Model optimization for faster inference

---

## Author

Mahek Patel
MS Information Technology & Analytics Graduate
Rutgers Business School

LinkedIn: linkedin.com/in/mahekpatel26
