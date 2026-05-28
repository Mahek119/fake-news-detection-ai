from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import requests

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "C:/Users/capstone/bert_model"
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

sim_model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model Loaded")

# =========================
# REQUEST SCHEMA
# =========================
class NewsRequest(BaseModel):
    text: str

# =========================
# HELPERS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def build_query(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return " ".join(text.split()[:5])


# =========================
# SUSPICIOUS LANGUAGE DETECTOR
# =========================
def is_suspicious(text):

    keywords = [
        "hidden",
        "secret",
        "shocking",
        "they don't want you",
        "underground civilization",
        "ancient aliens",
        "conspiracy",
        "miracle cure",
        "doctors hate",
        "banned",
        "censored",
        "what they don't tell you",
        "deep state",
        "illuminati",
        "microchip",
        "mind control",
        "government hiding",
        "cure cancer",
        "100% proven",
        "scientists baffled"
    ]

    return any(k in text.lower() for k in keywords)


# =========================
# FETCH TRUSTED NEWS
# =========================
def fetch_news(query):

    try:

        url = f"https://news.google.com/rss/search?q={query}"

        res = requests.get(url, timeout=3)

        titles = re.findall(r"<title>(.*?)</title>", res.text)

        # trusted sources
        good_sources = [
            "reuters",
            "bbc",
            "cnn",
            "forbes",
            "ap news",
            "bloomberg",
            "cnbc",
            "the guardian",
            "nytimes",
            "washington post",
            "abc news",
            "nbc news",
            "time",
            "the hill",
            "politico",
            "associated press"
        ]

        clean_titles = []

        for t in titles[1:15]:

            t_lower = t.lower()

            if "google news" in t_lower:
                continue

            if len(t.split()) < 5:
                continue

            # only trusted sources
            if not any(src in t_lower for src in good_sources):
                continue

            clean_titles.append(t)

        return clean_titles[:5]

    except:
        return []


# =========================
# CLAIM VERIFICATION
# =========================
def verify_claim(claim, evidence_list):

    if not evidence_list:
        return 0, 0, False

    claim_emb = sim_model.encode(claim, convert_to_tensor=True)

    scores = []

    fake_detected = False

    # debunk keywords
    fake_keywords = [
        "false",
        "fake",
        "fact check",
        "fact-check",
        "debunked",
        "hoax",
        "misleading",
        "conspiracy",
        "not true",
        "no evidence"
    ]

    for ev in evidence_list:

        ev_lower = ev.lower()

        # detect debunking article
        if any(k in ev_lower for k in fake_keywords):
            fake_detected = True

        ev_emb = sim_model.encode(ev, convert_to_tensor=True)

        score = util.cos_sim(claim_emb, ev_emb).item()

        scores.append(score)

    scores.sort(reverse=True)

    coverage = sum(1 for s in scores if s > 0.58)

    return max(scores), coverage, fake_detected


# =========================
# PREDICTION FUNCTION
# =========================
def predict_news(text):

    text = clean_text(text)

    # =========================
    # SHORT INPUT CHECK
    # =========================
    if len(text.split()) < 5:

        return {
            "prediction": "FAKE",
            "confidence": 0.0,
            "style_score": 0,
            "evidence_score": 0,
            "explanation": "Input too short",
            "evidence": []
        }

    # =========================
    # STYLE ANALYSIS
    # =========================
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

    # suspicious penalty
    if is_suspicious(text):
        style_score -= 0.25

    # =========================
    # FETCH EVIDENCE
    # =========================
    evidence = fetch_news(build_query(text))

    evidence_score, coverage, fake_detected = verify_claim(
        text,
        evidence
    )

    print("\n🧠 Style:", round(style_score, 4))
    print("🌍 Evidence:", round(evidence_score, 4))
    print("📊 Coverage:", coverage)
    print("🚨 Fake Evidence:", fake_detected)

    for e in evidence:
        print("-", e)

    # =========================
    # FINAL DECISION LOGIC
    # =========================

    # trusted sources debunking it
    if fake_detected:

        label = "FAKE"

    elif evidence_score >= 0.70 and coverage >= 2:

        label = "REAL"

    elif evidence_score >= 0.65 and style_score >= 0.60:

        label = "REAL"

    elif evidence_score <= 0.50:

        label = "FAKE"

    elif style_score <= 0.45:

        label = "FAKE"

    else:

        label = "REAL" if style_score > 0.65 else "FAKE"

    # no evidence = fake
    if not evidence:
        label = "FAKE"

    # =========================
    # CONFIDENCE
    # =========================
    confidence = round(max(style_score, evidence_score), 2)

    # =========================
    # EXPLANATION
    # =========================
    if fake_detected:

        explanation = "Trusted sources are debunking this claim"

    elif label == "REAL" and evidence_score >= 0.65:

        explanation = "Strong supporting evidence found"

    elif label == "FAKE" and not evidence:

        explanation = "No trusted source evidence found"

    elif label == "FAKE":

        explanation = "Insufficient evidence from trusted sources"

    elif style_score > 0.65:

        explanation = "Language resembles real news"

    else:

        explanation = "Mixed signals, used fallback"

    # =========================
    # FINAL OUTPUT
    # =========================
    return {

        "prediction": label,

        "confidence": confidence,

        "style_score": round(style_score, 3),

        "evidence_score": round(evidence_score, 3),

        "explanation": explanation,

        "evidence": evidence
    }


# =========================
# API ROUTES
# =========================
@app.get("/health")
def health():

    return {"status": "ok"}


@app.post("/predict")
def predict(req: NewsRequest):

    return predict_news(req.text)
