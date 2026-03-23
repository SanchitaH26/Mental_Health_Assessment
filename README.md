# 🎓 Advanced Student Well-being Assessment

A Streamlit-based mental health screening tool designed for students, using a mixed-input questionnaire powered by multiple NLP/ML models to assess stress, sentiment, emotion, and suicide risk.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Questionnaire Categories](#questionnaire-categories)
- [ML Models Used](#ml-models-used)
- [Configuration](#configuration)
- [Known Issues & Fixes](#known-issues--fixes)
- [Disclaimer](#disclaimer)

---

## Overview

This tool provides a structured, AI-powered mental health screening experience for university students. It combines quantifiable stress scoring (via Likert-scale and select inputs) with deep NLP analysis (via text responses) to generate a comprehensive well-being report covering sentiment, dominant emotions, risk levels, and areas of concern.

---

## Features

- **Mixed-input questionnaire** — combines dropdowns, radio buttons, and free-text responses
- **Real-time NLP analysis** per response — sentiment, emotion detection, and suicide risk scoring
- **Empathetic response generation** for open-text answers using BlenderBot
- **Smoothed emotion tracking** across the session using a rolling history window
- **Aggregate assessment** with an overall risk level and quantifiable stress index
- **Alert system** for detecting suicidal language, negative coping mechanisms, and strong negative emotions
- **Emotion profile bar chart** at the end of the assessment
- **Edit/re-answer** any previously submitted question
- **Sidebar** with live progress tracking and emergency helpline contacts
- **Soothing UI** with custom CSS theming

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Sentiment Analysis | DistilBERT (SST-2) + VADER (ensemble) |
| Emotion Detection | j-hartmann/emotion-english-distilroberta-base |
| Conversational Response | Facebook BlenderBot 400M Distill |
| Suicide Risk Detection | sentinet/suicidality |
| Spell Correction | pyspellchecker |
| Emoji Handling | emoji |
| PDF Export | fpdf2 |
| Data Handling | pandas |

---

## Project Structure

```
nlp_project/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Installation

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv nlp_env
nlp_env\Scripts\activate        # Windows
# source nlp_env/bin/activate   # macOS/Linux
```

### 2. Install PyTorch (CPU version)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install transformers streamlit vaderSentiment pyspellchecker emoji fpdf2 pandas
```

### requirements.txt (reference)

```
torch
transformers
streamlit
vaderSentiment
pyspellchecker
emoji
fpdf2
pandas
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** The first run will download all Hugging Face models (~1–2 GB total). Subsequent runs load from cache and are significantly faster.

---

## How It Works

### Step 1 — Questionnaire
The student answers 7 questions across different well-being categories. Each question uses one of three input types:
- `select` — dropdown with a mapped stress score (1–5)
- `radio` — radio buttons with a mapped stress score (1–5)
- `text` — free-text input processed by full NLP pipeline

### Step 2 — Per-Response Analysis
On submission of each answer, the app runs:
1. Text normalization + spell correction
2. VADER sentiment scoring
3. DistilBERT sentiment scoring → ensemble score
4. Emotion classification (top 5 labels, smoothed over rolling window)
5. Suicide risk classification
6. Alert rule checks (keyphrases, tokens, coping mechanisms, anger)
7. BlenderBot empathetic response (text questions only)

### Step 3 — Aggregate Assessment
After all 7 questions are answered, the app computes:
- Average ML sentiment score
- Dominant smoothed emotion
- Quantifiable stress index (average of scored inputs, out of 5)
- Overall risk level (Low / Moderate / High)
- Categories that triggered alerts

---

## Questionnaire Categories

| # | Category | Input Type | Scoring |
|---|---|---|---|
| 1 | Academic Stress | Select | 1 (low) → 5 (high) |
| 2 | Work-Life Balance | Text | ML sentiment-based |
| 3 | Physical Well-being | Radio | Inverted (less sleep = higher stress) |
| 4 | Social Connection | Select | Inverted (more isolated = higher stress) |
| 5 | Future Anxiety | Text | ML sentiment-based |
| 6 | Motivation & Interest | Radio | 1 (disagree) → 5 (strongly agree) |
| 7 | Coping Mechanisms | Text | ML sentiment-based + coping alert rules |

---

## ML Models Used

### 1. DistilBERT (Sentiment)
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Output: POSITIVE / NEGATIVE + confidence score
- Combined with VADER in a weighted ensemble

### 2. VADER (Sentiment)
- Rule-based lexicon sentiment analyzer
- Provides compound score (−1 to +1)
- Weighted lower for short inputs

### 3. Emotion Classifier
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Labels: joy, sadness, anger, fear, disgust, surprise, neutral
- Top-5 scores tracked and smoothed over a 3-response rolling window

### 4. BlenderBot (Empathetic Response)
- Model: `facebook/blenderbot-400M-distill`
- Generates a conversational, supportive reply for open-text answers

### 5. Suicide Risk Classifier
- Model: `sentinet/suicidality`
- Labels: LABEL_0 (non-suicidal), LABEL_1 (suicidal)
- Confidence mapped to: Safe / Low / Moderate / High / Critical Risk

---

## Configuration

Key thresholds can be adjusted at the top of `app.py`:

```python
SPELL_CORRECTION = True          # Enable/disable spell correction
SENTIMENT_THRESHOLD_POS = 0.35   # Ensemble score above this → POSITIVE
SENTIMENT_THRESHOLD_NEG = -0.35  # Ensemble score below this → NEGATIVE
EMOTION_ALERT_THRESHOLD = 0.65   # Smoothed emotion score to trigger alert
MIN_EMO_CONFIDENCE = 0.50        # Minimum confidence to label an emotion
SHORT_INPUT_LEN = 2              # Word count threshold for short-input weighting
HISTORY_WINDOW = 3               # Number of responses in emotion smoothing window
```

---

## Known Issues & Fixes

### `TypeError: string indices must be integers, not 'str'` in `detect_risk`
**Cause:** `pipeline(..., return_all_scores=True)` returns a nested list `[[{...}, {...}]]`. The original code assumed a flat list.

**Fix applied in `app.py`:**
```python
raw = risk_model(text, truncation=True)
inner = raw[0]
result = inner if isinstance(inner, list) else raw
scores = {item["label"]: item["score"] for item in result if isinstance(item, dict)}
```

### `ModuleNotFoundError: Could not import module 'pipeline'`
**Cause:** PyTorch backend not installed.

**Fix:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade transformers
```

### protobuf version conflicts
**Cause:** Global Python environment has conflicting packages (tensorflow, grpcio, snowflake).

**Fix:** Use a virtual environment (see Installation section) or pin protobuf:
```bash
pip install "protobuf>=3.20.3,<5.0.0"
```

---

## Disclaimer

> ⚠️ This tool is intended for **screening purposes only** and is **not a substitute** for professional mental health diagnosis or treatment. If you or someone you know is in crisis, please contact emergency services or one of the helplines below immediately.

| Region | Helpline | Number |
|---|---|---|
| India | KIRAN Mental Health Helpline | 1800-599-0019 |
| USA | 988 Suicide & Crisis Lifeline | 988 |
| UK | Samaritans | 116 123 |
