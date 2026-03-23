import streamlit as st
import re
import emoji
from collections import deque, defaultdict
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
import warnings
import pandas as pd
from datetime import datetime
from fpdf import FPDF, XPos, YPos
from io import BytesIO
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================================
# MODEL INITIALIZATION (Cached for performance)
# ==========================================================================
@st.cache_resource
def load_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=5
    )
    vader = SentimentIntensityAnalyzer()
    spell = SpellChecker()
    blenderbot_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
    risk_model = pipeline(
        "text-classification",
        model="sentinet/suicidality",
        return_all_scores=True
    )
    return sentiment_pipe, emotion_pipe, vader, spell, blenderbot_tokenizer, blenderbot_model, risk_model

try:
    sentiment_pipe, emotion_pipe, vader, spell, blenderbot_tokenizer, blenderbot_model, risk_model = load_models()
except Exception as e:
    st.error(f"Error loading ML models. Please check environment configuration: {e}")
    st.stop()


# ==========================================================================
# CORE ANALYSIS CONFIGURATION
# ==========================================================================
SPELL_CORRECTION = True
SENTIMENT_THRESHOLD_POS = 0.35
SENTIMENT_THRESHOLD_NEG = -0.35
EMOTION_ALERT_THRESHOLD = 0.65
MIN_EMO_CONFIDENCE = 0.50
SHORT_INPUT_LEN = 2
HISTORY_WINDOW = 3

SUICIDAL_KEYPHRASES = {
    "i want to die", "i wanna die", "i'm gonna kill myself", "kill myself",
    "i want to kill myself", "end my life", "i don't want to live", "i dont want to live",
}
SUICIDAL_TOKENS = {"die", "suicide", "kill", "kys"}
ANGER_KEYWORDS = {"idiot", "fool", "stupid", "trash", "jerk", "moron", "bastard", "hate"}

# ==========================================================================
# ADVANCED STUDENT-SPECIFIC QUESTIONNAIRE (MIXED INPUTS)
# ==========================================================================
STUDENT_QUESTIONNAIRE = [
    {
        "id": "q1",
        "question": "How often do you feel **overwhelmed or excessively stressed** by your academic workload?",
        "category": "Academic Stress",
        "type": "select",
        "options": [
            ("1 - Rarely or Never", 1),
            ("2 - Sometimes", 2),
            ("3 - Often", 3),
            ("4 - Very Frequently", 4),
            ("5 - Constantly or Always", 5),
        ]
    },
    {
        "id": "q2",
        "question": "Describe your current experience with **balancing studies and personal life**.",
        "category": "Work-Life Balance",
        "type": "text",
        "placeholder": "e.g., I barely have time for myself, or I feel I manage it well...",
        "default_score": 3
    },
    {
        "id": "q3",
        "question": "On average, how many hours of quality **sleep** do you get per night?",
        "category": "Physical Well-being",
        "type": "radio",
        "options": [
            ("Less than 5 hours", 5),
            ("5 to 6 hours", 4),
            ("6 to 7 hours", 3),
            ("7 to 8 hours", 2),
            ("More than 8 hours", 1),
        ],
        "inversion_score": True
    },
    {
        "id": "q4",
        "question": "How would you rate your **sense of connection and belonging** with peers at university?",
        "category": "Social Connection",
        "type": "select",
        "options": [
            ("1 - Extremely Isolated", 5),
            ("2 - Neutral, I keep to myself", 3),
            ("3 - Moderately Connected", 2),
            ("4 - Strong sense of belonging", 1),
        ],
        "inversion_score": True
    },
    {
        "id": "q5",
        "question": "What are your primary thoughts when you consider your **future career and academic goals**?",
        "category": "Future Anxiety",
        "type": "text",
        "placeholder": "e.g., I'm constantly worried about getting a job, or I feel confident about my path...",
        "default_score": 3
    },
    {
        "id": "q6",
        "question": "I've lost interest or **motivation** in activities I used to enjoy (hobbies, sports, socializing).",
        "category": "Motivation & Interest",
        "type": "radio",
        "options": [
            ("1 - Strongly Disagree", 1),
            ("2 - Disagree", 2),
            ("3 - Neutral", 3),
            ("4 - Agree", 4),
            ("5 - Strongly Agree", 5),
        ]
    },
    {
        "id": "q7",
        "question": "Describe how you typically **cope with high stress or difficult emotions**.",
        "category": "Coping Mechanisms",
        "type": "text",
        "placeholder": "e.g., I exercise, I talk to family, or I tend to smoke/isolate myself...",
        "default_score": 3
    }
]

# ==========================================================================
# CORE HELPER FUNCTIONS
# ==========================================================================
def normalize_text(text: str) -> str:
    text = emoji.demojize(text)
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_spell_fix(text: str) -> str:
    words = text.split()
    miss = spell.unknown(words)
    for w in miss:
        corr = spell.correction(w)
        if corr and corr != w and corr.isalpha():
            text = re.sub(rf'\b{re.escape(w)}\b', corr, text)
    return text

def contains_any_phrase(text_lower: str, phrases):
    for p in phrases:
        if p in text_lower:
            return True, p
    return False, None

def tokenize_words(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def average_emotion_distribution(history_deque):
    if not history_deque: return {}
    agg = defaultdict(float)
    for d in history_deque:
        for k, v in d.items():
            agg[k] += v
    n = len(history_deque)
    for k in list(agg.keys()):
        agg[k] = agg[k] / n
    return dict(agg)

# ==========================================================================
# CORE ANALYSIS FUNCTION
# ==========================================================================
def analyze_text(raw_text: str, emotion_history: deque):
    text = normalize_text(raw_text)
    if SPELL_CORRECTION:
        text = simple_spell_fix(text)
    text_l = text.lower()
    words = tokenize_words(text)

    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores["compound"]

    tr = sentiment_pipe(text)[0]
    tr_label = tr.get("label", "").upper()
    tr_score = float(tr.get("score", 0.0))
    signed_tr_score = tr_score if tr_label.startswith("POS") else -tr_score

    weight_tr = 1.0 if len(words) > SHORT_INPUT_LEN else 0.4
    ensemble_score = (signed_tr_score * weight_tr + vader_compound) / (1.0 + weight_tr)

    if ensemble_score > SENTIMENT_THRESHOLD_POS: ensemble_label = "POSITIVE"
    elif ensemble_score < SENTIMENT_THRESHOLD_NEG: ensemble_label = "NEGATIVE"
    else: ensemble_label = "NEUTRAL"

    emo_out = emotion_pipe(text)
    emo_list = emo_out[0] if isinstance(emo_out, list) and len(emo_out) > 0 else emo_out
    emo_dist = {d['label']: float(d['score']) for d in emo_list}
    emotion_history.append(emo_dist)
    avg_dist = average_emotion_distribution(emotion_history)

    raw_top_label, raw_top_score = max(emo_dist.items(), key=lambda kv: kv[1]) if emo_dist else (None, 0.0)
    smooth_top_label, smooth_top_score = max(avg_dist.items(), key=lambda kv: kv[1]) if avg_dist else (None, 0.0)

    if smooth_top_score < MIN_EMO_CONFIDENCE: smooth_top_label = "neutral"

    found_suicide, matched_suicide = contains_any_phrase(text_l, SUICIDAL_KEYPHRASES)
    found_suicide_token = any(t in SUICIDAL_TOKENS for t in words)
    found_suicide = found_suicide or found_suicide_token
    found_anger = any(w in ANGER_KEYWORDS for w in words)

    alert = False
    alert_reasons = []
    if found_suicide:
        alert = True
        alert_reasons.append(f"Suicidal phrase/token detected: '{matched_suicide or 'single-token match'}'")
    if ensemble_label == "NEGATIVE" and smooth_top_label in ("sadness", "fear") and smooth_top_score >= EMOTION_ALERT_THRESHOLD:
        alert = True
        alert_reasons.append("Negative sentiment + strong sadness/fear (smoothed)")
    if text_l.count("smoke") > 0 or text_l.count("drink") > 0 or text_l.count("isolate") > 0:
        if raw_top_label in ("sadness", "anger", "fear"):
            alert = True
            alert_reasons.append("Detection of negative coping mechanism (smoking, drinking, isolating) under stress.")
    if found_anger and smooth_top_label == "anger" and smooth_top_score >= EMOTION_ALERT_THRESHOLD:
        alert = True
        alert_reasons.append("Insult words + strong anger (smoothed)")

    return {
        "ensemble_sentiment": {"label": ensemble_label, "score": round(ensemble_score, 3)},
        "raw_top_emotion": (raw_top_label, round(raw_top_score, 3)),
        "smoothed_emotion_dist": {k: round(v, 3) for k, v in avg_dist.items()},
        "alert": alert,
        "alert_reasons": alert_reasons,
    }

# ==========================================================================
# EMPATHETIC RESPONSE GENERATOR
# ==========================================================================
def get_empathetic_response(user_input):
    inputs = blenderbot_tokenizer(user_input, return_tensors="pt")
    outputs = blenderbot_model.generate(**inputs, max_length=100)
    response = blenderbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ==========================================================================
# SUICIDE RISK DETECTOR — FIXED
# ==========================================================================
def map_risk(confidence_percent, label):
    if label == "LABEL_0":
        return "Safe"
    else:
        if confidence_percent < 50: return "Low Risk"
        elif confidence_percent < 75: return "Moderate Risk"
        elif confidence_percent < 90: return "High Risk"
        else: return "Critical Risk"

def detect_risk(text):
    """
    FIX: risk_model with return_all_scores=True returns a list of lists:
      [[{"label": "LABEL_0", "score": 0.9}, {"label": "LABEL_1", "score": 0.1}]]
    We unwrap both the outer list and handle both list-of-dicts and dict formats.
    """
    raw = risk_model(text, truncation=True)

    # Unwrap nested list: [[{...}, {...}]] → [{...}, {...}]
    if isinstance(raw, list) and len(raw) > 0:
        inner = raw[0]
        if isinstance(inner, list):
            result = inner          # [[{...}]] → [{...}]
        elif isinstance(inner, dict):
            result = raw            # [{...}] already flat
        else:
            result = raw
    else:
        result = raw

    # Build scores dict safely
    scores = {}
    for item in result:
        if isinstance(item, dict):
            scores[item["label"]] = item["score"]
        # If item is somehow a string (shouldn't happen), skip it
        else:
            continue

    label = "LABEL_1" if scores.get("LABEL_1", 0) > scores.get("LABEL_0", 0) else "LABEL_0"
    confidence_percent = round(scores.get(label, 0) * 100, 2)
    risk_category = map_risk(confidence_percent, label)

    return {
        "prediction": "Suicidal" if label == "LABEL_1" else "Non-Suicidal",
        "confidence_percent": confidence_percent,
        "risk_category": risk_category,
    }

# ==========================================================================
# AGGREGATE ANALYSIS ACROSS ALL RESPONSES
# ==========================================================================
def calculate_overall_assessment(responses_data):
    total_sentiment = 0
    quantifiable_stress_sum = 0
    num_quantifiable = 0
    emotion_aggregates = defaultdict(float)
    high_risk_count = 0
    alert_count = 0
    categories_at_risk = []

    for resp in responses_data:
        total_sentiment += resp['analysis']['ensemble_sentiment']['score']

        quantifiable_stress_sum += resp.get('quantifiable_score', 0)
        if resp.get('quantifiable_score', 0) > 0:
            num_quantifiable += 1

        for emotion, score in resp['analysis']['smoothed_emotion_dist'].items():
            emotion_aggregates[emotion] += score

        if resp['analysis']['alert']:
            alert_count += 1
            categories_at_risk.append(resp['category'])

        if resp['risk']['risk_category'] in ["High Risk", "Critical Risk"]:
            high_risk_count += 1

    num_responses = len(responses_data)

    avg_sentiment = total_sentiment / num_responses if num_responses > 0 else 0
    avg_quantifiable_stress = quantifiable_stress_sum / num_quantifiable if num_quantifiable > 0 else 0

    for emotion in emotion_aggregates:
        emotion_aggregates[emotion] /= num_responses

    filtered_emotions = {k: v for k, v in emotion_aggregates.items() if k not in ('joy', 'neutral')}
    if filtered_emotions:
        dominant_emotion = max(filtered_emotions.items(), key=lambda x: x[1])
    else:
        dominant_emotion = max(emotion_aggregates.items(), key=lambda x: x[1]) if emotion_aggregates else ("neutral", 0)

    if high_risk_count >= 1 or alert_count >= 3 or avg_quantifiable_stress > 4.0:
        overall_risk = "High - Professional Help Recommended"
        risk_color = "🟠"
    elif alert_count >= 2 or avg_quantifiable_stress > 3.0:
        overall_risk = "Moderate - Monitor Closely"
        risk_color = "🟡"
    else:
        overall_risk = "Low - General Support May Help"
        risk_color = "🟢"

    return {
        "overall_sentiment": avg_sentiment,
        "dominant_emotion": dominant_emotion,
        "overall_risk": overall_risk,
        "risk_color": risk_color,
        "alert_responses": alert_count,
        "categories_at_risk": list(set(categories_at_risk)),
        "emotion_profile": dict(emotion_aggregates),
        "quantifiable_stress_index": round(avg_quantifiable_stress, 2)
    }

# ==========================================================================
# STREAMLIT UI
# ==========================================================================

st.set_page_config(
    page_title="Advanced Student Mental Health Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%);
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px;
        border: 2px solid #64b5f6;
        background-color: #ffffff;
        font-size: 16px;
    }
    .stButton>button {
        border: none;
        border-radius: 10px;
        color: #ffffff;
        background: linear-gradient(135deg, #4db6ac 0%, #26a69a 100%);
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(77, 182, 172, 0.4);
    }
    .question-card {
        background-color: #f8f9fa;
        border-left: 4px solid #64b5f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .category-badge {
        background-color: #4db6ac;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 2px solid #b2dfdb;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(100, 181, 246, 0.1);
    }
    .high-stress-metric {
        background-color: #fff3e0;
        border-color: #ff9800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=HISTORY_WINDOW)
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'assessment_complete' not in st.session_state:
    st.session_state.assessment_complete = False

# Header
st.title("🎓 Advanced Student Well-being Assessment")
st.markdown(
    "**Focused on academic life, this tool uses a mixed-input approach for a detailed and accurate analysis.** "
    "Your honest answers help gauge stress, sentiment, and risk levels."
)
st.warning("⚠️ **Disclaimer:** This tool is for screening purposes only and is not a substitute for professional mental health care. If you're in crisis, please contact emergency services or a crisis helpline immediately.")

# Sidebar
with st.sidebar:
    st.header("📊 Progress")
    progress = len(st.session_state.responses) / len(STUDENT_QUESTIONNAIRE)
    st.progress(progress)
    st.metric("Questions Answered", f"{len(st.session_state.responses)}/{len(STUDENT_QUESTIONNAIRE)}")

    st.markdown("---")
    st.header("🆘 Emergency Contacts")
    st.markdown("""
    **Crisis Helpline (India):** KIRAN Helpline 📞 1800-599-0019
    
    **Crisis Helpline (US):** 988 Suicide & Crisis Lifeline 📞 988
    
    **UK:** Samaritans 📞 116 123
    """)

    if st.button("🔄 Reset Assessment"):
        st.session_state.responses = {}
        st.session_state.current_question = 0
        st.session_state.assessment_complete = False
        st.session_state.emotion_history = deque(maxlen=HISTORY_WINDOW)
        st.rerun()

# Main content area
if not st.session_state.assessment_complete:
    st.header("📝 Assessment Questionnaire")

    for idx, q in enumerate(STUDENT_QUESTIONNAIRE):
        is_current = (idx == st.session_state.current_question)
        is_answered = q['id'] in st.session_state.responses

        with st.expander(f"Question {idx + 1}: {q['category']}", expanded=is_current):
            st.markdown(f"<div class='question-card'>", unsafe_allow_html=True)
            st.markdown(f"<span class='category-badge'>{q['category']}</span>", unsafe_allow_html=True)
            st.markdown(f"**{q['question']}**")
            st.markdown("</div>", unsafe_allow_html=True)

            if is_answered:
                resp = st.session_state.responses[q['id']]
                st.success(f"✅ Answered (Score: {resp.get('quantifiable_score', 'N/A')}): {resp['answer'][:100]}...")
                if st.button(f"Edit/Re-Answer", key=f"edit_{q['id']}"):
                    del st.session_state.responses[q['id']]
                    if resp['type'] == 'text':
                        st.session_state.emotion_history.pop()
                    st.session_state.current_question = idx
                    st.rerun()
            elif is_current:
                answer = None
                quantifiable_score = 0

                if q['type'] == 'text':
                    answer = st.text_area(
                        "Your detailed response:",
                        placeholder=q['placeholder'],
                        height=120,
                        key=f"input_{q['id']}"
                    )
                    quantifiable_score = q.get("default_score", 3)

                elif q['type'] in ('select', 'radio'):
                    options_list = [opt[0] for opt in q['options']]

                    if q['type'] == 'select':
                        selected_option_label = st.selectbox(
                            "Choose the best fit:",
                            options_list,
                            index=0,
                            key=f"input_{q['id']}"
                        )
                    else:
                        selected_option_label = st.radio(
                            "Choose the best fit:",
                            options_list,
                            index=0,
                            key=f"input_{q['id']}"
                        )

                    selected_score = next((opt[1] for opt in q['options'] if opt[0] == selected_option_label), 0)
                    answer = selected_option_label
                    quantifiable_score = selected_score

                if st.button(f"Submit Answer {idx + 1}", key=f"submit_{q['id']}"):
                    if answer and (q['type'] == 'text' and answer.strip() or q['type'] != 'text' and answer is not None):
                        with st.spinner("Analyzing your response..."):
                            ml_input = answer if q['type'] == 'text' else q['question'] + " " + answer

                            analysis = analyze_text(ml_input, st.session_state.emotion_history)
                            risk = detect_risk(ml_input)
                            empathetic = get_empathetic_response(ml_input) if q['type'] == 'text' else None

                            st.session_state.responses[q['id']] = {
                                'answer': answer,
                                'category': q['category'],
                                'type': q['type'],
                                'quantifiable_score': quantifiable_score,
                                'analysis': analysis,
                                'risk': risk,
                                'empathetic_response': empathetic,
                                'timestamp': datetime.now().isoformat()
                            }

                            st.success("✅ Response recorded!")

                            if idx < len(STUDENT_QUESTIONNAIRE) - 1:
                                st.session_state.current_question = idx + 1

                            st.rerun()
                    else:
                        st.error("Please provide an answer before submitting.")

    if len(st.session_state.responses) == len(STUDENT_QUESTIONNAIRE):
        st.markdown("---")
        st.success("🎉 All questions answered! Click below to view your comprehensive assessment.")
        if st.button("📊 View Complete Assessment", key="complete_assessment"):
            st.session_state.assessment_complete = True
            st.rerun()

else:
    st.header("📊 Comprehensive Assessment Results")

    responses_list = [st.session_state.responses[q['id']] for q in STUDENT_QUESTIONNAIRE]
    overall = calculate_overall_assessment(responses_list)

    st.markdown("### Overall Mental Health Status")
    col1, col2, col3, col4 = st.columns(4)

    stress_class = "high-stress-metric" if overall['quantifiable_stress_index'] > 3.5 else ""

    with col1:
        sentiment_label = "Positive" if overall['overall_sentiment'] > 0.2 else "Negative" if overall['overall_sentiment'] < -0.2 else "Neutral"
        st.metric("Overall Sentiment (ML)", sentiment_label, f"{overall['overall_sentiment']:.2f}")

    with col2:
        st.metric("Dominant Emotion (ML)", overall['dominant_emotion'][0].capitalize(), f"{overall['dominant_emotion'][1]:.2f}")

    with col3:
        st.markdown(f'<div data-testid="stMetric" class="{stress_class}"><h3>Quantifiable Stress Index</h3><p data-testid="stMetricValue">{overall["quantifiable_stress_index"]:.2f}/5.00</p></div>', unsafe_allow_html=True)

    with col4:
        st.metric("Risk Level", overall['overall_risk'], overall['risk_color'])

    if overall['categories_at_risk']:
        st.markdown("### ⚠️ Areas of Concern (Based on ML and Alerts)")
        st.warning(f"The following areas triggered alerts: **{', '.join(overall['categories_at_risk'])}**")

    st.markdown("### 📋 Detailed Response Analysis")

    for idx, q in enumerate(STUDENT_QUESTIONNAIRE):
        resp = st.session_state.responses[q['id']]

        with st.expander(f"**{q['category']}**: {q['question']}", expanded=False):
            st.markdown(f"**Your Answer:** {resp['answer']}")

            if resp['type'] != 'text':
                st.markdown(f"**Quantifiable Stress Score:** **{resp['quantifiable_score']}** (out of 5)")

            col1, col2, col3 = st.columns(3)

            analysis = resp['analysis']
            risk = resp['risk']

            col1.metric("Sentiment", analysis['ensemble_sentiment']['label'], f"{analysis['ensemble_sentiment']['score']:.2f}")
            col2.metric("Emotion", analysis['raw_top_emotion'][0].capitalize(), f"{analysis['raw_top_emotion'][1]:.2f}")
            col3.metric("Risk", risk['risk_category'], f"{risk['confidence_percent']:.1f}%")

            if analysis['alert']:
                st.error("🚨 **Alert Triggered for this response**")
                for reason in analysis['alert_reasons']:
                    st.markdown(f"- {reason}")

            if resp.get('empathetic_response'):
                st.markdown("**Supportive Response:**")
                st.info(f"💙 {resp['empathetic_response']}")

    st.markdown("---")

    st.markdown("### 🎭 Complete Emotional Profile (Smoothed Average)")
    emotion_df = pd.DataFrame(list(overall['emotion_profile'].items()), columns=['Emotion', 'Score'])
    emotion_df = emotion_df.sort_values('Score', ascending=False)
    st.bar_chart(emotion_df.set_index('Emotion'))

    st.markdown("---")

    if st.button("Start New Assessment", key="final_reset"):
        st.session_state.responses = {}
        st.session_state.current_question = 0
        st.session_state.assessment_complete = False
        st.session_state.emotion_history = deque(maxlen=HISTORY_WINDOW)
        st.rerun()