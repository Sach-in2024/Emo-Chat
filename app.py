# app.py
import streamlit as st
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import os
import torch.nn.functional as F

MODEL_DIR = "models/emotion_distilbert"

EMOJI = {
    "joy":"😊",
    "sadness":"😔",
    "anger":"😡",
    "fear":"😨",
    "love":"❤️",
    "surprise":"😲"
}
COLOR = {
    "joy":"#FEEFB3",
    "sadness":"#DCE9FF",
    "anger":"#FFD6D6",
    "fear":"#F3E8FF",
    "love":"#FFDDEE",
    "surprise":"#E6FFE6"
}
TEMPLATES = {
    "joy":[ "That's wonderful to hear! Tell me more 🙂.",
            "Love that energy — keep it going!"],
    "sadness":[ "I'm sorry you feel that way. I'm here to listen.",
                "That sounds really hard — do you want to talk about it?"],
    "anger":[ "I hear you — that sounds frustrating.",
              "Anger is valid. What happened?"],
    "fear":[ "That sounds scary. You're not alone in this.",
             "It's okay to be afraid — would you like some calming tips?"],
    "love":[ "Aw, that's sweet. Tell me more about them!",
             "Love is beautiful—what makes this special?"],
    "surprise":[ "Wow — that's unexpected!",
                 "That's surprising — how did you feel about it?"]
}

st.set_page_config(page_title="EmoChat — Emotion-aware chatbot", layout="wide", page_icon="🤖")

# CSS
st.markdown("""
<style>
body {background: linear-gradient(120deg, #f6f9ff 0%, #fffaf6 100%);}
.chat-card {padding: 12px; border-radius: 12px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);}
.user {background: #f1f3f5; text-align: right;}
.bot {background: white; text-align: left;}
.title {font-size:28px; font-weight:700; margin-bottom:4px;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("<div class='title'>EmoChat 💬 — A friendly emotion-aware chatbot</div>", unsafe_allow_html=True)
    st.markdown("Type something and I'll detect the emotion and reply with empathy.")

    # Initialize session state variables
    if "history" not in st.session_state:
        st.session_state.history = []

    if "input" not in st.session_state:
        st.session_state.input = ""

    user_input = st.text_area("You", height=120, key="input")

    def send_message():
        if st.session_state.input.strip():
            st.session_state.history.append(("user", st.session_state.input.strip()))

            if not os.path.isdir(MODEL_DIR):
                st.session_state.history.append(("bot", "⚠️ Model not found. Train it first."))
            else:
                if "model" not in st.session_state:
                    st.session_state.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                    st.session_state.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                    with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
                        lm = json.load(f)
                        st.session_state.id2label = lm["id2label"]

                tok = st.session_state.tokenizer(st.session_state.input, truncation=True, padding=True, return_tensors="pt", max_length=128)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st.session_state.model.to(device)
                tok = {k:v.to(device) for k,v in tok.items()}

                with torch.no_grad():
                    out = st.session_state.model(**tok)
                    logits = out.logits
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    pred_id = int(np.argmax(probs))
                    pred_label = st.session_state.id2label[str(pred_id)]
                    reply = random.choice(TEMPLATES.get(pred_label, ["I see. Tell me more."]))
                    meta = {"label": pred_label, "prob": float(probs[pred_id])}
                    st.session_state.history.append(("bot", reply, meta))

            # Clear the input safely
            st.session_state.input = ""

    st.button("Send", on_click=send_message)

    # Show chat history
    for item in st.session_state.history[::-1]:
        role = item[0]
        text = item[1]
        if role == "user":
            st.markdown(f"<div class='chat-card user'>{text}</div>", unsafe_allow_html=True)
        else:
            meta = item[2] if len(item) > 2 else {}
            label = meta.get("label", "")
            prob = meta.get("prob", 0.0)
            color = COLOR.get(label, "#ffffff")
            emoji = EMOJI.get(label, "")
            st.markdown(f"""
            <div class='chat-card bot' style='background:{color}'>
                <div style='font-weight:600'>{emoji} {label.title()} — {prob*100:.1f}%</div>
                <div style='margin-top:6px'>{text}</div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("### Model Info")
    st.markdown(f"**Model dir:** `{MODEL_DIR}`")
    st.markdown("### Quick tips")
    st.markdown("- Try both short and long sentences.")
    st.markdown("- The model predicts emotion + replies empathetically.")
    st.markdown("### Sample prompts")
    st.markdown("- I feel so excited about my new project!\n- I'm sad I couldn't make it.\n- I'm furious about how they treated me.")
