# EmoChat 💬 — Emotion-Aware Chatbot

<img width="1091" height="342" alt="Screenshot 2025-09-07 225237" src="https://github.com/user-attachments/assets/6c395007-c71f-4f4f-8852-2f4f71b7652f" />


**EmoChat** is a friendly chatbot that detects emotions from your text and responds empathetically. Powered by **DistilBERT**, it understands emotions like joy, sadness, anger, fear, love, and surprise, and replies accordingly with personalized messages and emojis.

---

## 🌟 Features

* Detects emotions from user text input.
* Responds empathetically with relevant messages.
* Color-coded responses for each emotion.
* Emoji-enhanced messages for a better chat experience.
* Fast and lightweight model using DistilBERT.
* Web interface powered by **Streamlit**.

---

## 📁 Folder Structure

```
emotion-chatbot/
├─ app.py            # Streamlit application
├─ train.py          # Training script
├─ data/             # Dataset folder
├─ models/           # Trained model directory
├─ venv/             # Virtual environment (ignored in git)
└─ README.md
```

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/EmoChat.git
cd EmoChat
```

2. **Create virtual environment (optional but recommended):**

```bash
python -m venv venv
# Activate venv:
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🏃 Run the App

```bash
streamlit run app.py
```

* Open the link in the terminal (usually `http://localhost:8501`) to interact with EmoChat.
* Type any message and see how EmoChat detects your emotion and replies empathetically.

---

## 🧠 Training a Custom Model

If you want to train EmoChat with your own data:

```bash
python train.py
```

* Place your dataset in `data/train.txt` in the format:

```
I am so happy today! ; joy
I feel really sad. ; sadness
```

* The model will be saved in `models/emotion_distilbert`.

---

## 🎨 Customization

* **EMOJI** and **COLOR** dictionaries in `app.py` can be modified for different styles.
* **TEMPLATES** can be updated to change the chatbot’s replies for each emotion.

---

## 📌 Dependencies

* Python 3.8+
* Streamlit
* PyTorch
* Transformers
* scikit-learn
* pandas, numpy

*(See full list in `requirements.txt`)*

---

## 🤝 Contributing

Feel free to **fork**, **star**, and submit **pull requests**!
We welcome improvements like:

* Adding more emotions.
* Enhancing chat interface.
* Optimizing model performance.

---

## 📜 License

This project is licensed under **MIT License**.

---

## 💬 Sample Chat

| User                                    | EmoChat Reply                                                        |
| --------------------------------------- | -------------------------------------------------------------------- |
| I feel so excited about my new project! | 😊 Joy — 98%<br>That's wonderful to hear! Tell me more 🙂.           |
| I'm sad I couldn't make it.             | 😔 Sadness — 92%<br>I'm sorry you feel that way. I'm here to listen. |
| I'm furious about how they treated me.  | 😡 Anger — 95%<br>I hear you — that sounds frustrating.              |

---

Made with ❤️ by [Sachin Kumar](https://github.com/<Sach-in2024>)
