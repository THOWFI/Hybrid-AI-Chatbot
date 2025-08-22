# 🤖 Arshu – Hybrid AI Chatbot  

An advanced **offline-first AI assistant** powered by **local LLMs (Mistral-7B & Gemma-2B)** with smart model routing.  
This project runs entirely on **Streamlit** and provides a **ChatGPT-like interface** with support for:  

- Natural multi-turn conversations  
- Auto model routing (casual/basic → Gemma, technical → Mistral)  
- Wikipedia keyword scraping for smarter context  
- Streaming (token-by-token) responses like ChatGPT  
- Exportable chat history (`.json`)  
- Extendable with Translation, Summarization, and Image Generation  

---

## 📌 Project Overview  

Arshu (the AI assistant) is designed as a **modular hybrid chatbot**:  

- **Gemma-2B** → For casual/basic Q&A (lightweight, fast).  
- **Mistral-7B** → For deep technical reasoning.  
- **Router logic** → Automatically chooses best model (or user forces via sidebar).  

The system includes:  
- Streaming and full-reply modes.  
- Conversation memory (configurable turns).  
- Wikipedia scraping + keyword routing.  
- Export/import chat history.  

---

## 🚀 Features  

- ChatGPT-like UI with chat bubbles  
- Streaming replies (token preview with cursor effect)  
- Automatic Model Routing (based on keywords)  
- Sidebar controls: reload keywords, clear/export chat  
- Multi-turn memory → context-aware replies  
- Code rendering (auto detects ```fenced code``` and formats in IDE-style)  
- Lightweight → runs on CPU (no GPU required)  

---

## 🛠️ Tech Stack  

- Python 3.11+  
- Streamlit – Web UI framework  
- llama.cpp / llama-cpp-python – Local LLM inference  
- NLTK / SpaCy – Text cleaning & NLP enhancements  
- Wikipedia API – Keyword scraping  
- JSON – Chat history persistence  

---

## 📂 Project Structure  

```bash
AI-CHATBOT/
│── app.py                 # Streamlit app entry point
│── requirements.txt       # Python dependencies
│── tech_keywords.txt      # Extracted Wikipedia keywords
│── .gitignore
│── .gitattributes
│
├── models/                # Model storage
│   └── Download Link.txt  # Links to download Gemma/Mistral GGUF models
│
├── scripts/               # Core logic modules
│   ├── llm_utils.py       # Query + stream LLM responses
│   ├── router.py          # Routes query → Gemma or Mistral
│   ├── nlp_enhancer.py    # Cleans & enhances text
│   ├── scrape_keywords.py # Scrapes Wikipedia topics
│   └── __init__.py
│
└── .venv/                 # Virtual environment (local, ignored by git)
```

---

## 🔽 Model Downloads

The `models/Download Link.txt` file contains direct download links for:
      -Gemma-2B (GGUF quantized)
      -Mistral-7B (GGUF quantized)
Steps:
      1. Open `models/Download Link.txt`
      2. Download the models
      3. Place them inside the `models`

---

##⚙️ Installation & Setup

##1️⃣ Clone Repository
```bash
git clone https://github.com/THOWFI/AI-CHATBOT.git
cd AI-CHATBOT
```

## 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac
```

## 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 4️⃣ Download Models
```bash
# Open the text file for download links
notepad models/Download\ Link.txt   # Windows
cat models/Download\ Link.txt       # Linux/Mac
# Download Gemma-2B & Mistral-7B GGUF models and place in models/
```

## 5️⃣ Run the Chatbot
```bash
streamlit run app.py
```

---

## 📊 Example

```
🧑: What is AI?
🤖 (Gemma): AI stands for Artificial Intelligence, the field of building systems that can think and learn like humans.

🧑: Explain Transformer models in deep learning
🤖 (Mistral): Transformers are neural architectures introduced in "Attention is All You Need", designed for handling sequential data efficiently with self-attention.
```

---

## 🔮 Future Enhancements

- Multi-tabbed chat history (like ChatGPT)
- Tamil + English voice input/output (TTS & STT)
- Image generation support (Stable Diffusion)
- Android app integration (offline-first)
- Optimized model routing with embeddings

---

## 📜 License

This project is **for educational and personal use only**.                 
Models belong to their respective owners (Gemma/Mistral).                      





