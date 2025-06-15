# 🧘‍♂️ CalmMind — Your Gentle, Empathetic Mental Health Companion

![CalmMind Banner](https://imgur.com/your-banner-if-any.png)

> *“Helping you breathe easier, one kind word at a time.”*

CalmMind is a warm, AI-powered mental health chatbot built with **LangChain**, **Gemini API**, and **ChromaDB**, offering emotional support, self-care tips, and comforting dialogue to users in moments of need.

---

## 📽 Demo

<p align="center">
  <img src="https://media.giphy.com/media/3o7bu3XilJ5BOiSGic/giphy.gif" width="600" alt="Chatbot typing animation demo GIF">
</p>


## 🌟 Key Features

- 💬 Empathetic AI conversation that reflects user feelings
- 📚 Retrieval-Augmented Generation (RAG) for context-aware suggestions
- 🤗 Comfort actions like breathing cues, journaling prompts, and grounding techniques
- ⚙️ Keyword-based detection of emotional challenges (e.g., anxiety, burnout)
- 🎨 Clean and friendly **Gradio** interface for supportive chat

---

## 🛠 Tech Stack

| Tool                   | Purpose                                  |
|------------------------|------------------------------------------|
| LangChain              | Orchestrates LLM calls and retrieval     |
| Gemini API             | Powers natural, empathic conversation    |
| ChromaDB               | Stores mental health knowledge vectors   |
| Google GenAI Embeddings | Enables semantic search in documents   |
| NLTK                   | Helps with tokenization and keyword mapping |
| Gradio                 | Frontend interface for the chatbot       |
| dotenv                 | Manages access to your API key securely  |

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/CalmMind.git
cd CalmMind
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
