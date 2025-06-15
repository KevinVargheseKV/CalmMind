# type: ignore
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import gradio as gr
import nltk
import random

nltk.download('punkt')

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = "chroma_db"

# Setup embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Setup LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=api_key
)

# Setup ChromaDB
vector_store = Chroma(
    collection_name="mental_health_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# Comforting actions
comforting_actions = [
    "Wrap yourself in a soft blanket and feel its warmth.",
    "Hug a pillow tightly — just to remind yourself you're held.",
    "Take three slow, deep breaths and let your shoulders relax.",
    "Hold a warm mug with both hands and focus on its heat.",
    "Gently rub your palms together and feel the warmth.",
    "Listen to a calming sound or piece of music you love.",
    "Name five things around you to bring your mind back to the present.",
    "Place your hand over your heart and breathe into that space.",
    "Stretch your arms above your head and release tension.",
    "Splash some cool water on your face or wrists.",
    "Look out the window and notice the colors you see.",
    "Light a candle and watch its flame flicker gently.",
    "Squeeze something soft like a stuffed toy or stress ball.",
    "Put on your most comfortable hoodie or socks.",
    "Whisper kind words to yourself — like 'I am safe, I am loved.'",
    "Trace small circles on your palm with your fingertip.",
    "Take a short walk, even if it’s just across the room.",
    "Imagine someone who deeply cares about you giving you a warm hug.",
    "Draw or doodle something simple, like waves or clouds.",
    "Slowly sip water and notice how it feels going down.",
    "Gently place a warm cloth or heat pack on your shoulders.",
    "Write down one thing you're proud of — however small it may seem."
]

# Issue to suggestion map
issue_suggestion_map = {
    "Performance Anxiety": "Try deep breathing exercises before your meeting...",
    "Sleep Disturbance / Insomnia": "Maintain a consistent sleep schedule...",
    "Burnout": "Break tasks into very small steps and reward yourself...",
    "Chronic Stress": "Schedule 10 minutes of quiet time daily...",
    "Somatic Symptoms of Anxiety": "Try writing about your feelings each morning...",
    "Early Signs of Depression": "Make time for small enjoyable activities...",
    "Social Isolation": "Reach out to one person you trust...",
    "Attention Difficulties / Mental Fatigue": "Use the Pomodoro technique...",
    "Emotional Eating": "Try journaling what you're feeling instead...",
    "Low Self-Esteem": "Write down three small things you did well...",
    "Social Anxiety": "Challenge negative thoughts with positive self-talk...",
    "Generalized Anxiety": "Practice daily diaphragmatic breathing...",
    "Avoidant Coping": "Make a list of avoided tasks and tackle one...",
    "Depressive Symptoms": "Create a simple morning routine with small wins...",
    "Rumination / Overthinking": "Set a 10-minute timer for worry time...",
    "Loss of Agency / Helplessness": "Focus on one small decision you can make...",
    "Mood Swings": "Track your mood daily using a journal or app...",
    "Workaholism / Perfectionism": "Remind yourself that rest is productive too...",
    "Digital Addiction / Anxiety": "Try a 1-hour digital detox daily...",
    "Imposter Syndrome": "Write a list of your recent accomplishments...",
    "Loss of Appetite": "Try to eat small, nourishing snacks throughout the day...",
    "Panic Attacks": "Practice grounding techniques like focusing on breath..."
}

# Keyword detection
keywords = {
    "meeting": "Performance Anxiety",
    "presenting": "Performance Anxiety",
    "can't sleep": "Sleep Disturbance / Insomnia",
    "waking up": "Sleep Disturbance / Insomnia",
    "tired": "Burnout",
    "no energy": "Burnout",
    "on edge": "Chronic Stress",
    "tense": "Chronic Stress",
    "headache": "Somatic Symptoms of Anxiety",
    "stomach": "Somatic Symptoms of Anxiety",
    "lost interest": "Early Signs of Depression",
    "nothing feels good": "Early Signs of Depression",
    "alone": "Social Isolation",
    "disconnected": "Social Isolation",
    "can't concentrate": "Attention Difficulties / Mental Fatigue",
    "mind wandering": "Attention Difficulties / Mental Fatigue",
    "overeating": "Emotional Eating",
    "eating too much": "Emotional Eating",
    "failure": "Low Self-Esteem",
    "not good enough": "Low Self-Esteem",
    "judging": "Social Anxiety",
    "embarrassed": "Social Anxiety",
    "bad going to happen": "Generalized Anxiety",
    "worried all the time": "Generalized Anxiety",
    "avoiding": "Avoidant Coping",
    "putting off": "Avoidant Coping",
    "pointless": "Depressive Symptoms",
    "why bother": "Depressive Symptoms",
    "overthinking": "Rumination / Overthinking",
    "going in circles": "Rumination / Overthinking",
    "no control": "Loss of Agency / Helplessness",
    "helpless": "Loss of Agency / Helplessness",
    "mood swings": "Mood Swings",
    "emotional rollercoaster": "Mood Swings",
    "guilty": "Workaholism / Perfectionism",
    "can't relax": "Workaholism / Perfectionism",
    "phone": "Digital Addiction / Anxiety",
    "can't stop scrolling": "Digital Addiction / Anxiety",
    "imposter": "Imposter Syndrome",
    "not qualified": "Imposter Syndrome",
    "appetite": "Loss of Appetite",
    "not eating": "Loss of Appetite",
    "eating less": "Loss of Appetite",
    "lost appetite": "Loss of Appetite",
    "panic": "Panic Attacks",
    "panic attack": "Panic Attacks",
    "heart racing": "Panic Attacks",
    "can't breathe": "Panic Attacks"
}

def detect_issue(user_input):
    for word, label in keywords.items():
        if word.lower() in user_input.lower():
            return label
    return None

def generate_response(user_input, history):
    thinking_msg = "💬 CalmMind is gently gathering its thoughts..."
    history.append((user_input, thinking_msg))
    yield "", history

    docs = retriever.invoke(user_input)
    knowledge = "\n\n".join([doc.page_content for doc in docs])
    comfort = random.choice(comforting_actions)
    issue = detect_issue(user_input)
    suggestion = issue_suggestion_map.get(issue, "You're doing your best — and that's enough. 💙")

    prompt = f"""
You are CalmMind — a gentle, empathetic, human-like mental health companion. Someone is opening up to you. Your role is to make them feel deeply seen, understood, and safe.

Always:
- Respond like a warm-hearted friend
- Reflect back how they might be feeling emotionally
- Offer one simple self-care action they can do right now
- Reassure them with kindness and presence
- Never diagnose

Here’s the user input:
{user_input}

Here’s background knowledge that may help:
{knowledge}

Respond with this structure:

### 🌧 I Hear You...
<Empathize deeply and reflect on their emotional state>

### 🛠 Issue & Suggestion
*Issue:* {issue or "Not clearly detected"}  
*Suggestion:* {suggestion}

### 🤗 A Moment of Comfort
{comfort}

### 💬 You Are Not Alone
<Normalize their experience with warmth>

### 💡 Gentle Next Step
<Suggest journaling, breathwork, a walk, or other soft action>

### 📘 A Loving Reminder
<End with hope and encouragement>

### 🌱 You are not broken. You are becoming.
"""
    response = llm.invoke(prompt).content
    history[-1] = (user_input, response)
    yield "", history

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("""
    <h1 style='text-align: center; font-size: 2.8em; color: #264653; font-family: Georgia, serif;'>
    🧘‍♂ <span style='color:#2A9D8F;'>CalmMind</span>  
    </h1>
    <h3 style='text-align: center; color: #457B9D; font-weight: normal;'>
    Your Gentle, Empathetic Mental Health Companion 💙
    </h3>
    <p style='text-align: center; color: #6c757d; font-style: italic; font-size: 1.05em;'>
    "Helping you breathe easier, one kind word at a time."
    </p>
    """)
    chatbot = gr.Chatbot(show_copy_button=True, bubble_full_width=False, height=450)
    msg = gr.Textbox(placeholder="Type how you're feeling and press Enter...", lines=1)
    clear = gr.Button("🧹 Clear Chat")
    state = gr.State([])

    msg.submit(generate_response, [msg, state], [msg, chatbot], queue=True)
    clear.click(lambda: ([], []), None, [chatbot, state])

# ✅ Corrected launch block
if __name__ == "__main__":
    demo.launch()
