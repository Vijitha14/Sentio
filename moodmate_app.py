from groq import groq
import streamlit as st
import os
from dotenv import load_dotenv
from rag_retriever import retrieve_relevant_context

# ---------------------------------
# Load Environment Variables
# ---------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------
# Streamlit Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="MoodMate 🌷",
    page_icon="🌷",
    layout="centered"
)

st.title("🌷 MoodMate – Your Soft-Spoken Support AI")
st.write("Tell me how you're feeling today, and I'll hold space for you 💌")

# ---------------------------------
# User Input
# ---------------------------------
user_input = st.text_input(
    "💭 What's on your mind?",
    placeholder="I'm feeling a bit overwhelmed today..."
)

# ---------------------------------
# AI Response
# ---------------------------------
if user_input:

    with st.spinner("MoodMate is thinking... 🧠"):

        try:

            # Retrieve relevant context (RAG)
            context = retrieve_relevant_context(
                user_input,
                file_path="data/mood_knowledge.txt"
            )

            # Build prompt
            full_prompt = f"""
You are MoodMate, a calm, gentle, emotionally supportive AI companion.

Rules:
- Speak warmly and kindly.
- Never diagnose medical conditions.
- Never prescribe medicines.
- Validate the user's feelings.
- Encourage healthy coping strategies.
- Ask gentle follow-up questions when appropriate.
- Keep responses concise (100–150 words).

Relevant Context:
{context}

User:
{user_input}
"""

           response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are MoodMate, a calm, gentle, emotionally supportive AI companion. Never diagnose diseases or prescribe medication. Be warm, empathetic, and concise."
        },
        {
            "role": "user",
            "content": full_prompt
        }
    ],
    temperature=0.7,
    max_tokens=300
)

reply = response.choices[0].message.content

st.markdown("### 🕯️ MoodMate says")
st.success(reply)
