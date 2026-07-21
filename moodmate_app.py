from rag_retriever import retrieve_relevant_context
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="MoodMate 🌷",
    page_icon="🌷",
    layout="centered"
)

st.title("🌷 MoodMate – Your Soft-Spoken Support AI")
st.write("Tell me how you're feeling today, and I'll hold space for you 💌")

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_input(
    "💭 What's on your mind?",
    placeholder="I'm feeling a bit overwhelmed today..."
)

# ----------------------------
# AI Response
# ----------------------------
if user_input:

    with st.spinner("MoodMate is thinking... 🧠"):

        try:

            # Retrieve relevant context using RAG
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

            # Generate response using Gemini
            response = model.generate_content(full_prompt)

            st.markdown("### 🕯️ MoodMate says")
            st.success(response.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
