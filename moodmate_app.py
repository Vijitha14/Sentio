from rag_retriever import retrieve_relevant_context
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# 🪄 Streamlit Page Settings
st.set_page_config(page_title="MoodMate 🕯️", page_icon="🌷")
st.title("🌷 MoodMate – Your Soft-Spoken Support AI")
st.write("Tell me how you're feeling today, and I’ll hold space for you 💌")

# 🧠 User Input
user_input = st.text_input("💭 What's on your mind?", placeholder="I'm feeling a bit overwhelmed today...")


if user_input:
    with st.spinner("MoodMate is thinking... 🧠"):
        try:
            # 🔍 Step 1: Retrieve relevant context from file
            context = retrieve_relevant_context(user_input, file_path="data/mood_knowledge.txt")

            # 📝 Step 2: Combine context + query
            full_prompt = f"""
You are MoodMate, a calm and supportive AI companion.

Rules:
- Be empathetic and gentle.
- Never provide medical diagnoses.
- Encourage healthy coping strategies.
- Keep responses warm, conversational, and concise.

Context:
{context}

User:
{user_input}
"""

            # 🚀 Step 3: Prepare request
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }

            data = {
                "inputs": {
                    "past_user_inputs": [],
                    "generated_responses": [],
                    "text": full_prompt
                }
            }

         # 🤖 Generate response using Gemini
           response = model.generate_content(full_prompt)

           moodmate_reply = response.text

           st.markdown(f"🕯️ **MoodMate says:**\n\n{moodmate_reply}")
