from groq import Groq
import streamlit as st
import os
from dotenv import load_dotenv
from rag_retriever import retrieve_relevant_context


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


st.set_page_config(
    page_title="MoodMate 🌷",
    page_icon="🌷",
    layout="centered"
)

st.title("🌷 MoodMate – Your Soft-Spoken Support AI")
st.write("Tell me how you're feeling today, and I'll hold space for you 💌")


user_input = st.text_input(
    "💭 What's on your mind?",
    placeholder="I'm feeling a bit overwhelmed today..."
)


if user_input:

    with st.spinner("MoodMate is thinking... 🧠"):

        try:

            
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

            # Generate response using Groq
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are MoodMate, a calm, gentle, emotionally supportive AI companion. "
                            "Never diagnose diseases or prescribe medication. "
                            "Be warm, empathetic, and concise."
                        )
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )

            # Display response
            reply = response.choices[0].message.content

            st.markdown("### 🕯️ MoodMate says")
            st.success(reply)

        except Exception as e:
            st.error(f"An error occurred: {e}")
