import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

def get_gpt_response(prompt):
    """Fetch response from OpenAI's GPT model."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("GPT Prompt Application")
user_prompt = st.text_input("Enter your prompt:")
if st.button("Submit"):
    if user_prompt:
        with st.spinner("Fetching response..."):  # Loading indicator
            response = get_gpt_response(user_prompt)
        st.write("Response:", response)
    else:
        st.error("Please enter a prompt.")

# Note: Remember to replace 'YOUR_API_KEY' with your actual OpenAI API key.