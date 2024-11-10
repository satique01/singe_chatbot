import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set up the app interface
st.title("BlenderBot Chatbot")
st.write("This is a chatbot using the BlenderBot model. Type a message and press Enter.")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Get user input
user_input = st.text_input("You:", key="input")

# Generate and display bot response
if user_input:
    # Tokenize and generate the response
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=100)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    # Add user and bot responses to chat history
    st.session_state.chat_history.append((user_input, bot_response))
    
    # Clear the input field
    st.session_state.input = ""

# Display the chat history
for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
    st.write(f"You: {user_msg}")
    st.write(f"Bot: {bot_msg}")
