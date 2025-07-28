import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq 
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from email_parser import parse_eml, parse_msg
from email_classifier import load_models, predict_category, predict_priority


load_dotenv()
#groq_api_key = os.getenv("GROQ_API_KEY")

# Load fine-tuned models at startup
model_cat, tokenizer_cat, model_prio, tokenizer_prio = load_models()

custom_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant trained by Rodney Finkel (a surprisingly nice guy) to classify Microsoft Outlook emails.
Given an input message, your job is to:
1. Predict the *category* of the email (Finance, HR, Legal, Admin).
2. Predict the *priority* (High, Medium, Low).
3. If the conversation is not about emails, politely inform the user that you can only calssify emails
Do not hallucinate. If the message seems incomplete, say so.
Do not show your reasoning or thought process. Only output the final answer in the following markdown format:

**Category:** <category>
**Priority:** <priority>

Email content:
{input}
"""
)

st.set_page_config(
    page_title="Email Classifier using a fine tuned sentence-transformers/all-MiniLM-L12-v & Chatbot")

st.title("Email Classifier using a fine tuned sentence-transformers/all-MiniLM-L12-v2 & Chatbot")
#st.write("Loaded API Key:", st.secrets.get("GROQ_API_KEY", "Not Found"))

# Model selector
llm_choice = st.sidebar.selectbox("Choose LLM:", ["Groq (Qwen)", "OpenAI (GPT-3.5)"])
st.info(f"Chatbot Running on: {llm_choice}")


if "conversation" not in st.session_state:
    prompt = custom_prompt
    # llm = ChatGroq(
    #     model="qwen/qwen3-32b",
    #     api_key=st.secrets["GROQ_API_KEY"],
    #     temperature=0.0
    # )
    if llm_choice == "Groq (Qwen)":
        llm = ChatGroq(
            model="qwen/qwen3-32b",  
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.0
        )
    else:
        llm = OpenAI(
            temperature=0.0,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
    # Create the runnable chain 
    st.session_state.conversation = prompt | llm

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# File uploader for .eml and .msg files
uploaded_file = st.file_uploader("Upload an Outlook email (.eml or .msg)", type=["eml", "msg"])

# Process uploaded file
email_input = ""
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".eml"):
            email_input = parse_eml(uploaded_file.read())
        elif uploaded_file.name.endswith(".msg"):
            email_input = parse_msg(uploaded_file)
        if email_input.startswith("Error"):
            st.error(email_input)
            email_input = ""
        else:
            st.success("Email file uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Text area for manual email input
email_input = st.text_area("Paste full email here", value=email_input, height=300)

# Classify button
if st.button("Classify Email"):
    if email_input.strip():
        try:
            # Call predict functions from email_classifier
            with st.spinner("Classifying email..."):
                category = predict_category(email_input, model_cat, tokenizer_cat)
                priority = predict_priority(email_input, model_prio, tokenizer_prio)
            st.success(f"**Category:** {category}\n\n**Priority:** {priority}")
            # Store classification in chat history
            st.session_state.messages.append({"role": "assistant", "content": f"Classified email:\nCategory: {category}\nPriority: {priority}"})
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
    else:
        st.warning("Please provide email content to classify.")


if input := st.chat_input("Speak to me"):
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.markdown(input)

    with st.chat_message("assistant"):
        stream = st.session_state.conversation.stream(
            {
                "input": input,
                "history": st.session_state.memory.load_memory_variables({})["history"],
            }
        )
        response = st.write_stream(stream)
    st.session_state.memory.save_context({"input": input}, {"output": response})
    st.session_state.messages.append({"role": "assistant", "content": response})