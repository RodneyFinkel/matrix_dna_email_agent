A GenAI-powered email classification system that uses fine-tuned LLMs and a Streamlit chatbot interface to classify Microsoft Outlook emails by category (Finance, HR, Legal, Admin) and priority (High, Medium, Low).

Features

Fine-tuned sentence-transformers/all-MiniLM-L12-v2 using LoRA adapters.

Parameter Efficient Fine Tuning with LoRA

Dual-headed classification (category & priority)

Lightweight Streamlit UI with:

Support for .eml and .msg uploads and parsing

Manual text input

Chat interaction using Groq or OpenAI for email summaries and alt classifiers



OOP architecture will be added at a later time


Model Training Summary

Dataset: Sampled 12,000 emails from Enron corpus

Spam filtered heuristically, capped at 5%

Balanced by class using upsampling/downsampling

Fine-tuned with LoRA to reduce memory footprint

Evaluated with accuracy, F1, and confusion matrix


Quick Start

1. Clone repo

git clone https://github.com/your-username/enron-email-classifier.git
cd enron-email-classifier

2. Install dependencies

pip install -r requirements.txt

3. Run the app

streamlit run app.py
