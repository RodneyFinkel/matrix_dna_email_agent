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

Heuristic Labeling
Spam Detection:
Applied keyword-based heuristics to label emails as spam (e.g., 'offer', 'win', 'urgent') and filtered to ~5% to avoid skewing training.

Category & Priority Labels:
Where explicit labels were not available, category and priority were inferred using logical keyword matching and subject/body context.

 Note: As this is a weakly supervised setup, final evaluation metrics may reflect some noise in the labels. Manual labeling or active learning could improve this in future iterations.

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
