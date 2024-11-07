# Employee Database Chatbot

This project implements a chatbot that can answer questions about employee data using natural language. It uses two AI models:
- Gemini Pro 1.5 for SQL query generation
- Gemini Flash 1.5 for natural language response generation

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Create a `.env` file and add your Gemini API key.
6. Run the chatbot via Gradio Interface: `python main.py`