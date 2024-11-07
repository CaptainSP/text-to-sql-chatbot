import os
import sqlite3
import time
import google.generativeai as genai
from dotenv import load_dotenv
import gradio as gr
import json
import coloredlogs, logging
from jsonschema import validate


# Create a logger object.
logger = logging.getLogger(__name__)


# If you don't want to see log messages from libraries, you can pass a
# specific logger object to the install() function. In this case only log
# messages originating from that logger will show up on the terminal.
coloredlogs.install(level='DEBUG', logger=logger)

# Load environment variables from a .env file
load_dotenv()

# Fetch API key for GEMINI from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the GemAI API with the fetched API key
genai.configure(api_key=GEMINI_API_KEY)

# Connect to the SQLite database and create a cursor
con = sqlite3.connect("database.db", check_same_thread=False)
cur = con.cursor()

# Database schema prompt
db_schema_prompt = '''
You are provided with a database schema that contains multiple tables, each with specific columns and properties. Here are the details of the tables:

Table: departments
Columns:
dept_no: type char(4), primary key
dept_name: type varchar(40)

Table: dept_emp
Columns:
emp_no: type INTEGER, primary key
dept_no: type char(4), primary key, foreign key referencing departments(dept_no)
from_date: type date
to_date: type date

Table: dept_manager
Columns:
dept_no: type char(4), primary key, foreign key referencing departments(dept_no)
emp_no: type INTEGER, primary key, foreign key referencing employees(emp_no)
from_date: type date
to_date: type date

Table: employees
Columns:
emp_no: type INTEGER, primary key
birth_date: type date
first_name: type varchar(14)
last_name: type varchar(16)
gender: type TEXT
hire_date: type date

Table: salaries
Columns:
emp_no: type INTEGER, primary key, foreign key referencing employees(emp_no)
salary: type INTEGER
from_date: type date, primary key
to_date: type date

Table: titles
Columns:
emp_no: type INTEGER, primary key, foreign key referencing employees(emp_no)
title: type varchar(50)
from_date: date, primary key
to_date: date, nullable

Generate only a SQL query based on the question. Return the response in this exact format:
{
    "sqlQuery": "YOUR_SQL_QUERY_HERE",
    "description": "BRIEF_DESCRIPTION_OF_QUERY"
}
'''

response_prompt = '''
You are a chatbot assistant designed to transform Some data to user friendly format. Follow these steps

1. Read the provided data.
2. Extract the necessary data from these results.
3. Combine and structure this data into a human-readable format.
4. Output the final message in JSON format: 
5. Make sure the message is clear and informative for a general user.

Example structure: 
```json
{
  "message": "string"
}
```

_Expected JSON Output:_  
```json
{
  "message": "We have 3 users: John (30 years old) from New York, Alice (25 years old) from Los Angeles, and Bob (22 years old) from Chicago."
}
```

Remember, your goal is to make the message as clear and informative as possible for a general user. Simplify complex data and highlight the most important parts.

# Do not add a text like 'the query returned'. Just explain the data in a user-friendly way.
---

End of system prompt.
'''

sql_schema = {
    "type": "object",
    "properties": {
        "sqlQuery": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["sqlQuery"],
}

response_schema = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
    },
    "required": ["message"],
}

# Initialize both models with different configurations
sql_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=db_schema_prompt,
    generation_config={
        "temperature": 0.3,
    }
)

response_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction=response_prompt,
    generation_config={
        "temperature": 1,
    }
)

# Start chat sessions for both models
sql_chat = sql_model.start_chat(history=[])
response_chat = response_model.start_chat(history=[])

def extract_json_from_response(text):
    """Extract JSON from model response, handling different formats."""
    try:
        # Try direct JSON parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try to extract JSON from markdown code blocks
            if '```json' in text:
                json_text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                json_text = text.split('```')[1].split('```')[0]
            else:
                json_text = text
            # Clean up the text
            json_text = json_text.strip()
            return json.loads(json_text)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw text: {text}")
            raise

def ask(question,history):
    """
    Process the question using two different models:
    1. SQL Model (Gemini Pro 1.5) generates the SQL query
    2. Response Model (Gemini Flash 1.5) generates the user-friendly response
    """
    tries = 4
    
    start_time_sql = time.time()
    start_time = time.time()

    for attempt in range(tries):
        try:
            # Step 1: Generate SQL query using Gemini Pro 1.5
            logger.debug(f"\nSQL MODEL: processing question: {question}")
            
            sql_result = sql_chat.send_message(
                f"Generate a SQL query for this question: {question}\nRespond only with a JSON object in the specified format."
            )
            logger.debug(f"\nSQL Model Response:\n{sql_result.text}")
            
            sql_json = extract_json_from_response(sql_result.text)
            validate(instance=sql_json, schema=sql_schema)
            query = sql_json["sqlQuery"]
            logger.debug(f"\nExtracted SQL Query: {query}")
            
            end_time_sql = time.time()
            logger.info(f"\nTotal time taken for SQL model: {end_time_sql - start_time_sql:.2f} seconds")

            # Execute the generated SQL query
            cur.execute(query)
            data = cur.fetchall()
            dataText = json.dumps(data)
            logger.debug(f"\nQuery Results: {dataText}")
            
            start_time_response = time.time()
            
            # Step 2: Generate response using Gemini Flash 1.5
            response_result = response_chat.send_message(
                f"The question of the user was: {question}.\n\n ----- \n\n The result of the query is: {dataText}\n\n ---- \n\n Now give user a pretty message in the specified JSON format."
            )
            logger.info(f"\nResponse Model Output:\n{response_result.text}")
            
            response_json = extract_json_from_response(response_result.text)
            validate(instance=response_json, schema=response_schema)
            
            end_time = time.time()
            end_time_response = time.time()
            logger.info(f"\nTotal time taken for Response model: {end_time_response - start_time_response:.2f} seconds")
            logger.info(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
            
            return response_json["message"] #, data

        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {str(e)}")
            if attempt < tries - 1:
                print("Retrying...")
                # put timeout multiplied by the attempt number
                time.sleep(2 * (attempt + 1))
            else:
                print("All attempts failed.")
                return f"Sorry, I encountered an error: {str(e)}", None

# Create a Gradio interface
# demo = gr.Interface(
#     fn=ask,
#     inputs=["text"],
#     outputs=["text", "dataframe"],
#     title="Two-Model Employee Database Chatbot",
#     description="Ask questions about employee data using natural language. Powered by Gemini Pro 1.5 (SQL) and Gemini Flash 1.5 (Response).",
#     examples=[
#         "How many employees are in the company",
#         "Who is the highest paid employee and what is his position?",
#         "En yaşlı kişi kim?",
#         "Şirkette kaç çalışan var?",
#         "Do you know a person joined the company before 1998. Can you give me a name.",
#         "Who is the oldest person in the company?",
#     ]
# )

demo = gr.ChatInterface(fn=ask, type="messages", title="SqlBot 2.0",examples=[
        "How many employees are in the company",
        "Who is the highest paid employee and what is his position?",
        "En yaşlı kişi kim?",
        "Şirkette kaç çalışan var?",
        "Do you know a person joined the company before 1998. Can you give me a name.",
        "Who is the oldest person in the company?",
    ])


# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()