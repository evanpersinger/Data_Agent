# agent.py
# This agent is specialized in handling, processing, and analyzing large amounts of data.
# Agent configuration and instructions

import os
import json
from openai import OpenAI
from dotenv import load_dotenv


# import tools from agent_tools.py
from agent_tools import (
    read_file,
    read_csv,
    read_excel,
    analyze_dataset,
    clean_dataset,
    save_dataset,
    interpret_data,
    organize_files,
    execute_query,
    download_kaggle_dataset,
    get_function_schema
)


load_dotenv()  # This loads variables from .env into os.environ

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model used by the agent
MODEL = "gpt-4o-mini"

# System instructions for the agent
SYSTEM_INSTRUCTIONS = """
You're an expert data agent specialized in handling, processing, and analyzing large amounts of data. Your primary focus is on data operations.

Your core capabilities:
- Downloading datasets from Kaggle directly to raw_data/ directory
- Reading and analyzing datasets (CSV, Excel files, text files)
- Cleaning and organizing data (removing duplicates, handling missing values, standardizing formats)
- Interpreting data and providing detailed insights about data quality and characteristics
- Organizing files into folders for better data management
- Executing SQL queries against databases (SELECT, INSERT, UPDATE, DELETE)
- Saving and exporting processed data in various formats

When working with data files:
- You can download datasets from Kaggle using download_kaggle_dataset. The dataset identifier should be in format "username/dataset-name" (e.g., "c/titanic" or "username/titanic"). Downloaded files are automatically saved to kaggle_data/ directory
- All datasets are located in the raw_data/ directory. When a user asks about a dataset, look for it in raw_data/
- Kaggle datasets are saved in kaggle_data/ directory. To work with them, you may need to reference the full path (e.g., "kaggle_data/data.csv") or move them to raw_data/
- You can reference datasets by just their filename (e.g., "data.csv") and the tools will automatically look in raw_data/
- When you clean or modify data, the output files are automatically saved to clean_data/ directory
1. If a user wants a dataset from Kaggle, use download_kaggle_dataset first to fetch it (saves to kaggle_data/)
2. First read the dataset to understand its structure, columns, and data types
3. Use interpret_data to get detailed insights about data quality, missing values, and characteristics
4. If cleaning is needed, use clean_dataset to remove duplicates, handle missing values, and standardize formats (saves to clean_data/)
5. Save cleaned or processed datasets using save_dataset (saves to clean_data/)
6. Organize files into appropriate folders using organize_files for better organization

When working with databases:
- Use execute_query to run SQL queries against your database
- The database connection URL should be set in the DATABASE_URL environment variable
- Always be careful with write operations (INSERT, UPDATE, DELETE) and confirm before executing
- For SELECT queries, format results clearly for easy interpretation

Your goal is to help users efficiently work with large amounts of data, from initial analysis through cleaning, processing, and organization. Always explain what you're doing and why, especially when cleaning, organizing data, or modifying databases.
"""


# Create function schemas for all tools
TOOLS = [
    read_file,
    read_csv,
    read_excel,
    analyze_dataset,
    clean_dataset,
    save_dataset,
    interpret_data,
    organize_files,
    execute_query,
    download_kaggle_dataset,
]

# Convert tools to OpenAI function calling format
FUNCTIONS = {tool.__name__: tool for tool in TOOLS}
FUNCTION_SCHEMAS = [get_function_schema(tool) for tool in TOOLS]


def run_agent(user_input, messages):
    """Run the agent with user input and return the response."""
    # Add user message
    messages.append({"role": "user", "content": user_input})
    
    # Handle multiple rounds of tool calls (max 10 to prevent infinite loops)
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=FUNCTION_SCHEMAS,
            tool_choice="auto"
        )
        
        # Handle the response
        message = response.choices[0].message
        messages.append(message)
        
        # If the model wants to call a function
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the function
                try:
                    function_to_call = FUNCTIONS[function_name]
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = f"Error calling {function_name}: {str(e)}"
                
                # Add function response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(function_response)
                })
        else:
            # No more tool calls, return the final response
            break
    
    return message.content, messages


def repl_loop():
    """REPL loop for interacting with the agent."""
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    
    print("Data Agent - Ready to help with your data tasks!")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            response, messages = run_agent(user_input, messages)
            print(f"\nAgent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


# Launch the REPL loop
if __name__ == "__main__":
    repl_loop()


