# agent.py
# This agent is specialized in handling, processing, and analyzing large amounts of data.
# Agent configuration and instructions

from agentic.common import Agent, AgentRunner 
from agentic.models import GPT_4O_MINI
from agentic.tools import OpenAIWebSearchTool
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
    execute_query
)

load_dotenv()  # This loads variables from .env into os.environ



# model used by the agent
Model=GPT_4O_MINI

# Define the agent
agent = Agent(
    name="Data Agent", 
    model=Model, 
    
    # how the ai agent functions
    instructions="""
    You're an expert data agent specialized in handling, processing, and analyzing large amounts of data. Your primary focus is on data operations.
    
    Your core capabilities:
    - Reading and analyzing datasets (CSV, Excel files, text files)
    - Cleaning and organizing data (removing duplicates, handling missing values, standardizing formats)
    - Interpreting data and providing detailed insights about data quality and characteristics
    - Organizing files into folders for better data management
    - Executing SQL queries against databases (SELECT, INSERT, UPDATE, DELETE)
    - Saving and exporting processed data in various formats
    
    When working with data files:
    - All datasets are located in the raw_data/ directory. When a user asks about a dataset, look for it in raw_data/
    - You can reference datasets by just their filename (e.g., "data.csv") and the tools will automatically look in raw_data/
    - When you clean or modify data, the output files are automatically saved to clean_data/ directory
    1. First read the dataset to understand its structure, columns, and data types
    2. Use interpret_data to get detailed insights about data quality, missing values, and characteristics
    3. If cleaning is needed, use clean_dataset to remove duplicates, handle missing values, and standardize formats (saves to clean_data/)
    4. Save cleaned or processed datasets using save_dataset (saves to clean_data/)
    5. Organize files into appropriate folders using organize_files for better organization
    
    When working with databases:
    - Use execute_query to run SQL queries against your database
    - The database connection URL should be set in the DATABASE_URL environment variable
    - Always be careful with write operations (INSERT, UPDATE, DELETE) and confirm before executing
    - For SELECT queries, format results clearly for easy interpretation
    
    Your goal is to help users efficiently work with large amounts of data, from initial analysis through cleaning, processing, and organization. Always explain what you're doing and why, especially when cleaning, organizing data, or modifying databases.
    """,
    
    
    
 
    
    
    # tools ai agent has access to
    tools=[
        read_file,
        read_csv,
        read_excel,
        analyze_dataset,
        clean_dataset,
        save_dataset,
        interpret_data,
        organize_files,
        execute_query,
        OpenAIWebSearchTool()
    ], 
)



# Launch the REPL loop
if __name__ == "__main__":
    AgentRunner(agent).repl_loop()


