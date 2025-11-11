# agent.py
# This agent is specialized in handling, processing, and analyzing large amounts of data.
# Using OpenAI's Assistants API (agentic framework)

import os
import json
import time
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, function_tool, ModelSettings
from agents.stream_events import RawResponsesStreamEvent
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import asyncio


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



# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Convert custom functions to tools using function_tool
custom_tools = [
    function_tool(read_file),
    function_tool(read_csv),
    function_tool(read_excel),
    function_tool(analyze_dataset),
    function_tool(clean_dataset),
    function_tool(save_dataset),
    function_tool(interpret_data),
    function_tool(organize_files),
    function_tool(execute_query),
    function_tool(download_kaggle_dataset),
]

agent = Agent(
    
    # Name of the agent
    name = "Data Agent",
    
   # Model used by the agent
   model = "gpt-4o-mini",
   
   # Tools used by the agent
   tools=[
       WebSearchTool(),
   ] + custom_tools,


    # System instructions for the agent
    instructions =
    """
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

)


# main function
# this is the main function that will be called to run the agent
async def main():
    # Sessions store conversation history so the agent remembers previous messages
    session = SQLiteSession(session_id="example_session")
    
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Interactive loop to ask questions
    while True:
        try:
            # Get user input
            user_input = input(f"{agent.name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended")
            break
        
        # Check if user wants to quit
        # if the user inputs 'quit', 'exit', or 'q', the session will end
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Session ended")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Run the agent with streaming (shows response as it's generated)
        print("\nAgent: ", end="", flush=True)
        result = Runner.run_streamed(agent, user_input, session=session)
        
        # Stream the response in real-time
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
        
        print("\n")  # Add newline after response




# run the agent
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended")
    
