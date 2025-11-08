# Data Agent

An AI agent specialized in handling, processing, and analyzing large amounts of data. It can read datasets, clean messy data, execute database queries, interpret data insights, and organize your files.

## Features

- **Read Data**: Read CSV, Excel, and text files
- **Analyze Data**: Get statistics, missing values, and data structure insights
- **Clean Data**: Remove duplicates, handle missing values, standardize formats
- **Interpret Data**: Get detailed insights about your datasets and data quality
- **Database Operations**: Execute SQL queries against databases (SQLite, PostgreSQL, MySQL, etc.)
- **Organize Files**: Move files into organized folders for better data management
- **Save & Export**: Save processed data in CSV or Excel formats

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key and optional database URL:
```
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///database.db  # Optional: for database operations
```

3. Run the agent:
```bash
python agent.py
```

## Usage

Once the agent is running, you can interact with it in the REPL. Example commands:

- "Read the file data.csv"
- "Analyze the dataset sales.xlsx"
- "Clean the dataset messy_data.csv"
- "Interpret the data in customers.csv"
- "Run this SQL query: SELECT * FROM users WHERE age > 25"
- "Organize the file report.pdf into the reports folder"
- "Save the cleaned data to output.csv"

## Project Structure

```
research_agent/
├── agent.py          # Data agent configuration and instructions
├── agent_tools.py    # Data processing functions/tools the agent can use
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## How It Works

- **agent.py**: Contains the agent's configuration, instructions, and launches the REPL loop
- **agent_tools.py**: Contains all the functions the agent can use (reading, cleaning, analyzing data, etc.)

The agent uses GPT-4O Mini to understand your requests and automatically calls the appropriate tools to help you work with your data.

