# Data Agent

An AI agent specialized in handling, processing, and analyzing large amounts of data. It can read datasets, clean messy data, execute database queries, interpret data insights, and organize your files.

## Features

- **Download Data**: Download datasets directly from Kaggle
- **Read Data**: Read CSV, Excel, and text files
- **Analyze Data**: Get statistics, missing values, and data structure insights
- **Clean Data**: Remove duplicates, handle missing values, standardize formats
- **Interpret Data**: Get detailed insights about your datasets and data quality
- **Database Operations**: Execute SQL queries against databases (SQLite, PostgreSQL, MySQL, etc.)
- **Organize Files**: Move files into organized folders for better data management
- **Save & Export**: Save processed data in CSV or Excel formats

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# Or if using uv:
uv pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///database.db  # Optional: for database operations
KAGGLE_USERNAME=your_kaggle_username  # Optional: for Kaggle dataset downloads
KAGGLE_KEY=your_kaggle_api_key  # Optional: for Kaggle dataset downloads
```

   **Getting your API keys:**
   
   **OpenAI API Key:**
   1. Go to https://platform.openai.com/api-keys
   2. Sign in or create an account
   3. Click "Create new secret key"
   4. Copy the key (you won't be able to see it again!)
   5. Add it to your `.env` file as `OPENAI_API_KEY=sk-...`
   
   **Kaggle API Key:**
   1. Go to https://www.kaggle.com/ and sign in (create account if needed)
   2. Go to https://www.kaggle.com/settings
   3. Scroll to "API" section and click "Create New Token"
   4. This downloads a `kaggle.json` file to your computer
   5. Open the `kaggle.json` file - it contains:
      ```json
      {"username":"your_username","key":"your_api_key_here"}
      ```
   6. Copy the `username` and `key` values
   7. Add them to your `.env` file:
      ```
      KAGGLE_USERNAME=your_username
      KAGGLE_KEY=your_api_key_here
      ```
   
   **Alternative Kaggle setup (using kaggle.json file):**
   - Instead of adding to `.env`, you can place `kaggle.json` in `~/.kaggle/` directory
   - Create the directory: `mkdir -p ~/.kaggle`
   - Move the file: `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
   
   **Important:** Never commit your `.env` file or `kaggle.json` to git! They contain sensitive API keys.

4. Run the agent:
```bash
python agent.py
```

## Usage

Once the agent is running, you can interact with it in the REPL. Example commands:

- "Download the titanic dataset from Kaggle"
- "Read the file data.csv"
- "Analyze the dataset sales.xlsx"
- "Clean the dataset messy_data.csv"
- "Interpret the data in customers.csv"
- "Run this SQL query: SELECT * FROM users WHERE age > 25"
- "Organize the file report.pdf into the reports folder"
- "Save the cleaned data to output.csv"

## Project Structure

```
data_agent/
├── agent.py          # Data agent configuration and instructions
├── agent_tools.py    # Data processing functions/tools the agent can use
├── requirements.txt  # Python dependencies
├── .env              # Environment variables (API keys) - NOT committed to git
├── raw_data/         # Directory for raw datasets
├── clean_data/       # Directory for cleaned/processed datasets
├── kaggle_data/      # Directory for Kaggle downloads
├── venv/             # Virtual environment (not committed to git)
└── README.md         # This file
```

## How It Works

- **agent.py**: Contains the agent's configuration, instructions, and launches the REPL loop
- **agent_tools.py**: Contains all the functions the agent can use (reading, cleaning, analyzing data, etc.)

The agent uses GPT-4O Mini to understand your requests and automatically calls the appropriate tools to help you work with your data.

