# agent_tools.py
# Functions/tools that the agent can use

import pandas as pd
from pathlib import Path
import os
import inspect
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

# Base directory for file operations
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
CLEAN_DATA_DIR = BASE_DIR / "clean_data"
KAGGLE_DATA_DIR = BASE_DIR / "kaggle_data"

# Ensure directories exist
CLEAN_DATA_DIR.mkdir(exist_ok=True)
KAGGLE_DATA_DIR.mkdir(exist_ok=True)


# read content of a text file
def read_file(filename: str) -> str:
    file_path = BASE_DIR / filename
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Could not read {file_path}: {e}"


# read CSV file
def read_csv(filename: str) -> str:
    """Read a CSV file and return a summary of its structure and first few rows.
    If filename doesn't contain a path separator, looks in raw_data/ directory."""
    # If filename is just a name (no path), look in raw_data/
    if '/' not in filename and '\\' not in filename:
        file_path = RAW_DATA_DIR / filename
    else:
        file_path = BASE_DIR / filename
    try:
        df = pd.read_csv(file_path)
        summary = f"CSV file: {filename}\n"
        summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary += "First 5 rows:\n"
        summary += df.head().to_string()
        return summary
    except Exception as e:
        return f"Could not read CSV {file_path}: {e}"


# read Excel file
def read_excel(filename: str, sheet_name: str = None) -> str:
    """Read an Excel file and return a summary of its structure.
    If filename doesn't contain a path separator, looks in raw_data/ directory."""
    # If filename is just a name (no path), look in raw_data/
    if '/' not in filename and '\\' not in filename:
        file_path = RAW_DATA_DIR / filename
    else:
        file_path = BASE_DIR / filename
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        summary = f"Excel file: {filename}\n"
        summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary += "First 5 rows:\n"
        summary += df.head().to_string()
        return summary
    except Exception as e:
        return f"Could not read Excel {file_path}: {e}"


# analyze dataset
def analyze_dataset(filename: str) -> str:
    """Analyze a dataset (CSV/Excel) and return statistics and insights.
    If filename doesn't contain a path separator, looks in raw_data/ directory."""
    # If filename is just a name (no path), look in raw_data/
    if '/' not in filename and '\\' not in filename:
        file_path = RAW_DATA_DIR / filename
    else:
        file_path = BASE_DIR / filename
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file type. Use CSV or Excel files."
        
        summary = f"Analysis of {filename}:\n\n"
        summary += f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        summary += "Column info:\n"
        summary += str(df.info())
        summary += "\n\nBasic statistics:\n"
        summary += df.describe().to_string()
        summary += "\n\nMissing values:\n"
        summary += df.isnull().sum().to_string()
        return summary
    except Exception as e:
        return f"Could not analyze {file_path}: {e}"


# clean dataset
def clean_dataset(filename: str, remove_duplicates: bool = True, fill_missing: str = "drop", output_filename: str = None) -> str:
    """
    Clean a dataset by removing duplicates, handling missing values, and standardizing formats.
    
    Args:
        filename: Input file to clean (if just a filename, looks in raw_data/)
        remove_duplicates: Whether to remove duplicate rows (default: True)
        fill_missing: How to handle missing values - "drop" (drop rows), "fill" (fill with mean/median), or "zero" (fill with 0)
        output_filename: Optional output filename. If not provided, saves as "cleaned_{original_filename}"
    """
    # If filename is just a name (no path), look in raw_data/
    if '/' not in filename and '\\' not in filename:
        file_path = RAW_DATA_DIR / filename
    else:
        file_path = BASE_DIR / filename
    try:
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file type. Use CSV or Excel files."
        
        original_shape = df.shape
        changes = []
        
        # Remove duplicates
        if remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            if before != after:
                changes.append(f"Removed {before - after} duplicate rows")
        
        # Handle missing values
        if fill_missing == "drop":
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                changes.append(f"Dropped {before - after} rows with missing values")
        elif fill_missing == "fill":
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
            changes.append("Filled missing numeric values with median")
        elif fill_missing == "zero":
            df = df.fillna(0)
            changes.append("Filled missing values with 0")
        
        # Standardize column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        changes.append("Standardized column names")
        
        # Determine output filename
        if output_filename is None:
            if filename.endswith('.csv'):
                output_filename = f"cleaned_{filename}"
            else:
                output_filename = f"cleaned_{filename.rsplit('.', 1)[0]}.csv"
        
        # Save to clean_data/ directory
        output_path = CLEAN_DATA_DIR / output_filename
        df.to_csv(output_path, index=False)
        
        result = f"Cleaned dataset saved to clean_data/{output_filename}\n"
        result += f"Original shape: {original_shape[0]} rows × {original_shape[1]} columns\n"
        result += f"Cleaned shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        result += f"Changes made:\n" + "\n".join(f"  - {change}" for change in changes)
        return result
    except Exception as e:
        return f"Could not clean {file_path}: {e}"


# save dataset
def save_dataset(filename: str, output_filename: str, format: str = "csv") -> str:
    """
    Save a dataset to a new file in the specified format. Saves to clean_data/ directory.
    
    Args:
        filename: Input file to save (if just a filename, looks in raw_data/)
        output_filename: Name of the output file (will be saved in clean_data/)
        format: Output format - "csv" or "excel" (default: "csv")
    """
    # If filename is just a name (no path), look in raw_data/
    if '/' not in filename and '\\' not in filename:
        file_path = RAW_DATA_DIR / filename
    else:
        file_path = BASE_DIR / filename
    try:
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported input file type. Use CSV or Excel files."
        
        # Save to clean_data/ directory
        output_path = CLEAN_DATA_DIR / output_filename
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        else:
            return f"Unsupported output format. Use 'csv' or 'excel'."
        
        return f"Dataset saved to clean_data/{output_filename} ({df.shape[0]} rows × {df.shape[1]} columns)"
    except Exception as e:
        return f"Could not save dataset: {e}"


# interpret data
def interpret_data(filename: str) -> str:
    """
    Provide detailed interpretation and insights about a dataset.
    If filename doesn't contain a path separator, looks in raw_data/ directory.
    """
    # If filename is just a name (no path), look in raw_data/
    if '/' not in filename and '\\' not in filename:
        file_path = RAW_DATA_DIR / filename
    else:
        file_path = BASE_DIR / filename
    try:
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file type. Use CSV or Excel files."
        
        interpretation = f"Data Interpretation for {filename}:\n\n"
        
        # Basic info
        interpretation += f"Dataset Overview:\n"
        interpretation += f"  - Total records: {df.shape[0]:,}\n"
        interpretation += f"  - Total columns: {df.shape[1]}\n"
        interpretation += f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n\n"
        
        # Column types and insights
        interpretation += "Column Analysis:\n"
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            
            interpretation += f"  - {col} ({dtype}): "
            if null_count > 0:
                interpretation += f"{null_count} missing values ({null_pct:.1f}%)\n"
            else:
                interpretation += "No missing values\n"
            
            # Add specific insights based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                interpretation += f"    Range: {df[col].min()} to {df[col].max()}, Mean: {df[col].mean():.2f}\n"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                interpretation += f"    Date range: {df[col].min()} to {df[col].max()}\n"
            else:
                unique_count = df[col].nunique()
                interpretation += f"    {unique_count} unique values"
                if unique_count <= 10:
                    interpretation += f": {', '.join(map(str, df[col].unique()[:10]))}\n"
                else:
                    interpretation += f"\n"
        
        # Data quality summary
        interpretation += "\nData Quality Summary:\n"
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        interpretation += f"  - Completeness: {((total_cells - missing_cells) / total_cells * 100):.1f}%\n"
        interpretation += f"  - Duplicate rows: {df.duplicated().sum()}\n"
        
        return interpretation
    except Exception as e:
        return f"Could not interpret {file_path}: {e}"


# organize files
def organize_files(filename: str, target_folder: str = None) -> str:
    """
    Organize files into folders for better data management.
    
    Args:
        filename: Name of the file to organize (if just a filename, looks in current directory)
        target_folder: Target folder name. If not provided, organizes based on file type.
    """
    try:
        import shutil
        
        # If filename is just a name (no path), look in BASE_DIR
        if '/' not in filename and '\\' not in filename:
            file_path = BASE_DIR / filename
        else:
            file_path = BASE_DIR / filename
        
        if not file_path.exists():
            return f"File {file_path} not found."
        
        # Determine target folder
        if target_folder is None:
            # Organize by file extension
            ext = file_path.suffix.lower()
            if ext in ['.csv', '.xlsx', '.xls']:
                target_folder = "data_files"
            elif ext in ['.txt', '.md', '.json']:
                target_folder = "text_files"
            elif ext in ['.pdf', '.doc', '.docx']:
                target_folder = "documents"
            else:
                target_folder = "other_files"
        
        # Create target folder if it doesn't exist
        target_dir = BASE_DIR / target_folder
        target_dir.mkdir(exist_ok=True)
        
        # Move file
        destination = target_dir / file_path.name
        shutil.move(str(file_path), str(destination))
        
        return f"File {filename} moved to {target_folder}/{file_path.name}"
    except Exception as e:
        return f"Could not organize file: {e}"


# execute database query
def execute_query(query: str, database_url: str = None) -> str:
    """
    Execute a SQL query against a database and return the results.
    
    Args:
        query: SQL query to execute (SELECT, INSERT, UPDATE, DELETE, etc.)
        database_url: Database connection string. If not provided, uses DATABASE_URL from .env file.
                     Examples:
                     - SQLite: "sqlite:///database.db"
                     - PostgreSQL: "postgresql://user:password@localhost/dbname"
                     - MySQL: "mysql://user:password@localhost/dbname"
    """
    try:
        # Get database URL from parameter or environment variable
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
            if database_url is None:
                return "Error: No database URL provided. Set DATABASE_URL in .env file or pass it as a parameter."
        
        # Create database connection
        engine = create_engine(database_url)
        
        # Execute query
        with engine.connect() as connection:
            # For SELECT queries, return results as a formatted table
            if query.strip().upper().startswith('SELECT'):
                df = pd.read_sql(text(query), connection)
                if df.empty:
                    return "Query executed successfully. No rows returned."
                
                result_str = f"Query returned {len(df)} rows:\n\n"
                result_str += df.to_string(index=False)
                return result_str
            else:
                # For INSERT, UPDATE, DELETE, etc., execute and commit
                result = connection.execute(text(query))
                connection.commit()
                return f"Query executed successfully. Rows affected: {result.rowcount if hasattr(result, 'rowcount') else 'N/A'}"
                
    except Exception as e:
        return f"Database error: {str(e)}"


# download dataset from Kaggle
def download_kaggle_dataset(dataset: str, unzip: bool = True) -> str:
    """
    Download a dataset from Kaggle and save it to the kaggle_data/ directory.
    
    Args:
        dataset: Kaggle dataset identifier in format "username/dataset-name" (e.g., "titanic" or "username/titanic")
        unzip: Whether to automatically unzip downloaded files (default: True)
    
    Returns:
        A message indicating success and where files were saved
    """
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        
        # Authenticate - checks for KAGGLE_USERNAME/KAGGLE_KEY env vars or ~/.kaggle/kaggle.json
        api.authenticate()
        
        # Ensure kaggle_data directory exists
        KAGGLE_DATA_DIR.mkdir(exist_ok=True)
        
        # Download dataset to kaggle_data/
        api.dataset_download_files(dataset, path=str(KAGGLE_DATA_DIR), unzip=unzip)
        
        # Get list of downloaded files
        if unzip:
            # If unzipped, list all files in kaggle_data/
            downloaded_files = list(KAGGLE_DATA_DIR.glob("*"))
            # Filter out directories
            files = [f.name for f in downloaded_files if f.is_file()]
        else:
            # If not unzipped, look for zip files
            files = [f.name for f in KAGGLE_DATA_DIR.glob("*.zip")]
        
        if not files:
            return f"Dataset '{dataset}' downloaded but no files found in kaggle_data/"
        
        result = f"Successfully downloaded dataset '{dataset}' to kaggle_data/\n"
        result += f"Files downloaded: {', '.join(files[:10])}"  # Show first 10 files
        if len(files) > 10:
            result += f" (and {len(files) - 10} more files)"
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return f"Error: Kaggle authentication failed. Please set up your Kaggle API credentials using one of these methods:\n\n" \
                   f"Method 1 (Environment Variables - Recommended):\n" \
                   f"1. Go to https://www.kaggle.com/settings and create an API token\n" \
                   f"2. Add to your .env file:\n" \
                   f"   KAGGLE_USERNAME=your_username\n" \
                   f"   KAGGLE_KEY=your_api_key\n\n" \
                   f"Method 2 (kaggle.json file):\n" \
                   f"1. Go to https://www.kaggle.com/settings and create an API token\n" \
                   f"2. Place kaggle.json in ~/.kaggle/ directory\n" \
                   f"3. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return f"Error: Dataset '{dataset}' not found. Make sure the dataset name is correct (format: 'username/dataset-name')"
        else:
            return f"Error downloading dataset '{dataset}': {error_msg}"


def get_function_schema(func):
    """Convert a Python function to OpenAI function calling schema."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        param_type = "string"  # default
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
        
        param_info = {"type": param_type, "description": ""}
        
        if param.default != inspect.Parameter.empty:
            # Optional parameter - don't add to required
            if param.default is not None:
                param_info["default"] = str(param.default)
        else:
            required.append(param_name)
        
        properties[param_name] = param_info
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.split('\n')[0] if doc else f"Call the {func.__name__} function",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

