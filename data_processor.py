import os
import tempfile
import requests
import pandas as pd
import re
import time
from urllib.parse import urlparse
from io import StringIO


# Sensitive data patterns - enhanced for better detection
SENSITIVE_PATTERNS = [
    # Vulnerability and security-related patterns
    r'(?i)vulnerab(le|ility|ilities)',
    r'(?i)data\s*breach(es)?',
    r'(?i)security\s*breach(es)?',
    r'(?i)breach(es)?',
    r'(?i)confidential',
    r'(?i)sensitive',
    r'(?i)exploit(s|ation)?',
    r'(?i)attack(s|er)?',
    r'(?i)threat(s)?',
    r'(?i)risk(s)?',
    r'(?i)critical',
    r'(?i)malware',
    r'(?i)virus(es)?',
    r'(?i)trojan',
    r'(?i)ransomware',
    r'(?i)phishing',
    r'(?i)injection',
    r'(?i)xss|cross.site.scripting',
    r'(?i)sql.injection',
    r'(?i)buffer.overflow',
    r'(?i)zero.day',
    r'(?i)backdoor',

    # Credential and authentication patterns
    r'(?i)credentials?',
    r'(?i)password(s)?',
    r'(?i)secret(s)?',
    r'(?i)private\s*key(s)?',
    r'(?i)api\s*key(s)?',
    r'(?i)access\s*token(s)?',
    r'(?i)auth(entication)?',
    r'(?i)login',
    r'(?i)username(s)?',

    # Personal and financial data patterns
    r'(?i)ssn|social\s*security',
    r'(?i)credit\s*card',
    r'(?i)cvv|cvc',
    r'(?i)personal\s*data',
    r'(?i)pii|personally.identifiable',
    r'(?i)financial\s*data',
    r'(?i)bank\s*account',
    r'(?i)routing\s*number',

    # Names and personal identifiers (specific patterns to avoid false positives)
    r'(?i)\b(first\s*name|last\s*name|full\s*name|employee\s*name|customer\s*name|person\s*name|staff\s*name|user\s*name|contact\s*name)\b',
    r'(?i)\b(surname|firstname|lastname)\b',
    r'(?i)\b(mr\.|mrs\.|ms\.|dr\.|prof\.)\s*[A-Z][a-z]+',

    # Phone numbers and contact info
    r'(?i)\b(phone|mobile|cell|telephone|contact)\s*(number|no\.?|#)?\b',
    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US phone format
    r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b',  # International phone
    r'\b\d{10,15}\b',  # General phone number pattern

    # Addresses and postal codes
    r'(?i)\b(address|street|city|state|zip|postal|pincode|pin\s*code)\b',
    r'\b\d{5,6}(-\d{4})?\b',  # ZIP/PIN codes
    r'(?i)\b(apartment|apt|suite|unit|floor)\s*\d+\b',

    # Salary and compensation data
    r'(?i)\b(salary|wage|income|compensation|pay|earnings|bonus)\b',
    r'(?i)\b(annual|monthly|hourly)\s*(salary|wage|pay|income)\b',
    r'\$\s*\d{1,3}(,\d{3})*(\.\d{2})?',  # Currency amounts
    r'\b\d{1,3}(,\d{3})*\s*(dollars|usd|inr|rupees)\b',

    # Email addresses
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

    # ID numbers and sensitive identifiers
    r'(?i)\b(employee\s*id|emp\s*id|staff\s*id|user\s*id)\b',
    r'(?i)\b(id\s*number|identification|passport|license)\b',

    # Compliance and regulatory patterns
    r'(?i)gdpr',
    r'(?i)hipaa',
    r'(?i)pci.dss',
    r'(?i)sox|sarbanes.oxley',
    r'(?i)compliance',
    r'(?i)audit',
    r'(?i)regulatory',
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern) for pattern in SENSITIVE_PATTERNS]

def is_excel_or_csv_url(url):
    """Check if URL points to an Excel or CSV file by examining the file extension."""
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        if path_parts and path_parts[-1]:
            filename = path_parts[-1].lower()
            return filename.endswith((".xlsx", ".xls", ".csv"))
    except Exception:
        # Return False if URL parsing fails
        pass
    return False


def check_url_accessibility(url):
    """Test if a URL can be accessed and return status information."""
    try:
        # Try a HEAD request first (faster, doesn't download content)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        
        # If HEAD request fails, try a GET request with stream=True
        if response.status_code >= 400:
            response = requests.get(url, headers=headers, timeout=10, stream=True, allow_redirects=True)
            # Just get the headers, don't download the content
            response.close()
        
        if response.status_code == 200:
            return True, 200, ""
        else:
            return False, response.status_code, f"URL returned status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        # Return error details if request fails
        return False, 0, str(e)

def contains_sensitive_data(text):
    """Check if text contains potentially sensitive information like personal data."""
    if not text:
        return False

    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    # Check against all patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return True

    return False

def analyze_sensitive_data_detailed(text):
    """Analyze text for sensitive data and return detailed information about what was found."""
    if not text:
        return {"has_sensitive_data": False, "patterns_found": []}

    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    patterns_found = []

    # Check against all patterns and collect matches
    for i, pattern in enumerate(COMPILED_PATTERNS):
        matches = pattern.findall(text)
        if matches:
            # Get the original pattern description
            pattern_desc = SENSITIVE_PATTERNS[i]
            patterns_found.append({
                "pattern": pattern_desc,
                "matches": matches[:3]  # Limit to first 3 matches to avoid overwhelming output
            })

    return {
        "has_sensitive_data": len(patterns_found) > 0,
        "patterns_found": patterns_found
    }

def process_excel_csv_from_url(url):
    """Download CSV/Excel from URL, detect sensitive data (PII/security terms), then summarize/answer.
    Returns a rich result dict with success flag, summary text, and optional DataFrame if safe.
    """
    start_time = time.time()
    processing_steps = []
    
    # Step 0: Check if URL is accessible
    processing_steps.append(f"Checking if URL is accessible: {url}")
    is_accessible, status_code, error_message = check_url_accessibility(url)
    
    if not is_accessible:
        
        processing_steps.append(f"URL is not accessible: Status {status_code}, Error: {error_message}")
        return {
            "success": False,
            "error": error_message,
            "user_friendly_error": f"The file at the provided URL could not be accessed. Please check that the URL is correct and the file is available.",
            "summary": f"Unable to access file at {url}",
            "processing_steps": processing_steps
        }
    
    processing_steps.append(f"URL is accessible with status code {status_code}")
    
    # Step 1: Download the file with robust error handling
    try:
        
        processing_steps.append(f"Attempting to download file from {url}")
        
        # Try with different request configurations
        try:
            # First attempt: standard request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*'
            }
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            processing_steps.append("Download successful with standard request")
        except requests.exceptions.RequestException as e:
            
            processing_steps.append(f"Standard download failed: {str(e)}")
            
            # Second attempt: with session and different headers (TLS verification ON)
            try:
                session = requests.Session()
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
                response = session.get(url, headers=headers, timeout=90)
                response.raise_for_status()
                processing_steps.append("Download successful with session and custom headers")
            except requests.exceptions.RequestException as e2:
                
                processing_steps.append(f"All download attempts failed: {str(e2)}")
                raise Exception(f"Failed to download file after multiple attempts: {str(e2)}")
        
        # Check if we got any content
        if not response.content:
            error_msg = "Downloaded file is empty"
            
            processing_steps.append(error_msg)
            raise Exception(error_msg)
            
        content_length = len(response.content)
        processing_steps.append(f"Downloaded {content_length} bytes of data")
        
        
        # Step 2: Extract filename from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        if path_parts and path_parts[-1]:
            filename = path_parts[-1]
        else:
            # Try to determine file type from content or headers
            content_type = response.headers.get('Content-Type', '').lower()
            if 'excel' in content_type or 'spreadsheetml' in content_type:
                filename = "data.xlsx"
            elif 'csv' in content_type:
                filename = "data.csv"
            else:
                # Default based on URL
                filename = "data.xlsx" if ".xls" in url.lower() else "data.csv"
        
        processing_steps.append(f"Identified file as: {filename}")
        
        # Step 3: Save to temporary file with proper extension
        try:
            # Ensure we have the correct extension
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext or (file_ext not in ['.csv', '.xlsx', '.xls']):
                # Determine extension from content type
                if 'csv' in content_type:
                    file_ext = '.csv'
                elif 'excel' in content_type or 'spreadsheetml' in content_type:
                    file_ext = '.xlsx'
                else:
                    # Default to CSV if we can't determine
                    file_ext = '.csv'
            
            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
                
            processing_steps.append(f"Saved file to temporary location: {tmp_path}")
            
            
            # Verify file exists and has content
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise Exception("Failed to save file or file is empty")
                
        except Exception as save_error:
            
            processing_steps.append(f"Error saving file: {str(save_error)}")
            raise Exception(f"Failed to save downloaded file: {str(save_error)}")
        
        # Step 4: Process based on file type
        file_type = "csv" if filename.lower().endswith(".csv") else "excel"
        has_sensitive_data = False
        

        file_size_kb = os.path.getsize(tmp_path) / 1024
        processing_steps.append(f"Processing {file_type.upper()} file of size {file_size_kb:.2f} KB")
        
        
        try:
            if file_type == "csv":
                # Step 5A: Read CSV file with multiple fallback methods
                processing_steps.append("Attempting to read CSV file")
                
                # Define all the methods we'll try
                csv_methods = [
                    {"desc": "standard settings", "params": {"filepath_or_buffer": tmp_path}},
                    {"desc": "with UTF-8 encoding", "params": {"filepath_or_buffer": tmp_path, "encoding": "utf-8", "on_bad_lines": "skip", "low_memory": False}},
                    {"desc": "with Latin-1 encoding", "params": {"filepath_or_buffer": tmp_path, "encoding": "latin1", "on_bad_lines": "skip", "low_memory": False}},
                    {"desc": "with delimiter detection", "params": {"filepath_or_buffer": tmp_path, "delimiter": None, "on_bad_lines": "skip"}},
                    {"desc": "with tab delimiter", "params": {"filepath_or_buffer": tmp_path, "delimiter": "\t", "on_bad_lines": "skip"}},
                    {"desc": "with semicolon delimiter", "params": {"filepath_or_buffer": tmp_path, "delimiter": ";", "on_bad_lines": "skip"}},
                    {"desc": "as plain text", "params": {"filepath_or_buffer": tmp_path, "sep": None, "engine": "python", "on_bad_lines": "skip"}}
                ]
                
                # Try each method until one works
                success = False
                for method in csv_methods:
                    try:
                        
                        processing_steps.append(f"Trying to read CSV {method['desc']}")
                        
                        # Handle deprecated parameter names in pandas
                        params = method['params'].copy()
                        if 'error_bad_lines' in params:
                            params['on_bad_lines'] = 'skip'
                            del params['error_bad_lines']
                        if 'warn_bad_lines' in params:
                            del params['warn_bad_lines']
                            
                        df = pd.read_csv(**params)
                        success = True
                        processing_steps.append(f"Successfully read CSV file {method['desc']} with {len(df)} rows and {len(df.columns)} columns")

                        break
                    except Exception as e:
                        
                        processing_steps.append(f"Failed to read CSV {method['desc']}: {str(e)}")
                
                if not success:
                    # Try one last desperate approach - read as raw text
                    try:
                        with open(tmp_path, 'r', errors='ignore') as f:
                            raw_text = f.read()
                        
                        # Try to detect the delimiter by counting occurrences
                        delimiters = [',', ';', '\t', '|']
                        counts = {d: raw_text.count(d) for d in delimiters}
                        best_delimiter = max(counts, key=counts.get)
                        
                        # Try with the best delimiter

                        df = pd.read_csv(StringIO(raw_text), delimiter=best_delimiter, on_bad_lines='skip')
                        processing_steps.append(f"Successfully read CSV with raw text approach and delimiter '{best_delimiter}'")
                        success = True
                    except Exception as final_error:
                        
                        processing_steps.append("All CSV reading attempts failed")
                        raise Exception(f"Unable to read CSV file after trying multiple methods: {str(final_error)}")
            else:
                # Step 5B: Read Excel file with multiple fallback methods
                processing_steps.append("Attempting to read Excel file")
                
                # Define all the methods we'll try
                excel_methods = [
                    {"desc": "with default engine", "params": {"io": tmp_path, "sheet_name": None}},
                    {"desc": "with openpyxl engine", "params": {"io": tmp_path, "sheet_name": None, "engine": "openpyxl"}},
                    {"desc": "with xlrd engine", "params": {"io": tmp_path, "sheet_name": None, "engine": "xlrd"}},
                    {"desc": "first sheet only with default engine", "params": {"io": tmp_path, "sheet_name": 0}},
                    {"desc": "first sheet only with openpyxl engine", "params": {"io": tmp_path, "sheet_name": 0, "engine": "openpyxl"}}
                ]
                
                # Try each method until one works
                success = False
                for method in excel_methods:
                    try:
                        
                        processing_steps.append(f"Trying to read Excel {method['desc']}")
                        
                        excel_data = pd.read_excel(**method['params'])
                        
                        # Handle different return types based on sheet_name parameter
                        if isinstance(excel_data, dict):  # Multiple sheets
                            sheet_names = list(excel_data.keys())
                            processing_steps.append(f"Successfully read Excel file {method['desc']} with {len(sheet_names)} sheets")
                            
                            if sheet_names:
                                # Combine all sheets
                                df = pd.concat([excel_data[sheet] for sheet in sheet_names], ignore_index=True)
                                processing_steps.append(f"Combined data has {len(df)} rows and {len(df.columns)} columns")
                            else:
                                df = pd.DataFrame()
                                processing_steps.append("Excel file contains no data sheets")
                        else:  # Single sheet
                            df = excel_data
                            processing_steps.append(f"Successfully read Excel file {method['desc']} with {len(df)} rows and {len(df.columns)} columns")
                        
                        success = True
                        
                        break
                    except Exception as e:
                        
                        processing_steps.append(f"Failed to read Excel {method['desc']}: {str(e)}")
                
                if not success:
                    # Try one last approach - try to read as CSV in case it's mislabeled
                    try:
                        
                        processing_steps.append("Trying to read Excel file as CSV as last resort")
                        df = pd.read_csv(tmp_path, on_bad_lines='skip')
                        processing_steps.append(f"Successfully read file as CSV with {len(df)} rows and {len(df.columns)} columns")
                        # Update file type since we're reading it as CSV
                        file_type = "csv (originally marked as Excel)"
                        success = True
                    except Exception as final_error:
                        
                        processing_steps.append("All Excel reading attempts failed")
                        raise Exception(f"Unable to read Excel file after trying multiple methods: {str(final_error)}")
            
            # Enhanced sensitive data detection
            sensitive_columns = []
            sensitive_content_details = []

            # Check for sensitive data in column names with detailed analysis
            for col in df.columns:
                col_analysis = analyze_sensitive_data_detailed(col)
                if col_analysis["has_sensitive_data"]:
                    has_sensitive_data = True
                    sensitive_columns.append({
                        "column": col,
                        "patterns": col_analysis["patterns_found"]
                    })
                    processing_steps.append(f"Found sensitive data pattern in column name: {col}")

            # Check sample of data for sensitive patterns with column-by-column analysis
            sample_size = min(100, len(df))
            sample_df = df.head(sample_size)

            # Analyze each column individually for better detection
            for col in df.columns:
                # Convert column data to string and analyze
                col_data = sample_df[col].astype(str).str.cat(sep=' ')
                col_analysis = analyze_sensitive_data_detailed(col_data)

                if col_analysis["has_sensitive_data"]:
                    has_sensitive_data = True
                    sensitive_content_details.append({
                        "column": col,
                        "patterns": col_analysis["patterns_found"]
                    })
                    processing_steps.append(f"Found sensitive data patterns in column '{col}' content")

            # Also check the overall sample text as before
            sample_text = sample_df.to_string()
            overall_analysis = analyze_sensitive_data_detailed(sample_text)
            if overall_analysis["has_sensitive_data"]:
                has_sensitive_data = True
                processing_steps.append("Found sensitive data patterns in file content (overall analysis)")
            
            # Generate summary statistics
            column_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                non_null = df[col].count()
                null_percent = (len(df) - non_null) / len(df) * 100 if len(df) > 0 else 0
                
                col_info = {
                    "name": col,
                    "type": col_type,
                    "non_null_count": int(non_null),
                    "null_percent": f"{null_percent:.1f}%"
                }
                
                # Add sample values if not sensitive
                if not has_sensitive_data and not contains_sensitive_data(col):
                    unique_values = df[col].nunique()
                    col_info["unique_values"] = int(unique_values)
                    
                    # Add sample values for columns with few unique values
                    if unique_values <= 10:
                        col_info["sample_values"] = df[col].dropna().unique()[:10].tolist()
                
                column_info.append(col_info)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Build summary
            if has_sensitive_data:
                summary = f"⚠️ CONFIDENTIAL DATA DETECTED ⚠️\n\n"
                summary += f"The file '{filename}' contains potentially sensitive or confidential information.\n"
                summary += "To protect this data, detailed content cannot be displayed.\n\n"
                summary += f"File Type: {file_type.upper()}\n"
                summary += f"Total Rows: {len(df)}\n"
                summary += f"Total Columns: {len(df.columns)}\n\n"
                summary += "For security reasons, column names and data samples have been withheld."
            else:
                summary = f"File: {filename}\n\n"
                summary += f"File Type: {file_type.upper()}\n"
                summary += f"Total Rows: {len(df)}\n"
                summary += f"Total Columns: {len(df.columns)}\n\n"
                
                # Add column information
                summary += "Columns:\n"
                for col in column_info:
                    summary += f"  - {col['name']} ({col['type']}): {col['non_null_count']} non-null values ({col['null_percent']} null)\n"
                    if "unique_values" in col:
                        summary += f"    Unique Values: {col['unique_values']}\n"
                    if "sample_values" in col and col["sample_values"]:
                        sample_str = ", ".join([str(v) for v in col["sample_values"][:5]])
                        if len(col["sample_values"]) > 5:
                            sample_str += ", ..."
                        summary += f"    Samples: {sample_str}\n"
                
                # Add data preview if not sensitive
                summary += "\nData Preview (first 5 rows):\n"
                preview = df.head(5).to_string(index=False)
                summary += preview
            
            processing_time = time.time() - start_time
            
            
            return {
                "success": True,
                "summary": summary,
                "filename": filename,
                "file_type": file_type,
                "has_sensitive_data": has_sensitive_data,
                "sensitive_columns": sensitive_columns if has_sensitive_data else [],
                "sensitive_content_details": sensitive_content_details if has_sensitive_data else [],
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": list(df.columns) if not has_sensitive_data else [],
                "dataframe": df if not has_sensitive_data else None,  # Include the actual data for question answering
                "processing_steps": processing_steps,
                "processing_time": processing_time
            }
            
        except Exception as e:
            
            # Generate a more user-friendly error message
            error_msg = str(e)
            user_friendly_msg = f"Unable to process this {file_type.upper()} file"
            
            # Add more context based on common errors
            if "No such file" in error_msg:
                user_friendly_msg += ": The file could not be found or accessed"
            elif "Permission denied" in error_msg:
                user_friendly_msg += ": Permission denied when accessing the file"
            elif "UnicodeDecodeError" in error_msg:
                user_friendly_msg += ": The file contains characters that could not be decoded"
            elif "ParserError" in error_msg or "parser error" in error_msg.lower():
                user_friendly_msg += ": The file format appears to be invalid or corrupted"
            else:
                user_friendly_msg += ": An unexpected error occurred while processing the file"
            
            return {
                "success": False,
                "error": error_msg,
                "user_friendly_error": user_friendly_msg,
                "summary": user_friendly_msg,
                "filename": filename,
                "file_type": file_type,
                "processing_steps": processing_steps
            }
            
    except Exception as e:

        # Generate a more user-friendly error message
        error_msg = str(e)
        user_friendly_msg = "Unable to access or process the file"
        
        # Add more context based on common errors
        if "ConnectionError" in error_msg or "ConnectTimeout" in error_msg:
            user_friendly_msg = "Unable to connect to the server hosting this file. The URL may be incorrect or the server may be down."
        elif "SSLError" in error_msg:
            user_friendly_msg = "Secure connection failed when accessing this file."
        elif "404" in error_msg:
            user_friendly_msg = "The file could not be found at the specified URL (404 error)."
        elif "403" in error_msg:
            user_friendly_msg = "Access to this file is forbidden (403 error). You may not have permission to access it."
        elif "timeout" in error_msg.lower():
            user_friendly_msg = "The request to download this file timed out. The file may be too large or the server may be slow."
        
        return {
            "success": False,
            "error": error_msg,
            "user_friendly_error": user_friendly_msg,
            "summary": user_friendly_msg
        }

def answer_question_from_data(question, data_result):
    
    if not data_result["success"]:
        # Use the user-friendly error message if available
        return data_result.get("user_friendly_error",
                             data_result.get("summary",
                                           f"Sorry, I couldn't analyze the data file: {data_result.get('error', 'Unknown error')}"))

    if data_result.get("has_sensitive_data", False):
        # Provide more specific feedback about what type of sensitive data was found
        sensitive_msg = "This file contains sensitive or confidential information"

        # Check if we have details about what was found
        sensitive_columns = data_result.get("sensitive_columns", [])
        sensitive_content = data_result.get("sensitive_content_details", [])

        if sensitive_columns or sensitive_content:
            sensitive_msg += " including potential security-related data, personal information, or confidential content"

        sensitive_msg += ". For security and privacy reasons, I cannot provide specific details about its contents or answer questions based on this data."

        return sensitive_msg

    # Get the actual dataframe for analysis
    df = data_result.get("dataframe")
    if df is None or df.empty:
        return "No data available to answer your question."

    # Analyze the question and search through the data
    return analyze_question_and_search_data(question, df, data_result)

def analyze_question_and_search_data(question, df, data_result):
    
    question_lower = question.lower()

    try:
        # Question type 1: Who is the highest paid individual in a specific pincode?
        if "highest paid" in question_lower and "pincode" in question_lower:
            # Extract pincode from question
            pincode_match = re.search(r'pincode\s+(\d+)', question_lower)
            if pincode_match:
                target_pincode = int(pincode_match.group(1))

                # Filter data by pincode
                pincode_data = df[df['Pincode'] == target_pincode] if 'Pincode' in df.columns else pd.DataFrame()

                if pincode_data.empty:
                    return f"No data found for pincode {target_pincode}."

                # Find highest paid individual
                if 'Salary' in pincode_data.columns:
                    highest_paid = pincode_data.loc[pincode_data['Salary'].idxmax()]
                    name = highest_paid.get('Name', 'Unknown')
                    salary = highest_paid.get('Salary', 'Unknown')
                    phone = highest_paid.get('Mobile Number', 'Not available')

                    return f"The highest paid individual in pincode {target_pincode} is {name} with a salary of ₹{salary:,}. Phone number: {phone}"
                else:
                    return f"Salary information not available for pincode {target_pincode}."
            else:
                return "Please specify a valid pincode in your question."

        # Question type 2: Tell me the name of any 1 person from a specific pincode
        if "name" in question_lower and "person" in question_lower and "pincode" in question_lower:
            pincode_match = re.search(r'pincode\s+(\d+)', question_lower)
            if pincode_match:
                target_pincode = int(pincode_match.group(1))

                # Filter data by pincode
                pincode_data = df[df['Pincode'] == target_pincode] if 'Pincode' in df.columns else pd.DataFrame()

                if pincode_data.empty:
                    return f"No data found for pincode {target_pincode}."

                # Get first person from that pincode
                if 'Name' in pincode_data.columns:
                    first_person = pincode_data.iloc[0]
                    name = first_person.get('Name', 'Unknown')
                    return f"One person from pincode {target_pincode} is {name}."
                else:
                    return f"Name information not available for pincode {target_pincode}."
            else:
                return "Please specify a valid pincode in your question."

        # Question type 3: How many [specific name] exists in the document?
        if "how many" in question_lower and "exist" in question_lower:
            # Extract the name from the question
            name_match = re.search(r'how many\s+([^?]+?)\s+exist', question_lower)
            if name_match:
                target_name = name_match.group(1).strip()

                if 'Name' in df.columns:
                    # Count occurrences of the name (case-insensitive)
                    count = df['Name'].str.lower().str.contains(target_name.lower(), na=False).sum()
                    return f"There are {count} occurrences of '{target_name}' in the document."
                else:
                    return "Name information not available in the document."
            else:
                return "Please specify a name to search for."

        # Question type 4: Give me the contact number of [specific person]
        if "contact number" in question_lower or "phone number" in question_lower:
            # Extract the name from the question
            name_patterns = [
                r'contact number of\s+([^?.]+)',
                r'phone number of\s+([^?.]+)',
                r'number of\s+([^?.]+)'
            ]

            target_name = None
            for pattern in name_patterns:
                name_match = re.search(pattern, question_lower)
                if name_match:
                    target_name = name_match.group(1).strip()
                    break

            if target_name and 'Name' in df.columns:
                # Find the person (case-insensitive)
                person_data = df[df['Name'].str.lower().str.contains(target_name.lower(), na=False)]

                if person_data.empty:
                    return f"No person named '{target_name}' found in the document."

                if 'Mobile Number' in person_data.columns:
                    # If multiple matches, return the first one
                    contact = person_data.iloc[0]['Mobile Number']
                    actual_name = person_data.iloc[0]['Name']
                    return f"The contact number of {actual_name} is {contact}."
                else:
                    return f"Contact number information not available for {target_name}."
            else:
                return "Please specify a person's name to get their contact number."

        # Question type 5: What is the salary of [specific person]?
        if "salary of" in question_lower:
            # Extract the name from the question
            name_match = re.search(r'salary of\s+([^?.]+)', question_lower)
            if name_match:
                target_name = name_match.group(1).strip()

                if 'Name' in df.columns:
                    # Find the person (case-insensitive)
                    person_data = df[df['Name'].str.lower().str.contains(target_name.lower(), na=False)]

                    if person_data.empty:
                        return f"No person named '{target_name}' found in the document."

                    if 'Salary' in person_data.columns:
                        # If multiple matches, return the first one
                        salary = person_data.iloc[0]['Salary']
                        actual_name = person_data.iloc[0]['Name']
                        return f"The salary of {actual_name} is ₹{salary:,}."
                    else:
                        return f"Salary information not available for {target_name}."
                else:
                    return "Name information not available in the document."
            else:
                return "Please specify a person's name to get their salary."

        # Generic fallback: provide file summary
        return data_result.get("summary", "I couldn't find specific information to answer your question based on the available data.")

    except Exception as e:
        
        return f"Sorry, I encountered an error while analyzing your question: {str(e)}"