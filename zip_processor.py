import os
import tempfile
import zipfile
import requests
import shutil
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed



def is_zip_url(url):
    """Check for ZIP and other archive files via path suffix or Content-Type."""
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        if path_parts and path_parts[-1]:
            filename = path_parts[-1].lower()
            # Check for various archive formats
            if filename.endswith((".zip", ".rar", ".7z", ".tar.gz", ".tar.bz2")):
                return True
        
        # If no filename in path, try to check content type
        if '?' in url and any(ext in url.lower() for ext in ['.zip', '.rar', '.7z', '.tar']):
            return True
            
        # Check Content-Type header for archive formats
        try:
            response = requests.head(url, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()
            if any(archive_type in content_type for archive_type in [
                'application/zip', 'application/x-zip-compressed', 
                'application/rar', 'application/x-rar-compressed', 
                'application/x-7z-compressed', 'application/x-tar',
                'application/gzip', 'application/x-gzip'
            ]):
                return True
        except Exception:
            pass
            
    except Exception as e:
        return False
    
    return False


def optimize_for_nested_zips(url):
    try:
        # Download just the beginning of the file to check its structure
        # (Most ZIP files have directory at the end, but we can get basic info from the start)
        headers = {'Range': 'bytes=0-8192'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # Get content length if available
        content_length = int(response.headers.get('Content-Length', 0))
        if content_length == 0 and 'Content-Range' in response.headers:
            # Try to parse from Content-Range header (bytes 0-8192/total)
            range_header = response.headers.get('Content-Range', '')
            if '/' in range_header:
                try:
                    content_length = int(range_header.split('/')[-1])
                except (ValueError, IndexError):
                    pass
        
        # Estimate file size and complexity
        file_size_mb = content_length / (1024 * 1024) if content_length > 0 else 10  # Default to 10MB if unknown
        
        # Determine optimal parameters based on file size
        if file_size_mb > 100:  # Very large ZIP (>100MB)
            return {
                'max_workers': min(64, (os.cpu_count() or 4) * 4),  # More workers for large files
                'max_depth': 15,  # Allow deeper nesting for complex archives
                'chunk_size': 1024 * 1024,  # 1MB chunks for large files
                'estimated_size_mb': file_size_mb
            }
        elif file_size_mb > 10:  # Medium ZIP (10-100MB)
            return {
                'max_workers': min(32, (os.cpu_count() or 4) * 2),
                'max_depth': 12,
                'chunk_size': 512 * 1024,  # 512KB chunks
                'estimated_size_mb': file_size_mb
            }
        else:  # Small ZIP (<10MB)
            return {
                'max_workers': min(16, (os.cpu_count() or 4) * 2),
                'max_depth': 10,
                'chunk_size': 256 * 1024,  # 256KB chunks
                'estimated_size_mb': file_size_mb
            }
            
    except Exception as e:
        # Default parameters
        return {
            'max_workers': (os.cpu_count() or 4) * 2,
            'max_depth': 10,
            'chunk_size': 512 * 1024,
            'estimated_size_mb': 10
        }

def process_zip_from_url(url, max_workers=None, detect_zip_trap=True):
    start_time = time.time()
    
    # Determine optimal number of workers based on CPU cores
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)  # Default to 2x CPU cores, max 32
    
    try:
        # Download the ZIP file
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        if path_parts and path_parts[-1]:
            filename = path_parts[-1]
        else:
            filename = "archive.zip"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Create a temporary directory for extraction
        base_extract_dir = tempfile.mkdtemp(prefix="zip_processor_")
        
        
        # Shared data structures with thread-safe access
        from threading import Lock
        lock = Lock()
        all_files = []
        nested_zips = []
        extracted_content = []
        processing_steps = []
        
        # Add a processing step with thread safety
        def add_step(step):
            with lock:
                processing_steps.append(step)
        
        # Add a file with thread safety
        def add_file(file_data):
            with lock:
                all_files.append(file_data)
        
        # Add a nested zip with thread safety
        def add_nested_zip(zip_data):
            with lock:
                nested_zips.append(zip_data)
        
        # Add extracted content with thread safety
        def add_content(content_data):
            with lock:
                extracted_content.append(content_data)
        
        # Process a text file to extract its content
        def process_text_file(zip_ref, file_info, current_path):
            try:
                with zip_ref.open(file_info) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    # Truncate very large content
                    if len(content) > 10000:
                        content = content[:5000] + "\n\n[...content truncated...]\n\n" + content[-5000:]
                    add_content({
                        "path": current_path,
                        "content": content
                    })
            except Exception as e:
                add_step(f"Error extracting text from {current_path}: {str(e)}")
        
        # Process a single ZIP file
        def process_zip_file(zip_path, prefix="", depth=0, max_depth=10, detect_zip_trap=detect_zip_trap):
            if depth > max_depth:
                add_step(f"Maximum recursion depth ({max_depth}) reached at {prefix or filename}")
                return {"error": "Maximum recursion depth reached"}
            
            add_step(f"Processing ZIP at depth {depth}: {prefix or filename}")
            
            try:
                # Safely open the ZIP file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.infolist()
                    add_step(f"Found {len(file_list)} files in {prefix or filename}")
                    
                    # Check if this is a ZIP trap (only contains other ZIP files)
                    if detect_zip_trap and depth == 0:  # Only check at root level
                        non_dir_files = [f for f in file_list if not f.is_dir()]
                        zip_files = [f for f in non_dir_files if f.filename.lower().endswith(".zip")]
                        
                        # If all files are ZIP files, it might be a trap
                        if len(zip_files) > 0 and len(zip_files) == len(non_dir_files):
                            add_step("⚠️ ZIP trap detected: Archive contains only nested ZIP files")
                            
                            # Return early with a warning
                            zip_names = [f.filename for f in zip_files]
                            return {
                                "is_zip_trap": True,
                                "zip_count": len(zip_files),
                                "zip_files": zip_names,
                                "depth": depth,
                                "path": prefix or filename
                            }
                    
                    # Extract nested ZIP files for parallel processing
                    nested_zip_tasks = []
                    extract_dirs = []
                    
                    # First pass: process regular files and identify nested ZIPs
                    for file_info in file_list:
                        if file_info.is_dir():
                            continue
                            
                        # Build the full path
                        current_path = f"{prefix}/{file_info.filename}" if prefix else file_info.filename
                        
                        # Record this file
                        file_data = {
                            "name": file_info.filename,
                            "path": current_path,
                            "size": file_info.file_size,
                            "depth": depth
                        }
                        add_file(file_data)
                        
                        # Check if it's a text file we can extract
                        if file_info.filename.lower().endswith((".txt", ".md", ".csv", ".json", ".html", ".xml")):
                            process_text_file(zip_ref, file_info, current_path)
                        
                        # Prepare nested ZIP files for parallel processing
                        if file_info.filename.lower().endswith(".zip"):
                            add_nested_zip({
                                "path": current_path,
                                "depth": depth + 1
                            })
                            add_step(f"Found nested ZIP: {current_path}")
                            
                            # Extract the nested ZIP to a temporary directory
                            extract_dir = tempfile.mkdtemp(dir=base_extract_dir)
                            extract_dirs.append(extract_dir)
                            nested_zip_path = os.path.join(extract_dir, file_info.filename)
                            zip_ref.extract(file_info, extract_dir)
                            
                            # Add this ZIP to the tasks for parallel processing
                            nested_zip_tasks.append((nested_zip_path, current_path, depth + 1))
                    
                    # Process nested ZIPs in parallel if there are any
                    if nested_zip_tasks:
                        # Use a thread pool to process nested ZIPs in parallel
                        with ThreadPoolExecutor(max_workers=min(max_workers, len(nested_zip_tasks) + 1)) as executor:
                            # Submit all nested ZIP processing tasks
                            future_to_zip = {executor.submit(process_zip_file, zip_path, prefix, depth, max_depth): 
                                            (zip_path, prefix) for zip_path, prefix, depth in nested_zip_tasks}
                            
                            # Process results as they complete
                            for future in as_completed(future_to_zip):
                                zip_path, prefix = future_to_zip[future]
                                try:
                                    result = future.result()
                                    if result.get("error"):
                                        add_step(f"Error in nested ZIP {prefix}: {result['error']}")
                                except Exception as exc:
                                    add_step(f"Exception processing {prefix}: {exc}")
                
                return {
                    "file_count": len(file_list),
                    "depth": depth,
                    "path": prefix or filename,
                    "nested_count": len(nested_zip_tasks)
                }
                
            except zipfile.BadZipFile as e:
                add_step(f"Bad ZIP file at {prefix or filename}: {str(e)}")
                return {"error": f"Bad ZIP file: {str(e)}"}
            except Exception as e:
                add_step(f"Error processing ZIP at {prefix or filename}: {str(e)}")
                return {"error": str(e)}
        
        # Start recursive processing with the main ZIP file
        result = process_zip_file(tmp_path)
        
        # Clean up temporary files
        try:
            os.unlink(tmp_path)
            shutil.rmtree(base_extract_dir, ignore_errors=True)
        except Exception as e:
            add_step(f"Error cleaning up temporary files: {str(e)}")
            
        # Check if we detected a ZIP trap
        if result.get("is_zip_trap", False):
            # Generate a more user-friendly summary for nested ZIP files
            zip_count = result.get("zip_count", 0)
            zip_files = result.get("zip_files", [])
            
            # Create a more informative and less alarming message
            nested_summary = f"The ZIP file '{filename}' contains {zip_count} nested ZIP files.\n\n"
            
            # Add more context about the structure
            if zip_count > 1:
                nested_summary += f"This appears to be an archive of archives with a nested structure.\n"
            else:
                nested_summary += f"This appears to be an archive containing another archive.\n"
                
            # List the nested ZIP files
            if zip_files:
                nested_summary += "\nNested archives include:\n"
                for zip_name in zip_files[:10]:  # Show first 10
                    nested_summary += f"  - {zip_name}\n"
                if len(zip_files) > 10:
                    nested_summary += f"  ... and {len(zip_files) - 10} more\n"
            
            # Add helpful information about the content
            nested_summary += f"\nTotal size: {len(response.content) / 1024:.2f} KB\n"
            nested_summary += "\nTo view the contents of these nested archives, you would need to extract them individually."
            
            return {
                "success": True,
                "is_zip_trap": True,
                "summary": nested_summary,
                "filename": filename,
                "size_kb": len(response.content) / 1024,
                "total_files": zip_count,
                "total_nested_zips": zip_count,
                "processing_steps": processing_steps,
                "zip_files": zip_files
            }
        
        # Generate file types statistics
        file_types = {}
        for file in all_files:
            ext = os.path.splitext(file['name'].lower())[1]
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['no_extension'] = file_types.get('no_extension', 0) + 1
        
        # Sort files by path for better readability
        all_files.sort(key=lambda x: x['path'])
        
        # Build a detailed summary
        processing_time = time.time() - start_time
        summary = f"ZIP Archive: {filename}\n\n"
        summary += f"Total Files: {len(all_files)}\n"
        summary += f"Total Size: {sum(f['size'] for f in all_files)/1024:.2f} KB\n"
        summary += f"Nested ZIP Files: {len(nested_zips)}\n"
        summary += f"Processing Time: {processing_time:.2f} seconds\n\n"
        
        if file_types:
            summary += "File Types:\n"
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                summary += f"  - {ext}: {count} file(s)\n"
            summary += "\n"
        
        if nested_zips:
            summary += "Nested ZIP Files:\n"
            for zip_info in sorted(nested_zips, key=lambda x: x['path']):
                summary += f"  - {zip_info['path']} (depth: {zip_info['depth']})\n"
            summary += "\n"
        
        # List files (limit to first 50)
        summary += "Files:\n"
        for i, file in enumerate(all_files[:50]):
            summary += f"  - {file['path']} ({file['size']/1024:.2f} KB)\n"
        
        if len(all_files) > 50:
            summary += f"  ... and {len(all_files) - 50} more files\n"
        
        # Add extracted text content (limit to first 5)
        if extracted_content:
            summary += "\nExtracted Content Samples:\n"
            for i, content in enumerate(extracted_content[:5]):
                summary += f"\n--- {content['path']} ---\n{content['content'][:1000]}"
                if len(content['content']) > 1000:
                    summary += "...\n"
                summary += f"--- End of {content['path']} ---\n"
                
            if len(extracted_content) > 5:
                summary += f"\n... and {len(extracted_content) - 5} more text files\n"
        
        
        # Return the analysis results
        return {
            "success": True,
            "summary": summary,
            "filename": filename,
            "size_kb": len(response.content) / 1024,
            "total_files": len(all_files),
            "total_nested_zips": len(nested_zips),
            "processing_time": processing_time,
            "file_types": file_types,
            "nested_zips": [nz['path'] for nz in nested_zips],
            "processing_steps": processing_steps,
            "extracted_content": extracted_content[:10],  # Limit to first 10 text files
            "files": all_files[:100]  # Limit to first 100 files
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "summary": f"Error processing ZIP file: {str(e)}"
        }