import requests
from tqdm import tqdm
import os
import re
import shutil
import threading
from urllib.parse import unquote

def get_civitai_model_id_and_version(url):
    """
    Extracts the model ID and version ID from a CivitAI URL.
    Handles multiple formats:
    - https://civitai.com/models/123456
    - https://civitai.com/models/123456?modelVersionId=789
    - models/123456
    - 123456 (just the number)
    """
    url = str(url).strip()
    
    # Try to find model ID from URL format (models/123456)
    model_id_match = re.search(r'models/(\d+)', url)
    if model_id_match:
        model_id = model_id_match.group(1)
    else:
        # Try just a number (123456)
        number_match = re.search(r'^(\d+)', url)
        if number_match:
            model_id = number_match.group(1)
        else:
            model_id = None
    
    # Look for version ID
    version_id_match = re.search(r'modelVersionId=(\d+)', url)
    version_id = version_id_match.group(1) if version_id_match else None
    
    return model_id, version_id

def sanitize_filename(filename):
    """
    Remove invalid characters from filename for cross-platform compatibility.
    Ensures safe filenames on Windows, Linux, and macOS.
    """
    if not filename:
        return "downloaded_file"
    
    # URL decode first (handles %20 etc.)
    filename = unquote(filename)
    
    # Get just the filename, not any path components
    filename = os.path.basename(filename)
    
    # Replace invalid characters with underscore
    # Windows: <>:"|?*
    # Path separators: /\
    invalid_chars = '<>:"|?*/\\'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters (ASCII 0-31)
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Remove trailing dots and spaces (Windows requirement)
    filename = filename.rstrip('. ')
    
    # Remove leading dots and spaces
    filename = filename.lstrip('. ')
    
    # Ensure it's not empty after sanitization
    if not filename:
        filename = "downloaded_file"
    
    # Truncate to 200 chars to leave room for extensions and temp suffix
    # (filesystem limit is usually 255)
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename

class DownloadManager:
    active_downloads = {}
    _lock = threading.Lock()

    @staticmethod
    def cancel_download(node_id):
        node_id_str = str(node_id)
        
        print(f"===== CANCEL ATTEMPT =====")
        print(f"Cancelling node_id: {node_id_str}")
        
        with DownloadManager._lock:
            print(f"Active downloads: {list(DownloadManager.active_downloads.keys())}")
            
            if node_id_str in DownloadManager.active_downloads:
                print(f"Found and setting cancel event for: {node_id_str}")
                DownloadManager.active_downloads[node_id_str].set()
                return True
            
            print(f"No active download found for: {node_id_str}")
            return False

    @staticmethod
    def download_with_progress(url, save_path, filename=None, progress_callback=None, params=None, chunk_size=1024*1024, node_id=None):
        """
        Download a file with progress tracking and cancel support.
        
        Args:
            url: Download URL
            save_path: Directory to save file
            filename: Optional filename (if not provided, extracted from response/URL)
            progress_callback: Object with set_progress(percentage) method
            params: Query parameters for the request
            chunk_size: Download chunk size in bytes
            node_id: Node ID for cancel tracking
        """
        cancel_event = threading.Event()
        node_id_str = str(node_id) if node_id is not None else None
        
        if node_id_str:
            with DownloadManager._lock:
                DownloadManager.active_downloads[node_id_str] = cancel_event
                
                print(f"===== DOWNLOAD START =====")
                print(f"Registered cancel event for node_id: {node_id_str}")
                print(f"Active downloads now: {list(DownloadManager.active_downloads.keys())}")
        
        temp_path = None
        try:
            response = requests.get(url, stream=True, params=params)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Get filename: use provided, then Content-Disposition, then URL
            if not filename:
                filename = DownloadManager._extract_filename(response, url)
            
            # Sanitize filename for OS compatibility
            filename = sanitize_filename(filename)
            
            print(f"Downloading to: {os.path.join(save_path, filename)}")
            
            full_path = os.path.join(save_path, filename)
            temp_path = full_path + '.tmp'
            
            downloaded = 0
            with open(temp_path, 'wb') as file:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as pbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        # Check cancel event
                        if node_id_str and cancel_event.is_set():
                            print(f"===== DOWNLOAD CANCELLED =====")
                            print(f"Node {node_id_str} download was cancelled")
                            raise Exception("Download cancelled by user")

                        size = file.write(data)
                        downloaded += size
                        pbar.update(size)
                        pbar.refresh()
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100.0
                            progress_callback.set_progress(progress)
            
            shutil.move(temp_path, full_path)
            print(f"===== DOWNLOAD COMPLETE =====")
            print(f"Successfully downloaded: {full_path}")
            return full_path
            
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
            print(f"Error occurred during download: {str(e)}")
            raise
        finally:
            if node_id_str:
                with DownloadManager._lock:
                    if node_id_str in DownloadManager.active_downloads:
                        del DownloadManager.active_downloads[node_id_str]
                        print(f"Cleaned up cancel event for node: {node_id_str}")

    @staticmethod
    def _extract_filename(response, url):
        """
        Extract filename from response headers or URL.
        Handles multiple Content-Disposition formats including RFC 2231/5987.
        """
        content_disposition = response.headers.get('content-disposition', '')
        
        if content_disposition:
            # Try RFC 5987/2231 encoding first: filename*=UTF-8''file.txt
            match = re.search(r"filename\*=(?:UTF-8''|[^']*'[^']*')([^;\s]+)", content_disposition, re.IGNORECASE)
            if match:
                filename = unquote(match.group(1))
                print(f"Extracted filename from RFC 5987 encoding: {filename}")
                return filename
            
            # Try standard filename with quotes: filename="file.txt"
            match = re.search(r'filename=(["\'])([^"\']+)\1', content_disposition)
            if match:
                filename = match.group(2)
                print(f"Extracted filename from quoted Content-Disposition: {filename}")
                return filename
            
            # Try without quotes: filename=file.txt
            match = re.search(r'filename=([^;\s]+)', content_disposition)
            if match:
                filename = match.group(1).strip('"\'')
                print(f"Extracted filename from unquoted Content-Disposition: {filename}")
                return filename
        
        # Fallback to URL parsing
        filename = url.split('/')[-1].split('?')[0]
        if not filename:
            filename = "downloaded_file"
        
        print(f"Extracted filename from URL: {filename}")
        return filename
