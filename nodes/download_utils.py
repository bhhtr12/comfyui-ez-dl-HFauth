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

class DownloadCancelled(Exception):
    """Raised when user cancels the download via the node UI."""
    pass

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
    def download_with_progress(url, save_path, filename=None, progress_callback=None, params=None, chunk_size=1024*1024, node_id=None, headers=None):
        """
        Download with:
        - Full ComfyUI progress bar
        - Cancel support (deletes temp on cancel only)
        - Resume support (keeps .tmp on network errors, resumes on next run)
        - Headers support (for HF private/gated Bearer token)
        """
        cancel_event = threading.Event()
        node_id_str = str(node_id) if node_id is not None else None

        if node_id_str:
            with DownloadManager._lock:
                DownloadManager.active_downloads[node_id_str] = cancel_event

        temp_path = None
        try:
            # early filename handling for resume
            resume_from = 0
            early_filename = None
            if filename is not None:
                early_filename = sanitize_filename(filename)
                potential_temp = os.path.join(save_path, early_filename + '.tmp')
                if os.path.exists(potential_temp):
                    resume_from = os.path.getsize(potential_temp)
                    print(f"▶ Resuming from {resume_from:,} bytes for {early_filename}")

            # build request headers (with Range for resume)
            req_headers = dict(headers) if headers else {}
            if resume_from > 0:
                req_headers['Range'] = f'bytes={resume_from}-'

            response = requests.get(url, stream=True, params=params, headers=req_headers)
            response.raise_for_status()

            # get final filename (if not provided)
            if filename is None:
                filename = DownloadManager._extract_filename(response, url)
            filename = sanitize_filename(filename)

            print(f"Downloading to: {os.path.join(save_path, filename)}")

            full_path = os.path.join(save_path, filename)
            temp_path = full_path + '.tmp'

            # determine total size (handles Range responses)
            if 'content-range' in response.headers and resume_from > 0:
                total_size = int(response.headers['content-range'].split('/')[-1])
            else:
                total_size = int(response.headers.get('content-length', 0))
                if resume_from > 0:
                    total_size += resume_from

            downloaded = resume_from
            open_mode = 'ab' if resume_from > 0 else 'wb'

            with open(temp_path, open_mode) as file:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename, initial=downloaded) as pbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if node_id_str and cancel_event.is_set():
                            raise DownloadCancelled("Download cancelled by user")

                        size = file.write(data)
                        downloaded += size
                        pbar.update(size)
                        pbar.refresh()

                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100.0
                            progress_callback.set_progress(progress)

            shutil.move(temp_path, full_path)
            print(f"✅ Download complete: {full_path}")
            return full_path

        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                if isinstance(e, DownloadCancelled):
                    os.remove(temp_path)
                    print(f"🗑 Cleaned up temp file on cancel: {temp_path}")
                else:
                    print(f"⏸ Kept partial temp file for resume: {temp_path} (error: {e})")
            print(f"❌ Download error: {str(e)}")
            raise
        finally:
            if node_id_str:
                with DownloadManager._lock:
                    if node_id_str in DownloadManager.active_downloads:
                        del DownloadManager.active_downloads[node_id_str]

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
