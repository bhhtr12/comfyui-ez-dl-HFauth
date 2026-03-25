from ..base_downloader import BaseModelDownloader, get_model_dirs
from ..download_utils import DownloadManager
from .hf_utils import parse_hf_url
from huggingface_hub import hf_hub_download

class HFDownloader(BaseModelDownloader):     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {       
                "repo_id": ("STRING", {"multiline": False, "default": "runwayml/stable-diffusion-v1-5"}),
                "filename": ("STRING", {"multiline": False, "default": "v1-5-pruned-emaonly.ckpt"}),
                "local_path": (get_model_dirs(),),
                
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": True}),
                "local_path_override": ("STRING", {"default": ""}),
                "hf_token": ("STRING", {          #token support
                    "default": "", 
                    "multiline": False, 
                    "password": True
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    FUNCTION = "download"

    def download(self, model_url, local_path, node_id, overwrite=False, local_path_override=""):
        repo_id, filename = parse_hf_url(model_url)
        
    def download(self, repo_id, filename, local_path, node_id, overwrite=False, local_path_override="", hf_token=""):
        if not repo_id or not filename:
            print(f"Invalid Hugging Face URL: {model_url}")
            print(f"Missing required values: repo_id='{repo_id}', filename='{filename}'")
            return {}

        final_path = local_path_override if local_path_override else local_path

        print(f'downloading model {repo_id} {filename} {final_path} {node_id} {overwrite}')
        self.node_id = node_id
        save_path = self.prepare_download_path(final_path, filename)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        
        return self.handle_download(
            DownloadManager.download_with_progress,
            save_path=save_path,
            filename=filename,
            overwrite=overwrite,
            url=url,
            progress_callback=self
        )

        # use official HF library when token is provided
        if hf_token and hf_token.strip():
            print("Using hf_hub_download with token (private repo support)")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=save_path,
                    token=hf_token.strip(),
                    force_download=overwrite
                )
                self.update_status("Complete!", 100)
                return {}
            except Exception as e:
                print(f"HF hub download error: {str(e)}")
                raise e
        else:
            # public-repo path (unchanged)
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            return self.handle_download(
                DownloadManager.download_with_progress,
                save_path=save_path,
                filename=filename,
                overwrite=overwrite,
                url=url,
                progress_callback=self
            )

