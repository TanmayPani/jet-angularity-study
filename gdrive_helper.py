import os
import io
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SECRETS_FILE = "./client_secret_794630652292-stiu72nu8ctbhg1ljg5att09uc7k7gpt.apps.googleusercontent.com.json"
TOKEN_FILE = "token.json"


def get_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


class GDrivePath:
    def __init__(self, folder_id, path_str=""):
        self.id = folder_id
        self.path_str = path_str or f"gdrive://{folder_id}"
        self._service = get_service()

    def __truediv__(self, child_name):
        child_name = str(child_name)
        response = self._service.files().list(
            q=f"'{self.id}' in parents and name='{child_name}' and trashed=false",
            spaces='drive',
            fields="files(id, name, mimeType)"
        ).execute()
        files = response.get('files', [])
        if not files:
            raise FileNotFoundError(f"GDrivePath: Could not find '{child_name}' inside '{self.path_str}'")
        return GDrivePath(files[0]['id'], path_str=f"{self.path_str}/{child_name}")

    def __str__(self):
        return self.id

    def glob(self, pattern):
        parts = pattern.split("/")
        current_level_folders = [self]
        
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            next_level = []
            for folder in current_level_folders:
                page_token = None
                while True:
                    response = folder._service.files().list(
                        q=f"'{folder.id}' in parents and trashed=false",
                        spaces='drive',
                        fields="nextPageToken, files(id, name, mimeType)",
                        pageToken=page_token,
                        pageSize=1000
                    ).execute()
                    
                    for f in response.get("files", []):
                        if fnmatch(f["name"], part):
                            if is_last:
                                next_level.append({"id": f["id"], "name": f["name"]})
                            else:
                                if f["mimeType"] == "application/vnd.google-apps.folder":
                                    next_level.append(GDrivePath(f["id"], path_str=f"{folder.path_str}/{f['name']}"))
                    
                    page_token = response.get("nextPageToken", None)
                    if page_token is None:
                        break
            current_level_folders = next_level
            
        return current_level_folders

    def rglob(self, pattern):
        return self.glob("**/" + pattern)


def find_folder_id(name_query):
    """Search Google Drive globally for a folder matching the given name."""
    service = get_service()
    response = service.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and name='{name_query}' and trashed=false",
        spaces='drive',
        fields="files(id, name, parents)"
    ).execute()
    
    files = response.get('files', [])
    if not files:
        print(f"No folder found matching '{name_query}'")
    for f in files:
        print(f"Found Folder: {f['name']} (ID: {f['id']})")
    return files

def list_gdrive_folder(folder_id, glob_expr="*.tree.root"):
    """Fetch file IDs and names from a Google Drive folder matching a glob expression with nested folders."""
    if isinstance(folder_id, GDrivePath):
        gpath = folder_id
    else:
        gpath = GDrivePath(folder_id)
    return gpath.glob(glob_expr)


def _download_file(service, file_id, out_path):
    """Downloads a single file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    with open(out_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
    return out_path


class GdriveBatchPrefetcher:
    """
    Downloads batches of files concurrently into a local cache directory.
    Yields a list of local file paths for each batch.
    Uses a bounded queue so that batch N+1 is downloaded while batch N is being consumed.
    """

    def __init__(self, gdrive_files, batch_size, cache_dir="./.gdrive_cache", max_workers=5):
        self.gdrive_files = gdrive_files
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.max_workers = max_workers
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._q = queue.Queue(maxsize=1)
        self._thread = None
        self._service = None

    def _prefetch_loop(self):
        try:
            for start_idx in range(0, len(self.gdrive_files), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch = self.gdrive_files[start_idx:end_idx]
                
                local_paths = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    for f in batch:
                        local_path = self.cache_dir / f["name"]
                        futures[executor.submit(_download_file, self._service, f["id"], local_path)] = (local_path, f.get("tree", ""))
                        
                    for future in as_completed(futures):
                        # will raise exception if download failed
                        future.result()
                        local_path, tree_name = futures[future]
                        path_str = str(local_path)
                        if tree_name:
                            path_str += f":{tree_name}"
                        local_paths.append(path_str)
                
                self._q.put(local_paths)
            self._q.put(None)  # Sentinel for end of iteration
        except Exception as e:
            self._q.put(e)

    def __iter__(self):
        self._service = get_service()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        
        while True:
            item = self._q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "find":
        if len(sys.argv) < 3:
            print("Usage: python gdrive_helper.py find <folder_name>")
            sys.exit(1)
        find_folder_id(sys.argv[2])
    else:
        print("Testing Google Drive Authentication...")
        service = get_service()
        print("Authentication successful!")
        print("\nTip: You can search for a folder ID by running:")
        print("uv run gdrive_helper.py find \"folder_name\"")
