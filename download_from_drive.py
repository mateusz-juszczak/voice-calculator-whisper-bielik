import io
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import DATA_SET_ID

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# === ŚCIEŻKI BEZWZGLĘDNE WZGLĘDEM PLIKU .PY ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")


def get_drive():
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
        creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def download_data(folder_id, target="downloaded"):
    service = get_drive()

    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name, mimeType, size)",
        pageSize=1000
    ).execute()

    files = results.get('files', [])
    downloaded = 0
    skipped = 0

    target_path = os.path.join(BASE_DIR, target)
    os.makedirs(target_path, exist_ok=True)

    print(f"Sprawdzam {len(files)} plików w chmurze...")

    for f in files:
        if f.get('mimeType') == 'application/vnd.google-apps.folder':
            continue

        file_path = os.path.join(target_path, f['name'])

        if os.path.exists(file_path):
            skipped += 1
            continue

        try:
            request = service.files().get_media(fileId=f['id'])
            with io.FileIO(file_path, 'wb') as file:
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

            downloaded += 1
            print(".", end="", flush=True)

        except Exception as e:
            print(f"\nBłąd pobierania {f['name']}: {e}")

    print(f"\nZakończono. Pobrano: {downloaded}, Pominięto (już istniejące): {skipped}")
    return target_path


def sort_files(files):
    recordings_data = [f for f in files if f.lower().endswith('.wav')]
    reference_data = [f for f in files if f.lower().endswith('.csv')]
    return recordings_data, reference_data


# === URUCHOMIENIE ===
eval_path = download_data(DATA_SET_ID, target="eval")
files = os.listdir(eval_path)
wav_files, csv_files = sort_files(files)

print(f"WAV: {len(wav_files)} | CSV: {len(csv_files)}")
