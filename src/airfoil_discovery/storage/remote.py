from __future__ import annotations

import json
import shutil
from pathlib import Path

import requests

from airfoil_discovery.config import Settings


class RemoteArchiveClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.storage.provider
        self.manifest_dir = settings.paths.remote_manifest_dir

    def compress_folder(self, folder: Path) -> Path:
        archive_base = folder.with_suffix("")
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=folder)
        return Path(archive_path)

    def upload_archive(self, archive_path: Path, object_name: str) -> str:
        if self.provider == "supabase":
            return self._upload_supabase(archive_path, object_name)
        if self.provider == "firebase":
            return self._upload_firebase(archive_path, object_name)
        return self._upload_local_manifest(archive_path, object_name)

    def resolve_archive(self, object_name: str) -> str:
        if self.provider == "supabase":
            cfg = self.settings.storage
            return f"{cfg.supabase_url}/storage/v1/object/public/{cfg.supabase_bucket}/{object_name}.zip"
        if self.provider == "firebase":
            cfg = self.settings.storage
            return f"https://firebasestorage.googleapis.com/v0/b/{cfg.firebase_bucket}/o/{object_name}.zip"
        manifest = self.manifest_dir / f"{object_name}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Archive manifest not found for {object_name}")
        return json.loads(manifest.read_text(encoding='utf-8'))["url"]

    def _upload_local_manifest(self, archive_path: Path, object_name: str) -> str:
        dest = self.manifest_dir / f"{object_name}.zip"
        shutil.copy2(archive_path, dest)
        url = f"{self.settings.storage.local_manifest_base_url}/{dest.name}"
        manifest = self.manifest_dir / f"{object_name}.json"
        manifest.write_text(json.dumps({"url": url, "archive": str(dest)}, indent=2), encoding="utf-8")
        return url

    def _upload_supabase(self, archive_path: Path, object_name: str) -> str:
        cfg = self.settings.storage
        endpoint = f"{cfg.supabase_url}/storage/v1/object/{cfg.supabase_bucket}/{object_name}.zip"
        headers = {
            "apikey": cfg.supabase_api_key,
            "Authorization": f"Bearer {cfg.supabase_api_key}",
            "x-upsert": "true",
            "Content-Type": "application/zip",
        }
        response = requests.post(endpoint, headers=headers, data=archive_path.read_bytes(), timeout=120)
        response.raise_for_status()
        return f"{cfg.supabase_url}/storage/v1/object/public/{cfg.supabase_bucket}/{object_name}.zip"

    def _upload_firebase(self, archive_path: Path, object_name: str) -> str:
        cfg = self.settings.storage
        endpoint = (
            f"https://firebasestorage.googleapis.com/v0/b/{cfg.firebase_bucket}/o"
            f"?uploadType=media&name={object_name}.zip"
        )
        headers = {
            "Authorization": f"Bearer {cfg.firebase_bearer_token}",
            "Content-Type": "application/zip",
        }
        response = requests.post(endpoint, headers=headers, data=archive_path.read_bytes(), timeout=120)
        response.raise_for_status()
        payload = response.json()
        token = payload.get("downloadTokens", "")
        return f"https://firebasestorage.googleapis.com/v0/b/{cfg.firebase_bucket}/o/{object_name}.zip?alt=media&token={token}"
