import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


HF_BASE_URL = "https://huggingface.co"

WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".ggml",
    ".onnx",
    ".h5",
    ".msgpack",
    ".npy",
    ".npz",
    ".tflite",
)

METADATA_EXTENSIONS = (
    ".json",
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".jinja",
    ".model",
    ".spm",
    ".tiktoken",
    ".bpe",
    ".vocab",
    ".merges",
)

METADATA_FILENAMES = {
    ".gitattributes",
    "license",
    "license.txt",
    "notice",
    "notice.txt",
    "readme",
    "readme.md",
}


@dataclass
class RepoFile:
    path: str
    size: Optional[int] = None


@dataclass
class FetchResult:
    model_id: str
    revision: str
    snapshot_dir: Path
    downloaded_files: List[str] = field(default_factory=list)
    skipped_weight_files: List[str] = field(default_factory=list)
    skipped_large_files: List[str] = field(default_factory=list)
    repo_files: List[str] = field(default_factory=list)


class HuggingFaceError(RuntimeError):
    pass


def load_token(explicit_token: Optional[str] = None, token_file: Optional[str] = None) -> Optional[str]:
    if explicit_token:
        return explicit_token.strip()

    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token.strip()

    if token_file:
        path = Path(token_file)
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token

    return None


def is_weight_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(WEIGHT_EXTENSIONS)


def is_metadata_file(path: str) -> bool:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if is_weight_file(path):
        return False
    if name in METADATA_FILENAMES:
        return True
    return lower.endswith(METADATA_EXTENSIONS)


def safe_repo_dir(model_id: str) -> str:
    return model_id.replace("/", "--")


class HuggingFaceMetadataClient:
    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = ".llm_analyzer_cache",
        timeout_s: int = 60,
    ) -> None:
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.timeout_s = timeout_s

    def fetch_metadata(
        self,
        model_id: str,
        revision: str = "main",
        max_file_mb: float = 50.0,
    ) -> FetchResult:
        info = self.model_info(model_id, revision)
        resolved_revision = info.get("sha") or revision
        snapshot_dir = self.cache_dir / "models" / safe_repo_dir(model_id) / resolved_revision
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        (snapshot_dir / "model_info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        repo_files = self.list_repo_files(model_id, revision, info)
        max_bytes = int(max_file_mb * 1024 * 1024)
        result = FetchResult(
            model_id=model_id,
            revision=resolved_revision,
            snapshot_dir=snapshot_dir,
            repo_files=[repo_file.path for repo_file in repo_files],
        )

        for repo_file in repo_files:
            path = repo_file.path
            if is_weight_file(path):
                result.skipped_weight_files.append(path)
                continue
            if not is_metadata_file(path):
                continue
            if repo_file.size is not None and repo_file.size > max_bytes:
                result.skipped_large_files.append(path)
                continue
            self.download_file(model_id, revision, path, snapshot_dir / path)
            result.downloaded_files.append(path)

        return result

    def model_info(self, model_id: str, revision: str = "main") -> Dict[str, Any]:
        encoded_model = quote(model_id, safe="/")
        urls = [
            "%s/api/models/%s/revision/%s" % (HF_BASE_URL, encoded_model, quote(revision, safe="")),
            "%s/api/models/%s" % (HF_BASE_URL, encoded_model),
        ]
        last_error = None
        for url in urls:
            try:
                return self._get_json(url)
            except HuggingFaceError as exc:
                last_error = exc
        raise last_error or HuggingFaceError("Unable to fetch model info for %s" % model_id)

    def list_repo_files(
        self,
        model_id: str,
        revision: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> List[RepoFile]:
        info = info or self.model_info(model_id, revision)
        siblings = info.get("siblings") or []
        files = []
        for sibling in siblings:
            path = sibling.get("rfilename") or sibling.get("path")
            if path:
                files.append(RepoFile(path=path, size=sibling.get("size")))

        if files:
            return sorted(files, key=lambda item: item.path)

        return self._list_tree_files(model_id, revision)

    def download_file(self, model_id: str, revision: str, path: str, output_path: Path) -> None:
        encoded_model = quote(model_id, safe="/")
        encoded_path = quote(path, safe="/")
        encoded_revision = quote(revision, safe="")
        url = "%s/%s/resolve/%s/%s" % (HF_BASE_URL, encoded_model, encoded_revision, encoded_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        request = self._request(url)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                with output_path.open("wb") as out_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
        except HTTPError as exc:
            raise self._http_error("download %s" % path, exc)
        except URLError as exc:
            raise HuggingFaceError("Network error while downloading %s: %s" % (path, exc))

    def _list_tree_files(self, model_id: str, revision: str) -> List[RepoFile]:
        encoded_model = quote(model_id, safe="/")
        encoded_revision = quote(revision, safe="")
        url = "%s/api/models/%s/tree/%s?recursive=1&expand=1" % (
            HF_BASE_URL,
            encoded_model,
            encoded_revision,
        )
        tree = self._get_json(url)
        files = []
        for item in tree:
            if item.get("type") == "file" and item.get("path"):
                files.append(RepoFile(path=item["path"], size=item.get("size")))
        return sorted(files, key=lambda item: item.path)

    def _get_json(self, url: str) -> Dict[str, Any]:
        request = self._request(url)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            raise self._http_error("fetch %s" % url, exc)
        except URLError as exc:
            raise HuggingFaceError("Network error while fetching %s: %s" % (url, exc))

        try:
            return json.loads(payload)
        except ValueError as exc:
            raise HuggingFaceError("Invalid JSON response from %s: %s" % (url, exc))

    def _request(self, url: str) -> Request:
        headers = {"User-Agent": "llm-analyzer/0.1"}
        if self.token:
            headers["Authorization"] = "Bearer %s" % self.token
        return Request(url, headers=headers)

    def _http_error(self, action: str, exc: HTTPError) -> HuggingFaceError:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if exc.code in (401, 403):
            hint = "Access denied. If this is a gated/private model, pass --hf-token or keep .hf_token.txt in the repo root after accepting the model license."
        else:
            hint = "Hugging Face returned HTTP %s." % exc.code
        message = "%s failed: %s %s. %s" % (action, exc.code, exc.reason, hint)
        if body:
            message += " Response: %s" % body[:500]
        return HuggingFaceError(message)
