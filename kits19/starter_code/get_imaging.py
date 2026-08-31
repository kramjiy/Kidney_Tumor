"""
Faster parallel downloader for KITS19 imaging files (Windows / VSCode friendly).
Features:
- Uses a single requests.Session with retries and connection pooling.
- Downloads multiple files in parallel using ThreadPoolExecutor.
- Larger chunk size (64KB) to reduce Python/IO overhead.
- Robust temporary file handling and atomic move on completion.
- Adjustable concurrency (MAX_WORKERS).
- Safer and correct use of __file__ and Path resolution.

Usage: run this script from the project folder (same location as original script) with Python 3.8+.
"""

from pathlib import Path
from shutil import move
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import tempfile

from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====== CONFIG ======
IMAGING_URL = "https://kits19.sfo2.digitaloceanspaces.com/"
IMAGING_TMPL = "master_{:05d}.nii.gz"
TOTAL_CASES = 100
MAX_WORKERS = min(12, (os.cpu_count() or 4) * 2)  # tune this if you know your network
CHUNK_SIZE = 64 * 1024  # 64KB
SESSION_TIMEOUT = 30  # seconds for connect/read
RETRIES = 5
BACKOFF_FACTOR = 0.5

# Resolve project root reliably even when run from VSCode terminal
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_destination(i: int) -> Path:
    d = DATA_DIR / f"case_{i:05d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "imaging.nii.gz"


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=RETRIES,
                    backoff_factor=BACKOFF_FACTOR,
                    status_forcelist=(500, 502, 503, 504),
                    allowed_methods=("GET",))
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS + 2,
                          pool_maxsize=MAX_WORKERS + 2,
                          max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def download_one(session: requests.Session, cid: int, progress_position: int = 0) -> tuple:
    """Download a single case. Returns (cid, success, message)
    Uses a NamedTemporaryFile in the target folder to avoid cross-device move issues on Windows.
    """
    dest = get_destination(cid)
    if dest.exists():
        return (cid, True, "already exists")

    remote = IMAGING_TMPL.format(cid)
    url = IMAGING_URL + remote

    try:
        with session.get(url, stream=True, timeout=SESSION_TIMEOUT) as resp:
            resp.raise_for_status()
            total = resp.headers.get("content-length")
            if total is not None:
                total = int(total)

            # Create temp file in same directory to ensure atomic move works on Windows
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent), prefix=".tmp_", suffix=".nii.gz") as tmp:
                tmp_path = Path(tmp.name)
                if total:
                    with tqdm(total=total, unit="B", unit_scale=True, desc=f"case_{cid:05d}", position=progress_position) as pbar:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                tmp.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Unknown size
                    with tqdm(unit="B", unit_scale=True, desc=f"case_{cid:05d}", position=progress_position) as pbar:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                tmp.write(chunk)
                                pbar.update(len(chunk))

            # Move temp to destination (atomic on same filesystem)
            move(str(tmp_path), str(dest))
        return (cid, True, "downloaded")
    except Exception as e:
        # Cleanup partial tmp file if present
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return (cid, False, str(e))


def main():
    # Build list of missing cases
    left = [i for i in range(TOTAL_CASES) if not get_destination(i).exists()]
    if not left:
        print("All cases already downloaded.")
        return

    print(f"{len(left)} cases to download — using {MAX_WORKERS} workers")

    session = make_session()

    # We'll use per-worker progress bars; position argument keeps them from clobbering each other.
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        for pos, cid in enumerate(left):
            # limit number of concurrently displayed bars to MAX_WORKERS (positions 0..MAX_WORKERS-1)
            pos_mod = pos % MAX_WORKERS
            futures.append(exe.submit(download_one, session, cid, pos_mod))

        for fut in as_completed(futures):
            cid, ok, msg = fut.result()
            results.append((cid, ok, msg))

    # Summary
    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n{len(failed)} files failed to download:")
        for cid, ok, msg in failed:
            print(f"  case_{cid:05d}: {msg}")
    else:
        print("\nAll downloads completed successfully.")


if __name__ == "__main__":
    start = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user — exiting")
        sys.exit(1)
    finally:
        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.1f}s")
