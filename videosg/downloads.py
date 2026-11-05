"""Resumable byte-range downloads for large public research assets."""

from pathlib import Path
import time

import requests


def download_ranges(url, destination, chunk_size=8 * 1024 * 1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30) as probe:
        probe.raise_for_status()
        if probe.status_code != 206 or "Content-Range" not in probe.headers:
            raise RuntimeError(f"Server does not support byte ranges: {url}")
        total = int(probe.headers["Content-Range"].split("/")[-1])
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > total:
        raise RuntimeError(f"Partial download exceeds source size: {temporary}")
    while offset < total:
        end = min(total - 1, offset + chunk_size - 1)
        for attempt in range(3):
            try:
                with requests.get(url, headers={"Range": f"bytes={offset}-{end}"}, timeout=90) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{total}":
                        raise RuntimeError(f"Server did not honor byte range for {url}")
                    data = response.content
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if len(data) != end - offset + 1:
            raise RuntimeError(f"Short download at byte {offset}")
        with temporary.open("ab") as output:
            output.write(data)
        offset = end + 1
        print(f"{destination.name}: {offset}/{total} bytes", flush=True)
    temporary.replace(destination)
    return destination
