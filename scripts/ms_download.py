# -*- coding: utf-8 -*-
"""ModelScope 多线程分块下载器 — 强制直连（禁代理），8 线程 Range 分块，sha256 落盘校验。

用法: python ms_download.py <repo_id> [<repo_id> ...]
产物: H:\\SHARE\\arch-transfer\\models\\<repo 末段>\\
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

# ── 强制直连：清代理 env + 空代理 opener（双保险，用户纪律 2026-10-31）──
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

FILES_API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
DL_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={fp}"
UA = {"User-Agent": "Mozilla/5.0 (arch-transfer research downloader)"}
CHUNK = 64 << 20          # 64MB 分块
WORKERS = 8               # 8 线程（记忆打法实测 20-25MB/s）
DEST_ROOT = r"H:\SHARE\latex-tts\models"


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def list_files(repo):
    req = urllib.request.Request(FILES_API.format(repo=repo), headers=UA)
    with opener().open(req, timeout=30) as r:
        d = json.load(r)
    files = [
        (f["Path"], int(f.get("Size", 0)), f.get("Sha256") or None)
        for f in d["Data"]["Files"]
        if f.get("Type") == "blob" and int(f.get("Size", 0)) > 0
    ]
    if not files:
        raise RuntimeError(f"{repo}: files API returned no blobs")
    return files


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def download_file(repo, path, size, sha256, dest_dir):
    dest = os.path.join(dest_dir, path.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest) or dest_dir, exist_ok=True)
    t0 = time.time()

    if os.path.exists(dest) and os.path.getsize(dest) == size:
        if sha256 and file_sha256(dest) == sha256:
            print(f"SKIP(existing+sha-ok) {path}", flush=True)
            return
        print(f"REDO(sha-mismatch-or-missing) {path}", flush=True)

    url = DL_API.format(repo=repo, fp=urllib.parse.quote_plus(path))

    # Range 支持探测：必须 206，否则退化单流
    probe = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-1023"})
    with opener().open(probe, timeout=60) as r:
        if r.status != 206:
            print(f"WARN {path}: no Range support (HTTP {r.status}), single-stream fallback", flush=True)
            ranges = [(0, size - 1)]
        else:
            ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]

    if not (os.path.exists(dest) and os.path.getsize(dest) == size):
        with open(dest, "wb") as f:
            f.truncate(size)

    lock = threading.Lock()
    progress = {"n": 0, "bytes": 0}
    total_chunks = len(ranges)

    def work(ab):
        s, e = ab
        want = e - s + 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={s}-{e}"})
                with opener().open(req, timeout=180) as r, open(dest, "r+b") as f:
                    f.seek(s)
                    got = 0
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                if got != want:
                    raise IOError(f"short read {got}/{want}")
                with lock:
                    progress["n"] += 1
                    progress["bytes"] += want
                    if progress["n"] % 8 == 0 or progress["n"] == total_chunks:
                        el = time.time() - t0
                        mb = progress["bytes"] / 1e6
                        print(f"  {path}: chunk {progress['n']}/{total_chunks} {mb/el:.1f}MB/s", flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, ranges))

    if sha256:
        got = file_sha256(dest)
        if got != sha256:
            os.remove(dest)
            raise RuntimeError(f"sha256 mismatch {path}: {got[:12]} != {sha256[:12]}")
    el = time.time() - t0
    print(f"DONE {path} {size/1e9:.2f}GB in {el:.0f}s ({size/1e6/el:.1f}MB/s) sha-ok={bool(sha256)}", flush=True)


def main():
    repos = sys.argv[1:]
    if not repos:
        print(__doc__)
        sys.exit(2)
    for repo in repos:
        name = repo.split("/")[-1]
        dest_dir = os.path.join(DEST_ROOT, name)
        print(f"== {repo} -> {dest_dir}", flush=True)
        files = list_files(repo)
        total = sum(s for _, s, _ in files)
        print(f"   {len(files)} files, {total/1e9:.1f}GB", flush=True)
        for path, size, sha in files:
            download_file(repo, path, size, sha, dest_dir)
    print("ALL-DONE", flush=True)


if __name__ == "__main__":
    main()
