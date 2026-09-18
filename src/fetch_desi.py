#!/usr/bin/env python3
"""Acquire the raw DESI DR1 catalogues for the UNIONS x DESI IA analysis.

Reads the declarative manifest in config/data_sources.yaml, expands it into a flat
list of download targets, and fetches each one with curl (resume + retries). Already
present files of the correct size are skipped, so the script is safe to re-run and to
interrupt.

A provenance manifest (download_manifest.json) is written under <dest> recording, for
every target: source name, URL, local path, expected/actual byte size, and status.

Usage
-----
    python scripts/fetch_desi.py [--config CONFIG] [--source NAME [NAME ...]]
                                 [--workers N] [--dry-run] [--verify-sha256]

Examples
--------
    # Everything (DESI LSS + fastspec), as configured:
    python scripts/fetch_desi.py

    # Just preview what would be fetched and how big it is:
    python scripts/fetch_desi.py --dry-run

    # Only the LSS clustering catalogues:
    python scripts/fetch_desi.py --source desi_lss
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.request
from itertools import product
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Manifest expansion
# --------------------------------------------------------------------------- #
def expand_source(name: str, src: dict) -> list[dict]:
    """Turn one source block into a flat list of {name, filename, url, subdir} targets."""
    axes = {
        "tracer": src.get("tracers", []),
        "cap": src.get("caps", []),
        "i": src.get("random_indices", []),
        "program": src.get("programs", []),
        "hp": src.get("healpix", []),
    }
    targets: list[dict] = []
    seen: set[str] = set()
    for pat in src["patterns"]:
        per = pat["per"]
        value_lists = [axes[a] for a in per]
        for combo in product(*value_lists):
            subs = dict(zip(per, combo))
            fname = pat["template"].format(**subs)
            if fname in seen:        # full_data is per-tracer; dedupe across caps
                continue
            seen.add(fname)
            targets.append(
                {
                    "source": name,
                    "pattern": pat["name"],
                    "filename": fname,
                    "url": f"{src['base_url']}/{fname}",
                    "subdir": src["subdir"],
                }
            )
    return targets


# --------------------------------------------------------------------------- #
# Remote size resolution
# --------------------------------------------------------------------------- #
def sizes_via_propfind(base_url: str) -> dict[str, int]:
    """One WebDAV PROPFIND for the whole directory -> {filename: bytes}."""
    req = urllib.request.Request(
        base_url + "/", method="PROPFIND", headers={"Depth": "1"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode("utf-8", "replace")
    out: dict[str, int] = {}
    for resp in body.split("<D:response>"):
        n = re.search(r"<D:displayname>([^<]+)</D:displayname>", resp)
        s = re.search(r"<D:getcontentlength>(\d+)</D:getcontentlength>", resp)
        if n and s:
            out[n.group(1)] = int(s.group(1))
    return out


def size_via_head(url: str) -> int | None:
    """HTTP HEAD content-length, following redirects."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def resolve_sizes(name: str, src: dict, targets: list[dict]) -> None:
    """Annotate each target with an 'expected' byte size (or None)."""
    method = src.get("size_method")
    if method == "propfind":
        table = sizes_via_propfind(src["base_url"])
        for t in targets:
            t["expected"] = table.get(t["filename"])
    elif method == "head":
        for t in targets:
            t["expected"] = size_via_head(t["url"])
    else:
        for t in targets:
            t["expected"] = None


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_one(t: dict, dest_root: Path, retries: int) -> dict:
    """Fetch a single target with curl -C - (resume). Returns a status record."""
    out_dir = dest_root / t["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / t["filename"]
    expected = t.get("expected")

    if out_path.exists():
        actual = out_path.stat().st_size
        if expected is None or actual == expected:
            return {**_record(t, out_path), "actual": actual, "status": "skipped"}
        # partial/corrupt -> curl -C - will resume to completion

    cmd = [
        "curl", "-f", "-L", "-s", "-S", "-C", "-",
        "--retry", str(retries), "--retry-delay", "5",
        "-o", str(out_path), t["url"],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    actual = out_path.stat().st_size if out_path.exists() else 0

    if proc.returncode != 0:
        status = "error"
        err = proc.stderr.strip()[:300]
    elif expected is not None and actual != expected:
        status = "size_mismatch"
        err = f"expected {expected}, got {actual}"
    else:
        status = "ok"
        err = None
    rec = {**_record(t, out_path), "actual": actual, "status": status}
    if err:
        rec["error"] = err
    return rec


def _record(t: dict, out_path: Path) -> dict:
    return {
        "source": t["source"],
        "pattern": t["pattern"],
        "filename": t["filename"],
        "url": t["url"],
        "path": str(out_path),
        "expected": t.get("expected"),
    }


def human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "data_sources.yaml"))
    ap.add_argument("--source", nargs="+", help="Restrict to these source blocks.")
    ap.add_argument("--workers", type=int, help="Override parallel download workers.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List targets + sizes, download nothing.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dest_root = Path(cfg["dest"])
    workers = args.workers or cfg.get("workers", 4)
    retries = cfg.get("retries", 5)

    wanted = args.source or list(cfg["sources"])
    targets: list[dict] = []
    for name in wanted:
        src = cfg["sources"][name]
        ts = expand_source(name, src)
        print(f"[{name}] resolving sizes for {len(ts)} files ...", flush=True)
        resolve_sizes(name, src, ts)
        targets.extend(ts)

    total = sum(t["expected"] or 0 for t in targets)
    print(f"\n{len(targets)} files, total {human(total)} across: {', '.join(wanted)}\n")

    if args.dry_run:
        for t in targets:
            print(f"  {human(t['expected']):>9}  {t['subdir']}/{t['filename']}")
        return 0

    dest_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    done_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, t, dest_root, retries): t for t in targets}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            done_bytes += rec.get("actual") or 0
            flag = {"ok": "OK", "skipped": "--", "error": "!!",
                    "size_mismatch": "??"}.get(rec["status"], "??")
            print(f"[{i}/{len(targets)}] {flag} {rec['status']:13} "
                  f"{human(rec.get('actual')):>9}  {rec['filename']}", flush=True)

    # Provenance manifest next to the data.
    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "sources": wanted,
        "dest": str(dest_root),
        "n_files": len(results),
        "total_bytes": sum(r.get("actual") or 0 for r in results),
        "files": sorted(results, key=lambda r: (r["source"], r["filename"])),
    }
    mpath = dest_root / "download_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))

    bad = [r for r in results if r["status"] in ("error", "size_mismatch")]
    print(f"\nManifest -> {mpath}")
    n_ok = sum(r["status"] == "ok" for r in results)
    n_skip = sum(r["status"] == "skipped" for r in results)
    print(f"Done: {n_ok} downloaded, {n_skip} already present, {len(bad)} failed.")
    if bad:
        print("Failures (re-run to resume):")
        for r in bad:
            print(f"  {r['filename']}: {r['status']} {r.get('error','')}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
