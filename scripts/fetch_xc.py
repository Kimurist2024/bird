#!/usr/bin/env python3
"""
Polite Xeno-canto API v3 fetcher for BirdCLEF+ 2026.

Follows the etiquette codified in .claude/skills/birdclef-2026-rules/SKILL.md:
- API v3 only (v2 shut down), key required
- 1 req/sec hard rate limit (metadata + downloads share the budget)
- Sequential (concurrency = 1)
- Resume-safe: skips files already present locally or already in train.csv
- Per-session cap and per-species cap
- Exponential backoff on 429/5xx
- Full JSONL audit log

Usage:
    python scripts/fetch_xc.py --plan                  # show what would be fetched, no network
    python scripts/fetch_xc.py --metadata-only         # query API, no audio download
    python scripts/fetch_xc.py                         # actually download
    python scripts/fetch_xc.py --species 23150 --max 5 # narrow to one species

Key resolution order: $XC_API_KEY env var, then ~/.xenocanto_key file (mode 600).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "birdclef-2026"
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
TARGETS_CSV = KNOWLEDGE_DIR / "xc_targets.csv"
FETCH_LOG = KNOWLEDGE_DIR / "xc_fetch_log.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "external" / "xc_supplement"

API_URL = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "BirdCLEF2026-prep/0.1 (research; contact: 59061kimura@seiko.ac.jp)"
MIN_INTERVAL_S = 1.05
RETRY_BACKOFF_S = (5, 15, 45)
MAX_CONSECUTIVE_FAILURES = 5
PAGE_SIZE = 500  # v3 max per_page → 5x fewer requests vs default 100


def load_key() -> str:
    key = os.environ.get("XC_API_KEY", "").strip()
    if key:
        return key
    key_file = Path.home() / ".xenocanto_key"
    if key_file.exists():
        st = key_file.stat()
        if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            sys.stderr.write(
                f"refusing to read {key_file}: file is group/world accessible. "
                f"run: chmod 600 {key_file}\n"
            )
            sys.exit(2)
        return key_file.read_text().strip()
    sys.stderr.write(
        "no XC API key found. set $XC_API_KEY or write key to ~/.xenocanto_key (chmod 600)\n"
        "get a key at https://xeno-canto.org/account\n"
    )
    sys.exit(2)


class RateLimiter:
    def __init__(self, min_interval_s: float) -> None:
        self._min = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self._min:
            time.sleep(self._min - delta)
        self._last = time.monotonic()


@dataclass
class LogEntry:
    ts: str
    kind: str  # "query" | "download" | "skip" | "error"
    species: str
    detail: dict


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: LogEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def http_get(url: str, *, rate: RateLimiter) -> bytes:
    rate.wait()
    last_err: Exception | None = None
    for attempt, backoff in enumerate([0, *RETRY_BACKOFF_S]):
        if backoff:
            time.sleep(backoff + random.uniform(0, 0.5))
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json, audio/ogg, */*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < len(RETRY_BACKOFF_S):
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < len(RETRY_BACKOFF_S):
                continue
            raise
    assert last_err is not None
    raise last_err


def build_query(scientific_name: str) -> str:
    parts = scientific_name.strip().split()
    if len(parts) < 2:
        return f'sp:"{scientific_name}"'
    gen, sp = parts[0], parts[1]
    return f"gen:{gen} sp:{sp}"


def existing_xc_ids(train_csv: Path) -> set[str]:
    df = pd.read_csv(train_csv, usecols=["filename", "collection"])
    df = df[df["collection"] == "XC"]
    pat = re.compile(r"XC(\d+)\.")
    ids: set[str] = set()
    for fn in df["filename"]:
        m = pat.search(str(fn))
        if m:
            ids.add(m.group(1))
    return ids


def load_targets(species_filter: list[str] | None) -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        sys.stderr.write(
            f"missing {TARGETS_CSV}. regenerate via the snippet in birdclef-2026-rules.\n"
        )
        sys.exit(2)
    df = pd.read_csv(TARGETS_CSV, dtype={"primary_label": str})
    df = df[df["fetch_cap"] > 0].copy()
    if species_filter:
        df = df[df["primary_label"].isin(species_filter)]
    df = df.sort_values(["tier", "n_train", "primary_label"])
    return df


def fetch_metadata(
    query: str, key: str, *, rate: RateLimiter
) -> list[dict]:
    recs: list[dict] = []
    page = 1
    while True:
        url = (
            f"{API_URL}?"
            + urllib.parse.urlencode({
                "key": key, "query": query, "per_page": PAGE_SIZE, "page": page,
            })
        )
        body = http_get(url, rate=rate)
        data = json.loads(body)
        recs.extend(data.get("recordings") or [])
        n_pages = int(data.get("numPages") or 1)
        if page >= n_pages:
            break
        page += 1
    return recs


def normalize_url(u: str) -> str:
    """v3 returns protocol-relative URLs like //xeno-canto.org/123/download."""
    if u.startswith("//"):
        return "https:" + u
    return u


def is_restricted(rec: dict) -> bool:
    """Restricted species have redacted fields and no playable file."""
    meta = rec.get("_meta") or {}
    return bool(meta.get("redacted_fields")) or not rec.get("file")


def download_one(
    rec: dict, out_path: Path, *, rate: RateLimiter
) -> int:
    url = rec.get("file")
    if not url:
        raise ValueError(f"recording {rec.get('id')} has no 'file' (likely restricted species)")
    url = normalize_url(url)
    body = http_get(url, rate=rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)
    return len(body)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show planned work, no network calls")
    ap.add_argument("--metadata-only", action="store_true", help="hit API but skip audio downloads")
    ap.add_argument("--species", nargs="*", default=None, help="restrict to these primary_label values")
    ap.add_argument("--max", type=int, default=200, help="session cap on total downloads (default 200)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="audio output root")
    ap.add_argument("--min-quality", choices=["A", "B", "C", "D", "E"], default=None, help="filter q field")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_targets(args.species)
    if targets.empty:
        print("no target species (xc_targets.csv has 0 rows with fetch_cap>0 matching filter)")
        return 0

    print(f"targets: {len(targets)} species, est. cap sum = {int(targets['fetch_cap'].sum())}")
    if args.plan:
        cols = ["primary_label", "scientific_name", "class_name", "tier", "n_train", "fetch_cap"]
        print(targets[cols].to_string(index=False))
        return 0

    key = load_key()
    rate = RateLimiter(MIN_INTERVAL_S)
    logger = JsonlLogger(FETCH_LOG)
    train_csv = DATA_ROOT / "train.csv"
    skip_ids = existing_xc_ids(train_csv) if train_csv.exists() else set()
    print(f"loaded {len(skip_ids)} existing XC ids from train.csv to skip")

    session_total = 0
    consecutive_failures = 0

    for _, row in targets.iterrows():
        if session_total >= args.max:
            print(f"session cap {args.max} reached, stopping")
            break
        species = str(row["primary_label"])
        scientific = str(row["scientific_name"])
        cap = int(row["fetch_cap"])
        query = build_query(scientific)
        species_dir = args.outdir / species

        existing_local = {
            re.match(r"XC(\d+)\.", p.name).group(1)
            for p in species_dir.glob("XC*.ogg")
            if re.match(r"XC(\d+)\.", p.name)
        } if species_dir.exists() else set()

        print(f"\n== {species} ({scientific}) cap={cap} local={len(existing_local)} ==")
        try:
            recs = fetch_metadata(query, key, rate=rate)
        except Exception as e:
            consecutive_failures += 1
            logger.write(LogEntry(now_iso(), "error", species, {"phase": "query", "err": repr(e)}))
            print(f"  query failed: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"!! {MAX_CONSECUTIVE_FAILURES} consecutive failures, aborting")
                return 1
            continue
        consecutive_failures = 0
        logger.write(LogEntry(now_iso(), "query", species, {
            "query": query, "n_results": len(recs),
        }))

        candidates: list[dict] = []
        n_restricted = 0
        for rec in recs:
            rid = str(rec.get("id") or "").strip()
            if not rid:
                continue
            if rid in skip_ids or rid in existing_local:
                continue
            if is_restricted(rec):
                n_restricted += 1
                continue
            if args.min_quality:
                q = (rec.get("q") or "").upper()
                if q and q > args.min_quality:
                    continue
            candidates.append(rec)
            if len(candidates) >= cap:
                break

        msg_restricted = f", {n_restricted} restricted" if n_restricted else ""
        print(f"  XC reports {len(recs)} recordings, {len(candidates)} new after dedupe{msg_restricted}")
        if args.metadata_only:
            logger.write(LogEntry(now_iso(), "skip", species, {
                "reason": "metadata-only",
                "would_download": len(candidates),
                "n_restricted": n_restricted,
                "n_total": len(recs),
            }))
            continue

        for rec in candidates:
            if session_total >= args.max:
                break
            rid = str(rec["id"])
            out_path = species_dir / f"XC{rid}.ogg"
            try:
                n_bytes = download_one(rec, out_path, rate=rate)
            except Exception as e:
                consecutive_failures += 1
                logger.write(LogEntry(now_iso(), "error", species, {
                    "phase": "download", "id": rid, "err": repr(e),
                }))
                print(f"  XC{rid}: download failed: {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"!! {MAX_CONSECUTIVE_FAILURES} consecutive failures, aborting")
                    return 1
                continue
            consecutive_failures = 0
            session_total += 1
            logger.write(LogEntry(now_iso(), "download", species, {
                "id": rid, "bytes": n_bytes, "q": rec.get("q"), "length": rec.get("length"),
                "cnt": rec.get("cnt"), "lat": rec.get("lat"), "lon": rec.get("lon"),
            }))
            print(f"  XC{rid} -> {out_path.relative_to(REPO_ROOT)} ({n_bytes/1024:.0f} KB)")

    print(f"\nDONE. downloaded={session_total} log={FETCH_LOG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
