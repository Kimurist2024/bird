#!/usr/bin/env python3
"""
Polite iNaturalist API fetcher for BirdCLEF+ 2026.

Follows the etiquette codified in .claude/skills/birdclef-2026-rules/SKILL.md:
- 1.5s/request hard rate (tighter than iNat's 60 req/min recommendation —
  empirically iNat throttles around 1 req/s in bursts).
- Sequential (concurrency = 1).
- Optional Bearer token from ~/.inat_token (auth not required for reads;
  iNat JWT tokens expire ~24h).
- Resume-safe: skips iNat sound IDs already in train.csv or local mirror.
- 429 backoff: 30s, 120s, 300s; 3 consecutive 429 → abort.
- Skips insect sonotype rows (47158sonNN — all map to taxon_id=47158 Cicadidae,
  which is too coarse to be useful as labeled audio).
- Logs license + attribution for every download (cc-by, cc-by-nc, etc.).

Usage:
    python scripts/fetch_inat.py --plan
    python scripts/fetch_inat.py --metadata-only
    python scripts/fetch_inat.py --max 200
    python scripts/fetch_inat.py --species 23150 --max 5
    python scripts/fetch_inat.py --quality research,needs_id  # default: research,needs_id
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
FETCH_LOG = KNOWLEDGE_DIR / "inat_fetch_log.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "external" / "inat_supplement"

API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "BirdCLEF2026-prep/0.1 (research; contact: 59061kimura@seiko.ac.jp)"
MIN_INTERVAL_S = 1.5
RETRY_BACKOFF_S = (30, 120, 300)
MAX_CONSECUTIVE_FAILURES = 3
PAGE_SIZE = 200  # iNat max
SONOTYPE_PREFIX = "47158son"


def load_token() -> str | None:
    """Token is optional — read endpoints work without it."""
    env = os.environ.get("INAT_TOKEN", "").strip()
    if env:
        return env
    p = Path.home() / ".inat_token"
    if not p.exists():
        return None
    st = p.stat()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.stderr.write(
            f"refusing to read {p}: group/world accessible. run: chmod 600 {p}\n"
        )
        return None
    return p.read_text().strip() or None


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
    kind: str
    species: str
    detail: dict


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: LogEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def http_get(url: str, *, rate: RateLimiter, token: str | None, accept: str) -> bytes:
    rate.wait()
    last_err: Exception | None = None
    for attempt, backoff in enumerate([0, *RETRY_BACKOFF_S]):
        if backoff:
            time.sleep(backoff + random.uniform(0, 1.0))
        headers = {"User-Agent": USER_AGENT, "Accept": accept}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < len(RETRY_BACKOFF_S):
                sys.stderr.write(f"  [retry] HTTP {e.code}; sleeping before attempt {attempt+2}\n")
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < len(RETRY_BACKOFF_S):
                continue
            raise
    assert last_err is not None
    raise last_err


def existing_inat_sound_ids(train_csv: Path) -> set[str]:
    """train.csv stores iNat filenames like 'iNat1114648.ogg' — those are sound IDs."""
    if not train_csv.exists():
        return set()
    df = pd.read_csv(train_csv, usecols=["filename", "collection"])
    df = df[df["collection"] == "iNat"]
    pat = re.compile(r"iNat(\d+)\.")
    ids: set[str] = set()
    for fn in df["filename"]:
        m = pat.search(str(fn))
        if m:
            ids.add(m.group(1))
    return ids


def load_targets(species_filter: list[str] | None) -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        sys.stderr.write(f"missing {TARGETS_CSV}\n")
        sys.exit(2)
    df = pd.read_csv(TARGETS_CSV, dtype={"primary_label": str})
    df = df[df["fetch_cap"] > 0].copy()  # excludes tier4 + sonotypes
    df = df[~df["primary_label"].str.startswith(SONOTYPE_PREFIX, na=False)]
    df = df.dropna(subset=["inat_taxon_id"])
    df["inat_taxon_id"] = df["inat_taxon_id"].astype(int)
    if species_filter:
        df = df[df["primary_label"].isin(species_filter)]
    return df.sort_values(["tier", "n_train", "primary_label"])


def fetch_metadata(taxon_id: int, *, rate: RateLimiter, token: str | None, quality: str) -> list[dict]:
    """Walk pages, return observation list (each with .sounds[])."""
    obs: list[dict] = []
    page = 1
    while True:
        url = (
            f"{API_URL}?"
            + urllib.parse.urlencode({
                "taxon_id": taxon_id, "sounds": "true",
                "quality_grade": quality,
                "per_page": PAGE_SIZE, "page": page,
                "order_by": "created_at", "order": "desc",
            })
        )
        body = http_get(url, rate=rate, token=token, accept="application/json")
        data = json.loads(body)
        obs.extend(data.get("results") or [])
        total = int(data.get("total_results") or 0)
        if len(obs) >= total or not data.get("results"):
            break
        page += 1
        if page > 50:  # absolute safety: 50 pages × 200 = 10K observations
            break
    return obs


def download_sound(file_url: str, out_path: Path, *, rate: RateLimiter, token: str | None) -> int:
    body = http_get(file_url, rate=rate, token=token, accept="audio/*, */*")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)
    return len(body)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show planned targets, no network")
    ap.add_argument("--metadata-only", action="store_true", help="hit API, skip audio downloads")
    ap.add_argument("--species", nargs="*", default=None, help="restrict to these primary_label values")
    ap.add_argument("--max", type=int, default=300, help="session cap on total downloads (default 300)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="audio output root")
    ap.add_argument(
        "--quality", default="research,needs_id",
        help="iNat quality_grade filter (default: research,needs_id; use 'research' for stricter)",
    )
    ap.add_argument(
        "--exclude-cc-nd", action="store_true",
        help="skip recordings under no-derivatives licenses (cc-by-nd, cc-by-nc-nd)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_targets(args.species)
    if targets.empty:
        print("no target species after filtering")
        return 0

    print(f"targets: {len(targets)} species, est. cap sum = {int(targets['fetch_cap'].sum())}")
    if args.plan:
        cols = ["primary_label", "scientific_name", "class_name", "tier", "n_train", "fetch_cap", "inat_taxon_id"]
        print(targets[cols].to_string(index=False))
        return 0

    token = load_token()
    print(f"auth: {'token' if token else 'unauthenticated'}")
    rate = RateLimiter(MIN_INTERVAL_S)
    logger = JsonlLogger(FETCH_LOG)
    skip_ids = existing_inat_sound_ids(DATA_ROOT / "train.csv")
    print(f"loaded {len(skip_ids)} existing iNat sound IDs from train.csv to skip")

    session_total = 0
    consecutive_failures = 0

    for _, row in targets.iterrows():
        if session_total >= args.max:
            print(f"session cap {args.max} reached, stopping")
            break
        species = str(row["primary_label"])
        scientific = str(row["scientific_name"])
        taxon = int(row["inat_taxon_id"])
        cap = int(row["fetch_cap"])
        species_dir = args.outdir / species

        existing_local = set()
        if species_dir.exists():
            for p in species_dir.glob("iNat*.ogg"):
                m = re.match(r"iNat(\d+)\.", p.name)
                if m:
                    existing_local.add(m.group(1))
            for p in species_dir.glob("iNat*.mp3"):
                m = re.match(r"iNat(\d+)\.", p.name)
                if m:
                    existing_local.add(m.group(1))
            for p in species_dir.glob("iNat*.m4a"):
                m = re.match(r"iNat(\d+)\.", p.name)
                if m:
                    existing_local.add(m.group(1))
            for p in species_dir.glob("iNat*.wav"):
                m = re.match(r"iNat(\d+)\.", p.name)
                if m:
                    existing_local.add(m.group(1))

        print(f"\n== {species} ({scientific}) taxon={taxon} cap={cap} local={len(existing_local)} ==")
        try:
            obs_list = fetch_metadata(taxon, rate=rate, token=token, quality=args.quality)
        except Exception as e:
            consecutive_failures += 1
            logger.write(LogEntry(now_iso(), "error", species, {"phase": "query", "err": repr(e)}))
            print(f"  query failed: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"!! {MAX_CONSECUTIVE_FAILURES} consecutive failures, aborting")
                return 1
            continue
        consecutive_failures = 0

        # Flatten observations → (obs_id, sound_dict) candidates
        candidates: list[tuple[str, dict]] = []
        n_total_sounds = 0
        for obs in obs_list:
            obs_id = str(obs.get("id") or "")
            for s in (obs.get("sounds") or []):
                n_total_sounds += 1
                sid = str(s.get("id") or "")
                if not sid:
                    continue
                if sid in skip_ids or sid in existing_local:
                    continue
                if args.exclude_cc_nd and (s.get("license_code") or "").endswith("nd"):
                    continue
                if not s.get("file_url"):
                    continue
                candidates.append((obs_id, s))
                if len(candidates) >= cap:
                    break
            if len(candidates) >= cap:
                break

        logger.write(LogEntry(now_iso(), "query", species, {
            "taxon_id": taxon, "quality": args.quality,
            "n_observations": len(obs_list), "n_total_sounds": n_total_sounds,
            "n_candidates": len(candidates),
        }))
        print(f"  iNat: {len(obs_list)} obs, {n_total_sounds} sounds, {len(candidates)} new after dedupe")
        if args.metadata_only:
            continue

        for obs_id, s in candidates:
            if session_total >= args.max:
                break
            sid = str(s["id"])
            url = s["file_url"]
            ext = url.split("?", 1)[0].rsplit(".", 1)[-1].lower()
            if ext not in {"mp3", "wav", "m4a", "ogg", "flac", "aac"}:
                ext = "bin"
            out_path = species_dir / f"iNat{sid}.{ext}"
            try:
                n_bytes = download_sound(url, out_path, rate=rate, token=token)
            except Exception as e:
                consecutive_failures += 1
                logger.write(LogEntry(now_iso(), "error", species, {
                    "phase": "download", "sound_id": sid, "obs_id": obs_id, "err": repr(e),
                }))
                print(f"  iNat{sid}: download failed: {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"!! {MAX_CONSECUTIVE_FAILURES} consecutive failures, aborting")
                    return 1
                continue
            consecutive_failures = 0
            session_total += 1
            logger.write(LogEntry(now_iso(), "download", species, {
                "sound_id": sid, "obs_id": obs_id, "bytes": n_bytes,
                "license": s.get("license_code"),
                "attribution": s.get("attribution"),
                "ext": ext,
            }))
            print(f"  iNat{sid} -> {out_path.relative_to(REPO_ROOT)} ({n_bytes/1024:.0f} KB, {s.get('license_code')})")

    print(f"\nDONE. downloaded={session_total} log={FETCH_LOG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
