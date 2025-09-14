# wsip.py
# Python 3.x | Windows | GOG Galaxy 2.0
# - Full library (GOG + Steam + Epic)
# - Filters + fun selection modes
# - HowLongToBeat estimates (with cache) and aggressive normalization
# - Robust extraction of genres/tags/rating/year/playtime from multiple DB shapes

import os
import re
import sys
import json
import time
import random
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Iterable

# =========================
# Configuration
# =========================

DB_PATHS = [
    r"C:\ProgramData\GOG.com\Galaxy\storage\galaxy-2.0.db",
]
GALAXY_EXE_CANDIDATES = [
    r"C:\Program Files (x86)\GOG Galaxy\GalaxyClient.exe",
    r"C:\Program Files\GOG Galaxy\GalaxyClient.exe",
]
APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "RandomGalaxy"
APP_DIR.mkdir(parents=True, exist_ok=True)
HLTB_CACHE_PATH = APP_DIR / "hltb_cache.json"

# =========================
# Utilities (JSON, parsing)
# =========================

def _looks_like_json(value: str) -> bool:
    """Heuristically decides if a string may be JSON."""
    s = (value or "").strip()
    return s.startswith("{") or s.startswith("[")

def _json_flat_values(value: str) -> List[str]:
    """Flatten any JSON (dict/list) to a list of stringified terminal values."""
    results: List[str] = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            results.append(str(node))

    try:
        obj = json.loads(value)
        walk(obj)
    except Exception:
        pass
    return results

def _json_collect_lists_by_keys(value: str, keys: Iterable[str]) -> List[str]:
    """
    From JSON, collect strings found under any of the provided keys.
    Handles list-of-dicts with 'name' too.
    """
    results: List[str] = []
    if not _looks_like_json(value or ""):
        return results

    try:
        obj = json.loads(value)
    except Exception:
        return results

    keyset = {k.lower() for k in keys}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                lk = str(k).lower()
                if lk in keyset:
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, (str, int, float)):
                                results.append(str(item))
                            elif isinstance(item, dict):
                                name = item.get("name")
                                if isinstance(name, (str, int, float)):
                                    results.append(str(name))
                    elif isinstance(v, (str, int, float)):
                        results.append(str(v))
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return results

def _json_collect_numbers_by_keys(value: str, keys_regex: str = r"(score|rating)") -> List[float]:
    """
    From JSON, collect numeric values where the key matches keys_regex (case-insensitive).
    """
    numbers: List[float] = []
    if not _looks_like_json(value or ""):
        return numbers

    try:
        obj = json.loads(value)
    except Exception:
        return numbers

    rx = re.compile(keys_regex, re.IGNORECASE)

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if rx.search(str(k)):
                    n = _extract_any_number(v)
                    if n is not None:
                        numbers.append(float(n))
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return numbers

def _extract_title(raw: str) -> Optional[str]:
    """Extract a reasonable title from raw string or JSON."""
    if not raw:
        return None
    s = str(raw).strip()
    if _looks_like_json(s):
        try:
            j = json.loads(s)
            for key in ("title", "name"):
                if key in j and isinstance(j[key], (str, int, float)):
                    return str(j[key])
        except Exception:
            pass
    # unquote if it's a simple quoted string
    return re.sub(r'^\s*"{0,1}(.*?)"{0,1}\s*$', r'\1', s)

def _extract_string_list(pieces: List[str]) -> List[str]:
    """
    Merge & normalize genre/tag-like data from various CSV/JSON pieces.
    Deduplicate case-insensitively, drop short/noisy tokens.
    """
    values: List[str] = []
    for raw in pieces:
        if not raw:
            continue
        s = str(raw).strip()
        if _looks_like_json(s):
            values += _json_collect_lists_by_keys(s, ["genres", "tags", "categories", "features", "values"])
            values += _json_flat_values(s)
        else:
            values += [x.strip() for x in s.split(",") if x.strip()]

    seen, out = set(), []
    for v in values:
        norm = re.sub(r'\s+', ' ', str(v)).strip().strip('"').strip("'")
        if not norm or norm.isdigit() or len(norm) < 2:
            continue
        key = norm.lower()
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out

def _extract_any_number(val) -> Optional[float]:
    """Extract the first number found in val (number or embedded in a string)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r'[-+]?\d*\.?\d+', str(val))
    return float(m.group(0)) if m else None

def _extract_year(val) -> Optional[int]:
    """
    Try multiple representations:
    - int year
    - epoch seconds/milliseconds
    - JSON fields with date/time/timestamp
    - 4-digit substring fallback
    """
    if val is None:
        return None

    def year_from_epoch(seconds: float) -> Optional[int]:
        try:
            y = datetime.fromtimestamp(seconds, tz=timezone.utc).year
            return y if 1970 <= y <= 2100 else None
        except Exception:
            return None

    if isinstance(val, (int, float)):
        n = float(val)
        if n > 10_000_000_000:            # milliseconds epoch
            return year_from_epoch(n / 1000.0)
        if n > 10_000:                     # seconds epoch
            return year_from_epoch(n)
        y = int(n)                         # maybe it's already a year
        return y if 1970 <= y <= 2100 else None

    s = str(val)
    if _looks_like_json(s):
        try:
            j = json.loads(s)
            for key in ("date", "releaseDate", "release_date", "value", "released", "time", "timestamp"):
                if isinstance(j, dict) and key in j:
                    y = _extract_year(j[key])
                    if y:
                        return y
        except Exception:
            pass

    m = re.search(r'(\d{4})', s)
    if not m:
        # maybe an epoch as a long string
        m2 = re.search(r'\d{10,13}', s)
        if m2:
            try:
                n = float(m2.group(0))
                return _extract_year(n)
            except Exception:
                return None
        return None

    y = int(m.group(1))
    return y if 1970 <= y <= 2100 else None

def _clean_title_for_hltb(title: str) -> List[str]:
    """
    Generate cleaned variants for HLTB search to improve matching.
    """
    t = title.replace("™", "").replace("®", "")
    t = re.sub(r'\s+', ' ', t).strip()

    variants = {t}

    # remove bracketed parts
    v = re.sub(r'\s*[\(\[\{].*?[\)\]\}]\s*', ' ', t)
    variants.add(re.sub(r'\s+', ' ', v).strip())

    # remove common edition keywords
    v = re.sub(r'\b(complete|definitive|game of the year|goty|ultimate|remastered|remaster|redux|hd|enhanced|director[’\'`s ]*cut|anniversary|gold|deluxe|collection|bundle|edition)\b',
               '', t, flags=re.IGNORECASE)
    variants.add(re.sub(r'\s+', ' ', v).strip())

    # split on hyphen-like separators
    for sep in (' - ', ' — '):
        if sep in t:
            variants.add(t.split(sep)[0].strip())

    # keep the prefix before ':'
    if ':' in t:
        variants.add(t.split(':')[0].strip())

    ordered = sorted(variants, key=lambda x: (len(x), x))
    if t in ordered:
        ordered.remove(t)
    return [t] + ordered

# =========================
# HowLongToBeat integration
# =========================

HLTB_AVAILABLE = False
try:
    from howlongtobeatpy import HowLongToBeat
    HLTB_AVAILABLE = True
except Exception:
    HLTB_AVAILABLE = False

def hltb_lookup_cached(title: str, platform_hint: Optional[str], cache: dict) -> Optional[dict]:
    """
    Resolve HLTB main/main+extra/comp times for a title.
    Caches per cleaned-title+platform-hint variant to minimize requests.
    """
    if not HLTB_AVAILABLE:
        return None

    for cand in _clean_title_for_hltb(title):
        key = f"{cand}::{platform_hint or 'all'}"
        if key in cache:
            if cache[key] is not None:
                return cache[key]
            # else: negative cache, keep trying next variant

        try:
            client = HowLongToBeat()
            results = client.search_sync(cand) if hasattr(client, "search_sync") else client.search(cand)
            if not results:
                cache[key] = None
                continue

            def score(entry):
                sim = getattr(entry, "similarity", 0.0)
                bonus = 0.0
                if platform_hint:
                    try:
                        if platform_hint.lower() in entry.game_name.lower():
                            bonus = 0.1
                    except Exception:
                        pass
                return sim + bonus

            best = max(results, key=score)
            resolved = {
                "main": float(best.main_story) if best.main_story is not None else None,
                "main_extra": float(best.main_extra) if best.main_extra is not None else None,
                "comp": float(best.completionist) if best.completionist is not None else None,
            }
            cache[key] = resolved
            return resolved
        except Exception:
            cache[key] = None
            continue

    return None

# =========================
# IO helpers
# =========================

def find_db_path() -> str:
    """Find the Galaxy DB in known locations or raise an error."""
    for path in DB_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "galaxy-2.0.db not found in known paths. "
        "Edit DB_PATHS in the script to point to your real path "
        "(e.g. C:\\ProgramData\\GOG.com\\Galaxy\\storage\\galaxy-2.0.db)."
    )

def find_galaxy_client() -> Optional[str]:
    """Locate GalaxyClient.exe if present."""
    for p in GALAXY_EXE_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

def load_json(path: Path) -> dict:
    """Load JSON from path or return empty dict."""
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(path: Path, data: dict):
    """Persist JSON to path (best-effort)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def ask(prompt: str, default: Optional[str] = None) -> str:
    """Prompt user synchronously with default fallback."""
    s = input(prompt).strip()
    return s if s else (default or "")

# =========================
# Database readers
# =========================

def load_installed_release_keys(con: sqlite3.Connection) -> set:
    """
    Collect installed release keys (normalized to lowercase) from Galaxy DB
    for GOG and integrated platforms.
    """
    installed: set = set()
    cur = con.cursor()

    # GOG
    try:
        cur.execute("SELECT productId FROM InstalledProducts")
        for (pid,) in cur.fetchall():
            installed.add(f"gog_{pid}".lower())
    except sqlite3.Error:
        pass

    # External platforms (Steam/Epic/etc.)
    try:
        cur.execute("""
            SELECT Platforms.name, InstalledExternalProducts.productId
            FROM InstalledExternalProducts
            JOIN Platforms ON InstalledExternalProducts.platformId = Platforms.id
        """)
        for name, pid in cur.fetchall():
            installed.add(f"{str(name).lower()}_{pid}".lower())
    except sqlite3.Error:
        pass

    return installed

def _detect_key_column(columns: List[str]) -> Optional[str]:
    """Heuristically detect the column holding releaseKey/gameId."""
    raw = [c for c in columns if c]
    lower = [c.lower() for c in raw]

    candidates = (
        "releasekey", "release_key", "gamereleasekey", "game_release_key",
        "gameid", "game_id", "game"
    )
    for c in candidates:
        if c in lower:
            return raw[lower.index(c)]

    for i, lc in enumerate(lower):
        if (("release" in lc and "key" in lc) or (lc.startswith("game") and "key" in lc)):
            return raw[i]
    return None

def _detect_time_columns(columns: List[str]) -> List[str]:
    """Heuristically detect time columns (playtime/seconds/minutes/hours)."""
    raw = [c for c in columns if c]
    lower = [c.lower() for c in raw]
    out = []
    for i, lc in enumerate(lower):
        if any(k in lc for k in ("playtime", "timeplayed", "minutes", "seconds", "time_played", "hours", "time", "duration")):
            out.append(raw[i])
    return out

def _normalize_seconds_proxy_to_hours(secs_proxy: float) -> float:
    """
    Normalize a fuzzy 'time' sum to hours based on magnitude:
    - > 10_000   → treat as seconds
    - > 600      → treat as minutes
    - else       → already hours
    """
    hours = secs_proxy
    if hours > 10_000:
        return hours / 3600.0
    if hours > 600:
        return hours / 60.0
    return hours

def read_playtimes_hours(con: sqlite3.Connection) -> Dict[str, float]:
    """
    Extract playtimes in HOURS across several possible tables and JSON in GamePieces.
    Keys are stored in lowercase.
    """
    cur = con.cursor()
    playtime_hours: Dict[str, float] = {}

    table_candidates = (
        "GameActivities", "UserGameActivities",
        "Playtime", "UserPlaytime",
        "UserGameStats", "UserStats",
    )

    # Pass 1: scan candidate tables
    for table in table_candidates:
        try:
            cur.execute(f"PRAGMA table_info({table})")
            info = cur.fetchall()
            if not info:
                continue

            cols = [c[1] for c in info]
            key_col = _detect_key_column(cols)
            time_cols = _detect_time_columns(cols)
            if not key_col or not time_cols:
                continue

            cur.execute(f"SELECT {', '.join([key_col] + time_cols)} FROM {table}")
            for row in cur.fetchall():
                release_key = str(row[0]).lower()
                if not release_key:
                    continue

                secs_proxy = 0.0
                for val in row[1:]:
                    if val is None:
                        continue
                    if isinstance(val, (int, float)):
                        v = float(val)
                    else:
                        m = re.search(r'[-+]?\d*\.?\d+', str(val))
                        v = float(m.group(0)) if m else 0.0
                    secs_proxy += v

                if secs_proxy <= 0:
                    continue

                hours = _normalize_seconds_proxy_to_hours(secs_proxy)
                playtime_hours[release_key] = max(playtime_hours.get(release_key, 0.0), float(hours))
        except sqlite3.Error:
            continue

    # Pass 2: scan JSON in GamePieces (always, then merge with max())
    try:
        cur.execute("""
            SELECT gp.releaseKey, gpt.type, gp.value
            FROM GamePieces gp
            JOIN GamePieceTypes gpt ON gp.gamePieceTypeId = gpt.id
        """)
        for rk, _typ, val in cur.fetchall():
            release_key = str(rk).lower()
            s = str(val or "")
            if not _looks_like_json(s):
                continue

            try:
                obj = json.loads(s)
            except Exception:
                continue

            def pull_numbers(node) -> List[Tuple[str, float]]:
                nums: List[Tuple[str, float]] = []
                if isinstance(node, dict):
                    for k, v in node.items():
                        lk = str(k).lower()
                        if any(x in lk for x in (
                            "playtime", "timeplayed", "minutesplayed", "secondsplayed",
                            "hoursplayed", "time_played", "time", "duration"
                        )):
                            n = _extract_any_number(v)
                            if n is not None:
                                nums.append((lk, float(n)))
                        else:
                            nums.extend(pull_numbers(v))
                elif isinstance(node, list):
                    for item in node:
                        nums.extend(pull_numbers(item))
                return nums

            nums = pull_numbers(obj)
            if not nums:
                continue

            # choose the most reasonable interpretation and normalize to hours
            best_hours: Optional[float] = None
            for key, n in nums:
                if "hours" in key:
                    best_hours = n; break
                if "seconds" in key:
                    best_hours = n / 3600.0; break
                if "minutes" in key:
                    best_hours = n / 60.0; break
                if ("timeplayed" in key or "playtime" in key or "time_played" in key
                    or "duration" in key or key.endswith("time")):
                    best_hours = (n / 3600.0 if n > 10_000 else (n / 60.0 if n > 600 else n))
                    break

            if best_hours and best_hours > 0:
                playtime_hours[release_key] = max(playtime_hours.get(release_key, 0.0), float(best_hours))
    except sqlite3.Error:
        pass

    return playtime_hours

def build_library(con: sqlite3.Connection) -> List[Dict]:
    """
    Build the full library with robust metadata extraction across DB shapes.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT gp.releaseKey, lower(gpt.type), gp.value
        FROM GamePieces gp
        JOIN GamePieceTypes gpt ON gp.gamePieceTypeId = gpt.id
    """)
    rows = cur.fetchall()

    # group rows by releaseKey
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for release_key, piece_type, raw_value in rows:
        bucket = grouped.setdefault(release_key, {})
        bucket.setdefault(piece_type, []).append(raw_value)

    library: List[Dict] = []
    for rk, bucket in grouped.items():
        rk_raw = str(rk)
        rk_norm = rk_raw.lower()
        platform = rk_norm.split('_', 1)[0] if '_' in rk_norm else 'gog'

        # Title
        title = None
        for cand in ("originaltitle", "title", "names", "displayname"):
            if cand in bucket:
                title = _extract_title(bucket[cand][0])
                break
        if not title:
            found = False
            for vals in bucket.values():
                candidate = vals[0]
                if _looks_like_json(str(candidate)):
                    try:
                        j = json.loads(candidate)
                        for key in ("title", "name"):
                            if key in j:
                                title = str(j[key]); found = True; break
                    except Exception:
                        pass
                if found:
                    break
        title = title or rk_raw

        # Genres / Tags: scan all pieces together
        all_pieces = [v for vals in bucket.values() for v in vals]
        genres = _extract_string_list(all_pieces)
        tags = _extract_string_list(all_pieces)

        # Rating
        rating: Optional[float] = None
        for cand in ("userscore", "criticscore", "communityrating", "reviewscore", "rating", "score", "metascore"):
            if cand in bucket and rating is None:
                for v in bucket[cand]:
                    r = _extract_any_number(v)
                    if r is not None:
                        rating = r
                        break
            if rating is not None:
                break
        if rating is None:
            candidates: List[float] = []
            for v in all_pieces:
                candidates += _json_collect_numbers_by_keys(str(v), r"(user|critic|meta)?(score|rating)")
            if candidates:
                rating = max(candidates)

        # Year (first plausible value wins)
        year: Optional[int] = None
        for v in all_pieces:
            year = _extract_year(v)
            if year:
                break

        library.append({
            "releaseKey": rk_norm,
            "releaseKeyRaw": rk_raw,
            "platform": platform,
            "title": title,
            "genres": genres,
            "tags": tags,
            "rating": rating,
            "year": year,
        })

    return library

# =========================
# Filtering
# =========================

def filter_library(items: List[Dict],
                   platform: str = 'all',
                   genre: Optional[str] = None,
                   tag: Optional[str] = None,
                   min_rating: Optional[float] = None,
                   installed_only: Optional[bool] = None,
                   installed_set: Optional[set] = None) -> List[Dict]:
    """Apply platform/genre/tag/rating/install-status filters to the library."""
    results: List[Dict] = []
    for it in items:
        if platform != 'all' and it["platform"] != platform:
            continue
        if genre and not any(genre.lower() in g.lower() for g in it["genres"]):
            continue
        if tag and not any(tag.lower() in t.lower() for t in it["tags"]):
            continue
        if min_rating is not None:
            r = it["rating"] if it["rating"] is not None else 0.0
            if r < min_rating:
                continue
        if installed_only is not None:
            is_inst = (it["releaseKey"] in installed_set) if installed_set is not None else False
            if installed_only and not is_inst:
                continue
            if installed_only is False and is_inst:
                continue
        results.append(it)
    return results

# =========================
# Selection helpers & modes
# =========================

def pick_random(items: List[Dict]) -> Optional[Dict]:
    """Pick a random item, or None if empty."""
    if not items:
        return None
    return random.choice(items)

def pick_weighted(items: List[Dict], weights: List[float]) -> Optional[Dict]:
    """Weighted random choice with graceful fallback."""
    if not items or not weights or sum(weights) <= 0:
        return pick_random(items)
    return random.choices(items, weights=weights, k=1)[0]

def mode_roulette(candidates: List[Dict]) -> Optional[Dict]:
    return pick_random(candidates)

def mode_give_me_three(candidates: List[Dict]) -> Optional[Dict]:
    """Show three random games and let the user pick one."""
    # This mode requires user input to choose 1 of 3. Always prompt.
    if not candidates:
        return None
    k = min(3, len(candidates))
    picks = random.sample(candidates, k)
    for i, g in enumerate(picks, 1):
        print(f"[{i}] {g['title']}") # More info? ({g['platform']})  Year: {g.get('year', '-')}
    while True:
        choice = ask("Pick 1/2/3 or 'r' to reroll: ").lower()
        if choice in ('1', '2', '3') and int(choice) <= k:
            return picks[int(choice) - 1]
        if choice == 'r':
            return mode_give_me_three(candidates)
        print("Invalid input.")

def mode_mini_tournament(candidates: List[Dict]) -> Optional[Dict]:
    """Run a quick single-elimination tournament of up to 8 games."""
    if not candidates:
        return None
    size = min(8, len(candidates))
    bracket = random.sample(candidates, size)
    rnd = 1
    while len(bracket) > 1:
        print(f"\n* Round {rnd} (participants {len(bracket)}) *")
        next_round: List[Dict] = []
        for i in range(0, len(bracket), 2):
            if i + 1 >= len(bracket):
                next_round.append(bracket[i])
                continue
            a, b = bracket[i], bracket[i + 1]
            print(f"1) {a['title']}") # More info?  ({a['platform']}, {a.get('year', '-')})")
            print(f"2) {b['title']}") # More info? ({b['platform']}, {b.get('year', '-')})")
            while True:
                ch = ask("Choose 1/2: ").strip().lower()
                if ch in ('1', '2'):
                    next_round.append(a if ch == '1' else b)
                    break
                print("Invalid input.")
        bracket = next_round
        rnd += 1
    print("\n>>> TOURNAMENT WINNER:", bracket[0]['title'])
    return bracket[0]

def mode_weight_by_playtime(candidates: List[Dict], playtimes_hours: Dict[str, float]) -> Optional[Dict]:
    """Weight towards games with fewer played hours (to help backlog discovery)."""
    weights: List[float] = []
    for g in candidates:
        hrs = playtimes_hours.get(g["releaseKey"], 0.0)
        weights.append(1.0 / (1.0 + max(0.0, hrs)))
    return pick_weighted(candidates, weights)

def mode_weight_by_rating(candidates: List[Dict]) -> Optional[Dict]:
    """Weight towards higher-rated games (assuming 0-10 or 0-100 scales)."""
    weights: List[float] = []
    for g in candidates:
        r = g["rating"]
        if r is None:
            weights.append(1.0)
        else:
            weights.append(max(1.0, r / 10.0) if r > 10 else max(1.0, r))
    return pick_weighted(candidates, weights)

def mode_nostalgia(candidates: List[Dict], year_threshold: Optional[int] = None) -> Optional[Dict]:
    """
    Prefer older titles at or before a cutoff year.
    If no year provided, prompt for it; default to current year on invalid input.
    """
    if year_threshold is None:
        try:
            y_str = ask("Nostalgia cutoff year: ").strip()
            year_threshold = int(y_str)
        except Exception:
            print("Invalid year, defaulting to current year")
            year_threshold = datetime.now(timezone.utc).year

    oldies = [g for g in candidates if g.get("year") and g["year"] <= year_threshold]
    pool = oldies if oldies else candidates
    return pick_random(pool)

def mode_opposite(candidates: List[Dict], all_items: List[Dict], current_filtered: List[Dict]) -> Optional[Dict]:
    """
    Pick something 'different': outside current filter set, skewed to lower-rated recent games.
    """
    filtered_keys = {g["releaseKey"] for g in current_filtered}
    others = [g for g in all_items if g["releaseKey"] not in filtered_keys] or candidates or all_items
    low_rating = [g for g in others if (g["rating"] or 0) < 60] or others
    recent = [g for g in low_rating if (g.get("year") or 0) >= 2018] or low_rating
    return pick_random(recent)

# ---------------------------------
# HLTB under-hours fast search mode
# ---------------------------------
def mode_hltb_under_hours(candidates: List[Dict], hltb_cache: dict, max_hours: float, check_type: str = "2",
    timeout_seconds: float = 60.0, batch_size: int = 1, tolerance: int = 0.2) -> Optional[Dict]:
    """
    Quickly find a game whose HLTB 'main' time is <= max_hours.
    Strategy:
      - Randomly sample batches of size `batch_size` from unseen candidates.
      - In each batch, resolve HLTB and keep games with main <= max_hours.
      - If there are eligible games in the batch, choose the one closest to max_hours (max main).
      - Keep trying new batches until a match is found or `timeout_seconds` elapse.
      - Show a live progress bar so the UI doesn't look stuck.
    """
    if not HLTB_AVAILABLE:
        print("HowLongToBeat is not available. Install 'howlongtobeatpy'.")
        return None

    start = time.monotonic()
    total = len(candidates)
    tried_idxs: set[int] = set()

    def _print_progress_bar():
        """Render a simple progress bar based on elapsed time (or items if uncommented), so it feels responsive."""
        elapsed = time.monotonic() - start
        pct = min(100.0, (elapsed / timeout_seconds) * 100.0 if timeout_seconds > 0 else 100.0)
        #pct = (len(tried_idxs) / total) * 100 if total > 0 else 0
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r[{bar}] {pct:5.1f}% "
            f"(checked {len(tried_idxs)}/{total}) | elapsed {elapsed:4.1f}s"
        )
        sys.stdout.flush()

    while (time.monotonic() - start) < timeout_seconds and len(tried_idxs) < total:
        _print_progress_bar()
        # choose a fresh batch of unseen candidates
        pool = [i for i in range(total) if i not in tried_idxs]
        if not pool:
            break
        k = min(batch_size, len(pool))
        picked_idxs = random.sample(pool, k)
        tried_idxs.update(picked_idxs)

        # resolve HLTB for this batch and filter by <= max_hours
        eligible: List[Tuple[Dict, float]] = []
        for idx in picked_idxs:
            game = candidates[idx]
            _print_progress_bar()
            row = hltb_lookup_cached(game["title"], game["platform"], hltb_cache)    
            _print_progress_bar()
            if not row:
                continue
            main = row.get("main")

            if (main >= (max_hours - max_hours * tolerance) and main <= (max_hours + max_hours * tolerance)) or \
               ((check_type == "1") and main >= max_hours) or \
               ((check_type == "3") and main <= max_hours):
                checked_ok = True
            else:
                checked_ok = False

            if isinstance(main, (int, float)) and checked_ok:
                eligible.append((game, float(main)))
                checked_ok = False

        if eligible:
            # pick the one closest to max_hours (i.e., largest main <= max_hours)
            chosen, _ = max(eligible, key=lambda x: x[1])
            _print_progress_bar()
            sys.stdout.write("\n")
            sys.stdout.flush()
            return chosen

    _print_progress_bar()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("No game found within timeout")
    return None

# =========================
# Launch & display
# =========================

def launch_game(release_key: str, title: str):
    """Launch a game via Galaxy client or URI fallback."""
    client = find_galaxy_client()
    game_id = release_key
    if client:
        cmd = [client, "/command=runGame", f"/gameId={game_id}"]
        print(f"\n> Launching via Galaxy: {title}  [{game_id}]")
        subprocess.Popen(cmd, shell=False)
    else:
        uri = f"goggalaxy://openGame/{game_id}"
        print(f"\n> GalaxyClient.exe not found. Trying URI: {uri}")
        try:
            os.startfile(uri)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"Open failed: {e}")

def print_game_brief(game: Dict, installed_keys: set, playtimes_hours: Dict[str, float], hltb_row: Optional[dict]):
    """Print the selected game summary."""
    badge = "Installed" if game["releaseKey"] in installed_keys else "Not Installed"
    rating_str = f"{game['rating']:.2f}" if isinstance(game.get("rating"), (int, float)) else "-"
    year_str = game.get("year", "-")
    played = playtimes_hours.get(game["releaseKey"])
    played_str = f"{played:.1f} h" if played is not None else "-"
    print("\n================ SELECTED GAME ================")
    print(f"Title     : {game['title']}")
    print(f"Platform  : {game['platform']}")
    print(f"Year      : {year_str}")
    print(f"Status    : {badge}")
    print(f"Playtime  : {played_str}")
    # If you want to show these, uncomment:
    # print(f"Genres    : {', '.join(game['genres']) if game['genres'] else '-'}")
    # print(f"Tags      : {', '.join(game['tags']) if game['tags'] else '-'}")
    print(f"Rating    : {rating_str}")
    if hltb_row:
        def fmt(x): return f"{x:.1f} h" if isinstance(x, (int, float)) else "-"
        print(f"HLTB Main : {fmt(hltb_row.get('main'))}")
        # If preferred, also show:
        # print(f"HLTB M+E  : {fmt(hltb_row.get('main_extra'))}")
        # print(f"HLTB Comp.: {fmt(hltb_row.get('comp'))}")
    print("===============================================")

# =========================
# Select mode logic
# =========================

def run_selection_mode(
    mode_key: str,
    candidates: List[Dict],
    library: List[Dict],
    playtimes_hours: Dict[str, float],
    hltb_cache: dict,
    last_params: Optional[dict] = None,
    allow_prompt: bool = True,
) -> Tuple[Optional[Dict], dict]:
    """
    Execute a selection mode and return (picked_game, params_used).
    If allow_prompt=False, reuse last_params and avoid re-asking the user.
    """
    params_used: dict = {}

    if mode_key == '1':
        print("\n--- ROULETTE ---")
        picked = mode_roulette(candidates)

    elif mode_key == '2':
        print("\n--- GIVE ME THREE ---")
        picked = mode_give_me_three(candidates)

    elif mode_key == '3':
        print("\n--- MINI TOURNAMENT ---")
        picked = mode_mini_tournament(candidates)

    elif mode_key == '4':
        print("\n--- CHOOSE BY PLAYED HRS ---")
        picked = mode_weight_by_playtime(candidates, playtimes_hours)

    elif mode_key == '5':
        print("\n--- CHOOSE BY RATING ---")
        picked = mode_weight_by_rating(candidates)

    elif mode_key == '6':
        print("\n--- NOSTALGIA CHOOSER ---")
        # Nostalgia (cutoff year): reuse prior if available, otherwise prompt (if allowed)
        if last_params and "year_threshold" in last_params:
            year_threshold = last_params["year_threshold"]
        elif allow_prompt:
            try:
                y_str = ask("Choose cutoff year: ").strip()
                year_threshold = int(y_str)
            except Exception:
                print("Invalid year, defaulting to current year")
                year_threshold = datetime.now(timezone.utc).year
        else:
            # No prior param and we aren't allowed to prompt — abort gracefully
            return None, {}
        params_used["year_threshold"] = year_threshold
        picked = mode_nostalgia(candidates, year_threshold=year_threshold)

    elif mode_key == '7':
        print("\n--- OPPOSITE FILTERS ---")
        picked = mode_opposite(candidates, library, candidates)

    elif mode_key == '8':
        print("\n--- HOW LONG TO BEAT ---")
        # HLTB under-hours: reuse prior if available, otherwise prompt (if allowed)
        timeout_seconds = (last_params or {}).get("timeout_seconds", 30.0)
        batch_size = (last_params or {}).get("batch_size", 10)
        if last_params and "max_hours" in last_params:
            max_hours = float(last_params["max_hours"])
            check_type = last_params["check_type"]
        elif allow_prompt:
            hrs_str = ask("HLTB 'Main' hours: ").strip()
            print("1) More | 2) Equal (default) | 3) Less")
            check_type = ask("> ").lower()
            try:
                max_hours = float(hrs_str)
            except ValueError:
                print("Invalid number.")
                return None, {}
        else:
            # No prior param and we aren't allowed to prompt — abort gracefully
            return None, {}

        params_used.update({"max_hours": max_hours, "timeout_seconds": timeout_seconds, 
                            "batch_size": batch_size, "check_type": check_type})
        picked = mode_hltb_under_hours(candidates, hltb_cache, max_hours, check_type)
        # Persist cache after lookups in this mode
        save_json(HLTB_CACHE_PATH, hltb_cache)

    else:
        print("Unknown option.")
        return None, {}

    return picked, params_used

# =========================
# Interactive UI
# =========================

def interactive_loop(library: List[Dict], installed_keys: set, playtimes_hours: Dict[str, float]):
    platform = 'all'
    genre = None
    tag = None
    min_rating = None
    installed_only = None

    hltb_cache = load_json(HLTB_CACHE_PATH)

    # Track last used mode & params so 'R' (reroll) can reuse them
    last_mode: Optional[str] = None
    last_params: dict = {}

    def show_filters():
        print("\n--- Current filters ---")
        print(f"Platform    : {platform}")
        print(f"Genre       : {genre or '-'}")
        print(f"Tag         : {tag or '-'}")
        print(f"Rating ≥    : {min_rating if min_rating is not None else '-'}")
        st = "installed only" if installed_only is True else ("not installed only" if installed_only is False else "any")
        print(f"Install     : {st}")

    def change_filters():
        nonlocal platform, genre, tag, min_rating, installed_only
        print("\nChange filters (Enter to keep value):")
        platform = (ask("Platform [all/gog/steam/epic/...]: ", platform) or platform).lower()
        g = ask("Genre (substring): ", genre or "")
        genre = g or None
        t = ask("Tag (substring): ", tag or "")
        tag = t or None
        r = ask("Min rating (number): ", str(min_rating) if min_rating is not None else "")
        if r.strip():
            try:
                min_rating = float(r)
            except ValueError:
                print("  (invalid; keeping previous)")
        else:
            min_rating = None
        s = ask("Install state [any/installed/not]: ", "any").lower()
        if s in ("installed", "i", "yes", "y"):
            installed_only = True
        elif s in ("not", "no", "n"):
            installed_only = False
        else:
            installed_only = None

    def current_candidates():
        return filter_library(
            library,
            platform=platform,
            genre=genre,
            tag=tag,
            min_rating=min_rating,
            installed_only=installed_only,
            installed_set=installed_keys
        )

    while True:
        candidates = current_candidates()
        
        print(f"=== WHAT SHOULD I PLAY? ===")
        print(f"\nGames in Galaxy library: {len(library)}")
        print(f"Matches with filters: {len(candidates)}")

        print("\nSelection modes:")
        print(" 1) Roulette")
        print(" 2) Give me three")
        print(" 3) Mini tournament")
        print(" 4) Weight by played hours (fewer hours -> higher prob.)")
        print(" 5) Weight by rating (higher rating -> higher prob.)")
        print(" 6) Nostalgia mode")
        print(" 7) Opposite filtering")
        print(" 8) How Long to Beat")
        print(" F) Change filters   S) Show filters   Q) Quit")

        mode_choice = ask("> ").lower()

        if mode_choice == 'q':
            print("Bye!")
            break
        if mode_choice == 'f':
            change_filters(); continue
        if mode_choice == 's':
            show_filters(); continue

        if not candidates:
            print("No candidates. Adjust filters (press F).")
            continue

        picked, used_params = run_selection_mode(
            mode_choice, candidates, library, playtimes_hours, hltb_cache,
            last_params=None, allow_prompt=True
        )
        if not picked:
            print("Could not select a game (no candidates in selected mode).")
            continue

        # Remember last mode & parameters for rerolls
        last_mode = mode_choice
        last_params = used_params

        # Print brief and enter actions loop
        while True:
            hltb_row = hltb_lookup_cached(picked["title"], picked["platform"], hltb_cache)
            save_json(HLTB_CACHE_PATH, hltb_cache)
            print_game_brief(picked, installed_keys, playtimes_hours, hltb_row)

            print("\nActions: [L] Launch  [R] Reroll  [C] Change filters  [M] Back to modes  [Q] Quit")
            act = ask("> ").lower()

            if act == 'l':
                launch_game(picked.get("releaseKeyRaw", picked["releaseKey"]), picked["title"])
                # stay in actions after launch

            elif act == 'r':
                # Reroll: run the SAME mode with SAME params; do not prompt.
                if not last_mode:
                    print("No previous mode to reroll.")
                    continue

                # Use current candidates (same filters) but identical mode/params
                candidates = current_candidates()
                new_picked, _ = run_selection_mode(
                    last_mode, candidates, library, playtimes_hours, hltb_cache,
                    last_params=last_params, allow_prompt=False
                )
                if not new_picked:
                    print("Reroll failed (no candidates or params missing).")
                    continue

                picked = new_picked  # update current selection and loop to show it again
                continue

            elif act == 'c':
                change_filters()
                # After changing filters, recompute candidates and attempt immediate reroll with same mode/params
                if last_mode:
                    candidates = current_candidates()
                    new_picked, _ = run_selection_mode(
                        last_mode, candidates, library, playtimes_hours, hltb_cache,
                        last_params=last_params, allow_prompt=False
                    )
                    if new_picked:
                        picked = new_picked
                        continue  # show new pick with same mode/params under new filters
                break  # go back to mode list

            elif act == 'm':
                break  # back to mode list

            elif act == 'q':
                print("Bye!")
                return

            else:
                print("Unknown option.")

# =========================
# Entry point
# =========================

def main():
    try:
        db_path = find_db_path()
        con = sqlite3.connect(db_path)
        with con:
            installed_keys = load_installed_release_keys(con)
            playtimes_hours = read_playtimes_hours(con)
            library = build_library(con)

        library.sort(key=lambda x: x["title"].lower() if x["title"] else x["releaseKey"].lower())
        
        if not HLTB_AVAILABLE:
            print("Note: 'howlongtobeatpy' not available. Install with: pip install howlongtobeatpy")

        interactive_loop(library, installed_keys, playtimes_hours)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()