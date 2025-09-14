# What Should I Play?

*WSIP is a Python-based tiny terminal utility that helps you stop dithering and start playing. It scans your local GOG Galaxy 2.0 library (GOG + integrated Steam/Epic) so you can filter, pick a mode, get a random smart suggestion, and launch the game to play.*

---

## Features

* Reads your full Galaxy library (GOG + integrated platforms) directly from the **Galaxy DB**.
* Robust extraction of **title / genres / tags / rating / year / playtime** from multiple DB shapes.
* Multiple selection modes: roulette, “give me three”, mini-tournament, rating/playtime weighting, nostalgia, “opposite”, and **HLTB target time**.
* Optional **HowLongToBeat** lookups with a persistent cache to speed up repeats.
* Launches the selected game via **GalaxyClient.exe** (or Galaxy URI fallback).
* **Reroll** action re-runs the **same mode with the same parameters** (and current filters).

---

## Requirements

* **OS:** Windows (paths target common Windows Galaxy locations; other OSes may work for non-launch tasks if you point to a valid DB).
* **Python:** 3.8+ recommended.
* **GOG Galaxy 2.0** installed with a populated database.
* **Internet:** only needed for HowLongToBeat lookups.

### Python dependencies

Only one third-party package:

* [`howlongtobeatpy`](https://pypi.org/project/howlongtobeatpy/) — HLTB lookups (optional; script runs without it but HLTB mode is disabled).

Everything else uses Python’s standard library (`sqlite3`, `json`, `pathlib`, etc.).

---

## Installation

### 1) Install Python

On Windows, install Python from [python.org](https://www.python.org/downloads/) and check **“Add Python to PATH.”**

### 2) (Optional) Create a virtual environment

```powershell
# In the folder where you cloned/saved the script:
py -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
# or (Command Prompt):
# .\.venv\Scripts\activate.bat
```

### 3) Install the dependency

```powershell
pip install --upgrade pip
pip install howlongtobeatpy
```

> If you skip this, the script still runs, but the HLTB-based mode will be unavailable.

---

## Configuration

Open **`wsip.py`** and adjust these if needed.

### `DB_PATHS`

Paths where the script looks for the Galaxy DB file:

```
C:\ProgramData\GOG.com\Galaxy\storage\galaxy-2.0.db
```

If your DB is elsewhere, add the full path to `DB_PATHS`.

### `GALAXY_EXE_CANDIDATES`

Candidate paths to `GalaxyClient.exe` (used by the **Launch** action):

```
C:\Program Files (x86)\GOG Galaxy\GalaxyClient.exe
C:\Program Files\GOG Galaxy\GalaxyClient.exe
```

Update if your Galaxy is installed somewhere else.

### Cache file

HLTB results are cached here (created automatically):

```
%APPDATA%\RandomGalaxy\hltb_cache.json
```

Delete this file to clear the cache.

---

## Running

From the folder containing the script:

```powershell
# If using the venv:
.\.venv\Scripts\Activate.ps1

# Run the script:
py wsip.py
# or:
python wsip.py
```

On start, the script prints how many games it found. If `howlongtobeatpy` isn’t installed, it will say so (HLTB mode will be disabled).

---

## Using the app

### Filters

Press **`F`** to change filters (Enter keeps current values):

* **Platform:** `all` / `gog` / `steam` / `epic` / …
* **Genre:** substring match (e.g., `rpg`)
* **Tag:** substring match (e.g., `roguelike`)
* **Min rating:** number (works with 0–10 or 0–100 styles)
* **Install state:** `any` / `installed` / `not`

Press **`S`** to show current filters.

### Selection modes

After filtering, pick a mode:

* **Roulette** — random pick from the current matches.
* **Give me three** — shows 3 random games; choose `1/2/3` (or `r` to reroll the trio).
* **Mini tournament** — quick single-elimination bracket up to 8 games; you choose winners.
* **Weight by played hours** — favors games you’ve played **less** (good for backlog).
* **Weight by rating** — favors higher-rated games.
* **Nostalgia mode** — prompts for a **cutoff year**; prefers games released **on or before** that year.
* **Opposite filtering** — tries to pick something “different” (outside current set, leaning recent & lower-rated).
* **How Long To Beat** — prompts for a target HLTB **‘Main’ hours** and a comparison:

  * `1) More` — find a game **≥ target** hours
  * `2) Equal (default)` — find a game **near** target (±20%)
  * `3) Less` — find a game **≤ target** hours

  This mode samples the library in small random batches, resolves HLTB times (using cache), shows a **live progress bar**, and stops when it finds a match close to your target. If nothing matches before the timeout, it prints **“No game found within timeout.”**

> **Tip:** The HLTB cache grows over time, making lookups faster on reruns.

### After a game is picked (Actions)

* **`L` Launch** — starts the game via Galaxy (or falls back to `goggalaxy://` URI).
* **`R` Reroll** — re-runs the **same mode with the same parameters** (and current filters) to pick a new candidate without prompting again.
* **`C` Change filters** — returns to the filter prompt; after changing, the app attempts a reroll with the **same mode/params** under the new filters.
* **`M` Back to modes** — returns to the mode menu.
* **`Q` Quit** — exits.

---

## Troubleshooting

### “galaxy-2.0.db not found”

Update the `DB_PATHS` list in the script to point to your actual DB. Common default:

```
C:\ProgramData\GOG.com\Galaxy\storage\galaxy-2.0.db
```

*(Hidden by default; you may need admin privileges or enable hidden items.)*

### “HowLongToBeat is not available …”

Install:

```powershell
pip install howlongtobeatpy
```

Without it, all modes work **except** the HLTB-based mode.

### Launch fails

Ensure `GALAXY_EXE_CANDIDATES` matches your install, or open the selected game once in Galaxy so the URI scheme works. The script attempts:

```
goggalaxy://openGame/<id>
```

if the EXE isn’t found.

### Slow HLTB lookups

First-time lookups can be slower. They’re cached at:

```
%APPDATA%\RandomGalaxy\hltb_cache.json
```

Repeated runs will be faster.

### Tiny match count

Loosen your filters (`F`) or switch selection modes.

---

## Author

**jperezu** - jmp51.21@gmail.com

---

## License (MIT)

Copyright (c) 2025 **jperezu**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to **deal in the Software without restriction**, including without limitation the rights to **use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software**, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in **all copies or substantial portions** of the Software.


THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF **MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT**. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


