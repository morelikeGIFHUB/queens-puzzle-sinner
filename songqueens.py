from __future__ import annotations

import os
import random
import pygame
import colorsys
from collections import deque
import time
from typing import Dict, Set, Tuple, List
from pathlib import Path
from pygame import Surface


_SCRIPT_DIR = Path(__file__).resolve().parent
_PHOTO_DIR = _SCRIPT_DIR / "jannik_sinner_photos"
_QUEEN_FACE_PATH = _PHOTO_DIR / "Jannik Sinner(2).jpg"
_WIN_IMAGE_PATH = _PHOTO_DIR / "sinner win .jpg"
_LAST_WIN_FACE_PATH: Path | None = None


def load_fixed_queen_surface() -> Surface:
    if not _QUEEN_FACE_PATH.exists():
        raise FileNotFoundError(f"Missing queen face image: {_QUEEN_FACE_PATH}")
    return pygame.image.load(str(_QUEEN_FACE_PATH))


def load_fixed_win_surface() -> Surface:
    if not _WIN_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing win image: {_WIN_IMAGE_PATH}")
    return pygame.image.load(str(_WIN_IMAGE_PATH))


def load_random_win_surface() -> tuple[Surface, str]:
    """Returns (surface, credit) from the local jannik_sinner_photos folder.

    Avoids repeating the immediately previous win photo when possible.
    """
    global _LAST_WIN_FACE_PATH

    if not _PHOTO_DIR.exists():
        raise FileNotFoundError(f"Missing photos folder: {_PHOTO_DIR}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    photos = [p for p in _PHOTO_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    photos = [p for p in photos if p.resolve() != _QUEEN_FACE_PATH.resolve()]

    if not photos:
        raise FileNotFoundError(f"No win images found in: {_PHOTO_DIR}")

    chosen = random.choice(photos)
    if _LAST_WIN_FACE_PATH is not None and len(photos) > 1:
        for _ in range(12):
            if chosen != _LAST_WIN_FACE_PATH:
                break
            chosen = random.choice(photos)

    _LAST_WIN_FACE_PATH = chosen
    return pygame.image.load(str(chosen)), chosen.name


pygame.init()
clock = pygame.time.Clock()

# ---------------- Audio ----------------
SOUND_ON = True
try:
    # macOS often needs specific buffer/frequency settings to avoid lag or failure
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
except pygame.error:
    SOUND_ON = False

WIN_SOUND = None
if SOUND_ON:
    win_path = os.path.join(os.path.dirname(__file__), "jannik_sinner_photos", "CCmain i need.mp3")
    if os.path.exists(win_path):
        try:
            WIN_SOUND = pygame.mixer.Sound(win_path)
        except pygame.error as e:
            print(f"Couldn't load win sound: {win_path}. Error: {e}")
            WIN_SOUND = None

# ---------------- Visuals ----------------
FPS = 60
DOUBLE_CLICK_MS = 420

PAD = 24
TOP_BAR = 150
GRID_BORDER = 1
REGION_BORDER = 4

BG = (245, 245, 245)
TEXT = (20, 20, 20)
BLACK = (0, 0, 0)
X_COLOR = (300, 1800, 45000)  # Hot Pink
ILLEGAL_RED = (220, 40, 40)

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]



# Difficulty: mainly size + generation strictness
# (We still bias towards "more forced" but never crash.)
DIFFICULTIES = {
    # Updated sizes requested:
    # easy 6x6, normal 8x8, evil 10x10
    # Keep generation smooth via time budgets.
    "easy":   {"N": 6,  "CELL": 86, "void": 0, "tries": 1200, "target_forced": 5, "require_zero_guess": True,  "prefer_zero_guess": True,  "time_limit_s": 2.4},
    # Normal: restore the exact generation constraints from earlier today:
    # - 8x8
    # - must be deduction-solvable (0-guess) under the full deduction rules
    # Normal: prefer deduction-solvable boards, but don't hard-crash if one
    # can't be found quickly on this machine.
    "normal": {"N": 8,  "CELL": 66, "void": 0, "tries": 1600, "target_forced": 6, "require_zero_guess": False, "prefer_zero_guess": True,  "time_limit_s": 3.0},
    # Lolly: 7x7 with mostly small regions.
    # Generation targets 4 regions with <5 cells, remaining regions expand.
    "lolly":  {"N": 7,  "CELL": 76, "void": 0, "tries": 1400, "target_forced": 6, "require_zero_guess": False, "prefer_zero_guess": True,  "time_limit_s": 2.7},
    "idk_bruv": {"N": 12, "CELL": 42, "void": 0, "tries": 2600, "target_forced": 7, "require_zero_guess": False, "prefer_zero_guess": True, "time_limit_s": 4.2},
    "evil":   {"N": 10, "CELL": 52, "void": 6, "tries": 2200, "target_forced": 6, "require_zero_guess": False, "prefer_zero_guess": True,  "time_limit_s": 3.6},
}

_FONT_NAMES = ["Times New Roman", "Times"]

font = pygame.font.SysFont(_FONT_NAMES, 28)
font_small = pygame.font.SysFont(_FONT_NAMES, 22)
font_tiny = pygame.font.SysFont(_FONT_NAMES, 18)
big_font = pygame.font.SysFont(_FONT_NAMES, 46)

Cell = Tuple[int, int]

UNASSIGNED_ID = -1


# ============================================================
# Helpers
# ============================================================
def in_bounds(r, c, N):
    return 0 <= r < N and 0 <= c < N

def touches(r1, c1, r2, c2):
    return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1

def cell_rect(r, c, CELL):
    x = PAD + c * CELL
    y = PAD + TOP_BAR + r * CELL
    return pygame.Rect(x, y, CELL, CELL)


def pastel_palette(k: int):
    cols = []
    for i in range(k):
        h = (i / k) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.40, 0.98)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    random.shuffle(cols)
    return cols

def draw_text(screen, msg, x, y, f, color=TEXT):
    screen.blit(f.render(msg, True, color), (x, y))


def _wrap_by_chars(msg: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [msg]
    words = msg.split()
    if not words:
        return [""]

    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            if len(w) <= max_chars:
                cur = w
            else:
                for i in range(0, len(w), max_chars):
                    lines.append(w[i : i + max_chars])
                cur = ""
            continue

        if len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            lines.append(cur)
            if len(w) <= max_chars:
                cur = w
            else:
                for i in range(0, len(w), max_chars):
                    lines.append(w[i : i + max_chars])
                cur = ""
    if cur:
        lines.append(cur)
    return lines


def draw_info_panel(state):
    screen = state["screen"]
    N = state["N"]
    diff_key = state["diff"]
    diff_names = {
        "easy": "Easy",
        "normal": "Normal",
        "idk_bruv": "idk bruv",
        "lolly": "lolly",
        "evil": "Evil",
    }
    diff = diff_names.get(diff_key, str(diff_key))
    qcount = len(state["queens"])

    max_chars = 50
    lines: list[str] = []
    lines += _wrap_by_chars(f"Difficulty: {diff}", max_chars)
    lines += _wrap_by_chars("1 Easy", max_chars)
    lines += _wrap_by_chars("2 Normal", max_chars)
    lines += _wrap_by_chars("3 Evil", max_chars)
    lines += _wrap_by_chars("4 lolly", max_chars)
    lines += _wrap_by_chars("R New  C Clear", max_chars)
    lines += _wrap_by_chars("Click=X  Drag=paint X  Dbl=Q", max_chars)
    lines += _wrap_by_chars("Ctrl/Right=Q", max_chars)
    lines += _wrap_by_chars("ESC Quit", max_chars)
    lines += _wrap_by_chars(f"Board {N}x{N}", max_chars)
    lines += _wrap_by_chars(f"Queens {qcount}/{N}", max_chars)

    y = 10
    lh = font_tiny.get_linesize() + 2
    for line in lines:
        if y + lh > TOP_BAR - 6:
            break
        draw_text(screen, line, PAD, y, font_tiny, TEXT)
        y += lh


# ============================================================
# Solution generator (no diagonals, only col-unique + no-touch)
# ============================================================
def generate_solution(N: int):
    def valid_partial(placed, r, c):
        for rr, cc in placed:
            if cc == c:
                return False
            if touches(rr, cc, r, c):
                return False
        return True

    def backtrack(r, placed):
        if r == N:
            return placed
        cols = list(range(N))
        random.shuffle(cols)
        for c in cols:
            if valid_partial(placed, r, c):
                res = backtrack(r + 1, placed + [(r, c)])
                if res is not None:
                    return res
        return None

    return backtrack(0, [])


def _cells_connected(cells: Set[Cell]) -> bool:
    if not cells:
        return True
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(cells)


def _open_connected(N: int, blob: Set[Cell]) -> bool:
    open_cells = {(r, c) for r in range(N) for c in range(N)} - blob
    return _cells_connected(open_cells)


def generate_negative_space_blob(N: int, blob_size: int, seed: Cell, *, attempts: int = 250) -> Set[Cell]:
    """Returns a connected blob of cells to assign to a special region.

    The blob is still playable (not blocked). It is only used to force one region to have a
    distinct, large/irregular shape.
    """
    if blob_size <= 0:
        return set()

    # Keep enough cells for the other regions to exist.
    blob_size = min(blob_size, N * N - (N * 2))
    if blob_size <= 0:
        return set()

    sr, sc = seed
    if not in_bounds(sr, sc, N):
        return set()

    for _ in range(attempts):
        blob: Set[Cell] = {seed}
        frontier = [seed]

        while frontier and len(blob) < blob_size:
            r, c = random.choice(frontier)
            nbrs = []
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc, N) and (rr, cc) not in blob:
                    nbrs.append((rr, cc))
            if not nbrs:
                frontier.remove((r, c))
                continue
            nxt = random.choice(nbrs)
            blob.add(nxt)
            frontier.append(nxt)

        if len(blob) != blob_size:
            continue

        # Must not completely consume any row/col (would make region constraints impossible).
        row_counts = [0] * N
        col_counts = [0] * N
        for r, c in blob:
            row_counts[r] += 1
            col_counts[c] += 1
        if any(v >= N for v in row_counts):
            continue
        if any(v >= N for v in col_counts):
            continue

        # Don't disconnect the remaining board too often; helps region growth succeed.
        if not _open_connected(N, blob):
            continue

        return blob

    return set()


# ============================================================
# Region generator (GUARANTEED assignment)
#   - Assign each cell to a seed via randomized BFS growth
#   - Then do local swaps to create more jagged/creative borders
#   - Keeps all regions connected by construction
# ============================================================
def generate_regions(seed_cells: List[Cell], N: int, mode: str, *, void_blob: Set[Cell] | None = None, void_rid: int | None = None):
    void_blob = void_blob or set()
    # Start each region at its seed
    regions = [[UNASSIGNED_ID for _ in range(N)] for _ in range(N)]
    fronts = [deque() for _ in range(N)]

    if void_rid is not None and void_blob:
        for r, c in void_blob:
            if in_bounds(r, c, N):
                regions[r][c] = void_rid

    for rid, (r, c) in enumerate(seed_cells):
        if regions[r][c] != UNASSIGNED_ID and regions[r][c] != rid:
            # Seed landed inside the void blob for a different region.
            return None
        regions[r][c] = rid
        fronts[rid].append((r, c))

    # Track region sizes as we grow (used for lolly target caps).
    region_sizes = [0] * N
    for r in range(N):
        for c in range(N):
            rid = regions[r][c]
            if rid != UNASSIGNED_ID and 0 <= rid < N:
                region_sizes[rid] += 1

    size_targets: list[int] | None = None
    if mode == "lolly":
        # Lolly: force *most* regions to be small, but not all.
        # Exactly 4 regions should end up with <5 cells (3 or 4), and the
        # remaining regions fill the rest of the board.
        if N < 4:
            return None

        small_ids = set(random.sample(range(N), k=4))
        large_ids = [rid for rid in range(N) if rid not in small_ids]

        small_sizes = {rid: random.choice([3, 4]) for rid in small_ids}
        small_total = sum(small_sizes.values())

        remaining = (N * N) - small_total
        if remaining <= 0:
            return None

        base = remaining // len(large_ids)
        extra = remaining % len(large_ids)
        large_sizes = [base] * len(large_ids)
        for i in range(extra):
            large_sizes[i] += 1
        random.shuffle(large_sizes)

        targets = [0] * N
        for rid in small_ids:
            targets[rid] = max(region_sizes[rid], small_sizes[rid])
        for rid, sz in zip(large_ids, large_sizes):
            targets[rid] = max(region_sizes[rid], sz)

        # Defensive: targets must exactly fill the board.
        if sum(targets) != (N * N):
            return None
        if any(targets[rid] >= 5 for rid in small_ids):
            return None

        size_targets = targets

    unassigned = {(r, c) for r in range(N) for c in range(N) if regions[r][c] == UNASSIGNED_ID}

    # Multi-source growth: randomly pick which region expands each step.
    # This guarantees all cells are assigned if the grid is connected.
    while unassigned:
        expandable = [
            rid
            for rid in range(N)
            if fronts[rid] and (size_targets is None or region_sizes[rid] < size_targets[rid])
        ]
        if not expandable:
            # If we're enforcing targets, don't allow overruns.
            if size_targets is not None:
                return None
            expandable = [rid for rid in range(N) if fronts[rid]]
            if not expandable:
                return None

        # Bias growth: encourage varied sizes (LinkedIn feel)
        # easy/normal: more variance; evil: slightly more even
        if mode == "lolly" and size_targets is not None:
            weights = []
            for rid in expandable:
                cap = max(0, size_targets[rid] - region_sizes[rid])
                weights.append((cap + 0.25) ** 1.15)
        elif mode in ("easy",):
            weights = [random.random() ** 2.0 for _ in expandable]
        elif mode == "normal":
            weights = [random.random() ** 1.7 for _ in expandable]
        elif mode == "idk_bruv":
            weights = [random.random() ** 1.55 for _ in expandable]
        else:
            weights = [random.random() ** 1.4 for _ in expandable]
        rid = random.choices(expandable, weights=weights, k=1)[0]

        r, c = fronts[rid].popleft()

        nbrs = []
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc, N) and regions[rr][cc] == UNASSIGNED_ID:
                nbrs.append((rr, cc))

        random.shuffle(nbrs)
        for rr, cc in nbrs[: max(1, len(nbrs))]:
            if size_targets is not None and region_sizes[rid] >= size_targets[rid]:
                break
            if (rr, cc) in unassigned:
                regions[rr][cc] = rid
                unassigned.remove((rr, cc))
                fronts[rid].append((rr, cc))
                region_sizes[rid] += 1

    # Post-pass: make borders more creative with swaps (connectivity-safe)
    # Because the assignment is already connected, small swaps that keep both sides connected are ok.
    if mode == "easy":
        jitter_steps = 40
    elif mode == "normal":
        jitter_steps = 70
    elif mode == "evil":
        jitter_steps = 140
    elif mode == "idk_bruv":
        jitter_steps = 160
    else:
        jitter_steps = 220

    def region_cells(rid: int) -> List[Cell]:
        out = []
        for i in range(N):
            for j in range(N):
                if regions[i][j] == rid:
                    out.append((i, j))
        return out

    def is_connected(rid: int) -> bool:
        cells = region_cells(rid)
        if not cells:
            return False
        start = cells[0]
        cellset = set(cells)
        q = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            for dr, dc in DIRS4:
                xx, yy = x + dr, y + dc
                if (xx, yy) in cellset and (xx, yy) not in seen:
                    seen.add((xx, yy))
                    q.append((xx, yy))
        return len(seen) == len(cells)

    def boundary_positions() -> List[Cell]:
        b = []
        for r in range(N):
            for c in range(N):
                a = regions[r][c]
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if in_bounds(rr, cc, N) and regions[rr][cc] != a:
                        b.append((r, c))
                        break
        return b

    for _ in range(jitter_steps):
        b = boundary_positions()
        if not b:
            break
        r, c = random.choice(b)
        a = regions[r][c]

        # pick neighboring region breg
        adj = []
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc, N) and regions[rr][cc] != a:
                adj.append((rr, cc))
        if not adj:
            continue
        rr, cc = random.choice(adj)
        breg = regions[rr][cc]

        # swap one boundary cell from each side (keeps sizes roughly stable)
        # choose another boundary cell in breg adjacent to a somewhere
        candidates = []
        for i in range(N):
            for j in range(N):
                if regions[i][j] != breg:
                    continue
                for dr, dc in DIRS4:
                    ii, jj = i + dr, j + dc
                    if in_bounds(ii, jj, N) and regions[ii][jj] == a:
                        candidates.append((i, j))
                        break
        if not candidates:
            continue

        i, j = random.choice(candidates)

        # do swap
        regions[r][c], regions[i][j] = regions[i][j], regions[r][c]

        # keep connectivity
        if not is_connected(a) or not is_connected(breg):
            regions[r][c], regions[i][j] = regions[i][j], regions[r][c]

    return regions


# ============================================================
# Deduction solver (propagation rules; no diagonals)
# ============================================================
def adjacent_to_any(cell: Cell, placed: Set[Cell]) -> bool:
    r, c = cell
    for rr, cc in placed:
        if abs(r - rr) <= 1 and abs(c - cc) <= 1:
            return True
    return False

def compute_domains(regions, N: int, assignment: Dict[int, int]) -> Dict[int, Set[int]]:
    used_cols = set(assignment.values())
    used_regions = set(regions[r][assignment[r]] for r in assignment)
    placed = {(r, assignment[r]) for r in assignment}

    domains: Dict[int, Set[int]] = {}
    for r in range(N):
        if r in assignment:
            continue
        poss = set()
        for c in range(N):
            if c in used_cols:
                continue
            rid = regions[r][c]
            if rid in used_regions:
                continue
            if adjacent_to_any((r, c), placed):
                continue
            poss.add(c)
        domains[r] = poss
    return domains


def compute_domains_with_bans(
    regions,
    N: int,
    assignment: Dict[int, int],
    banned_cells: Set[Cell],
) -> Dict[int, Set[int]]:
    used_cols = set(assignment.values())
    used_regions = set(regions[r][assignment[r]] for r in assignment)
    placed = {(r, assignment[r]) for r in assignment}

    domains: Dict[int, Set[int]] = {}
    for r in range(N):
        if r in assignment:
            continue
        poss = set()
        for c in range(N):
            if (r, c) in banned_cells:
                continue
            if c in used_cols:
                continue
            rid = regions[r][c]
            if rid in used_regions:
                continue
            if adjacent_to_any((r, c), placed):
                continue
            poss.add(c)
        domains[r] = poss
    return domains


def _region_bounds(regions, N: int) -> Dict[int, tuple[int, int, int, int]]:
    """Returns rid -> (min_r, max_r, min_c, max_c)."""
    b: Dict[int, tuple[int, int, int, int]] = {}
    for r in range(N):
        for c in range(N):
            rid = regions[r][c]
            if rid not in b:
                b[rid] = (r, r, c, c)
            else:
                r0, r1, c0, c1 = b[rid]
                b[rid] = (min(r0, r), max(r1, r), min(c0, c), max(c1, c))
    return b


def _apply_region_window_capacity_cols(
    regions,
    N: int,
    assignment: Dict[int, int],
    banned_cells: Set[Cell],
    *,
    max_window: int = 4,
) -> tuple[bool, bool]:
    """LinkedIn-style region-count deduction over column windows (sound).

    Uses region geometry (not candidate sets):
    - Regions fully contained within a window of columns MUST place their queen
      in those columns.
    - If #contained (unused) regions equals the number of available columns in
      the window, then any other (unused) region that intersects the window must
      place its queen outside -> ban its cells inside the window.

    Returns (ok, changed).
    """
    used_cols = set(assignment.values())
    used_regions = set(regions[r][assignment[r]] for r in assignment)
    bounds = _region_bounds(regions, N)

    changed = False
    cap = min(max_window, N)
    for k in range(2, cap + 1):
        for start in range(0, N - k + 1):
            window_cols = set(range(start, start + k))
            avail_cols = [c for c in window_cols if c not in used_cols]
            cap_cols = len(avail_cols)
            if cap_cols == 0:
                continue

            contained_unused: Set[int] = set()
            intersecting_unused: Set[int] = set()

            for rid, (r0, r1, c0, c1) in bounds.items():
                if rid in used_regions:
                    continue
                intersects = not (c1 < start or c0 > start + k - 1)
                if not intersects:
                    continue
                intersecting_unused.add(rid)
                contained = (c0 >= start) and (c1 <= start + k - 1)
                if contained:
                    contained_unused.add(rid)

            if len(contained_unused) > cap_cols:
                return False, changed

            if len(contained_unused) == cap_cols:
                for rid in intersecting_unused - contained_unused:
                    for r in range(N):
                        for c in window_cols:
                            if regions[r][c] == rid:
                                if (r, c) not in banned_cells:
                                    banned_cells.add((r, c))
                                    changed = True

    return True, changed


def _apply_region_window_capacity_rows(
    regions,
    N: int,
    assignment: Dict[int, int],
    banned_cells: Set[Cell],
    *,
    max_window: int = 4,
) -> tuple[bool, bool]:
    """Same as column window rule, but applied to windows of rows."""
    used_rows = set(assignment.keys())
    used_regions = set(regions[r][assignment[r]] for r in assignment)
    bounds = _region_bounds(regions, N)

    changed = False
    cap = min(max_window, N)
    for k in range(2, cap + 1):
        for start in range(0, N - k + 1):
            window_rows = set(range(start, start + k))
            avail_rows = [r for r in window_rows if r not in used_rows]
            cap_rows = len(avail_rows)
            if cap_rows == 0:
                continue

            contained_unused: Set[int] = set()
            intersecting_unused: Set[int] = set()

            for rid, (r0, r1, c0, c1) in bounds.items():
                if rid in used_regions:
                    continue
                intersects = not (r1 < start or r0 > start + k - 1)
                if not intersects:
                    continue
                intersecting_unused.add(rid)
                contained = (r0 >= start) and (r1 <= start + k - 1)
                if contained:
                    contained_unused.add(rid)

            if len(contained_unused) > cap_rows:
                return False, changed

            if len(contained_unused) == cap_rows:
                for rid in intersecting_unused - contained_unused:
                    for r in window_rows:
                        for c in range(N):
                            if regions[r][c] == rid:
                                if (r, c) not in banned_cells:
                                    banned_cells.add((r, c))
                                    changed = True

    return True, changed

def region_singletons(domains: Dict[int, Set[int]], regions, assignment: Dict[int, int]) -> List[Tuple[int, int]]:
    used_regions = set(regions[r][assignment[r]] for r in assignment)
    per: Dict[int, List[Tuple[int, int]]] = {}
    for r, cols in domains.items():
        for c in cols:
            rid = regions[r][c]
            if rid in used_regions:
                continue
            per.setdefault(rid, []).append((r, c))
    return [lst[0] for lst in per.values() if len(lst) == 1]

def column_singletons(domains: Dict[int, Set[int]]) -> List[Tuple[int, int]]:
    col_to_rows: Dict[int, List[int]] = {}
    for r, cols in domains.items():
        for c in cols:
            col_to_rows.setdefault(c, []).append(r)
    return [(rows[0], c) for c, rows in col_to_rows.items() if len(rows) == 1]

def assignment_consistent(regions, N: int, assignment: Dict[int, int]) -> bool:
    """
    True if the current partial assignment obeys:
      - at most one queen per column
      - at most one queen per region
      - no-touch constraint (including diagonals)
    (Rows are unique by construction because assignment keys are rows.)
    """
    cols_seen: Set[int] = set()
    regs_seen: Set[int] = set()
    placed: List[Cell] = []

    for r, c in assignment.items():
        if not (0 <= r < N and 0 <= c < N):
            return False

        if c in cols_seen:
            return False
        cols_seen.add(c)

        rid = regions[r][c]
        if rid in regs_seen:
            return False
        regs_seen.add(rid)

        placed.append((r, c))

    # no-touch
    for i in range(len(placed)):
        r1, c1 = placed[i]
        for j in range(i + 1, len(placed)):
            r2, c2 = placed[j]
            if touches(r1, c1, r2, c2):
                return False

    return True


def propagate(regions, N: int, assignment: Dict[int, int], *, max_window: int = 4) -> Tuple[bool, int]:
    forced = 0

    banned_cells: Set[Cell] = set()

    # If caller gave us an inconsistent partial assignment, fail fast.
    if not assignment_consistent(regions, N, assignment):
        return False, forced

    while True:
        domains = compute_domains_with_bans(regions, N, assignment, banned_cells)
        if any(len(v) == 0 for v in domains.values()):
            return False, forced

        progress = False

        # NOTE: assign forced moves *one-by-one* and validate immediately,
        # to avoid the "two forced rows pick same column" bug.
        row_forced = [(r, next(iter(cols))) for r, cols in domains.items() if len(cols) == 1]
        if row_forced:
            for r, c in row_forced:
                if r in assignment:
                    continue
                assignment[r] = c
                if not assignment_consistent(regions, N, assignment):
                    return False, forced
                forced += 1
                progress = True
            continue

        reg_forced = region_singletons(domains, regions, assignment)
        if reg_forced:
            for r, c in reg_forced:
                if r in assignment:
                    continue
                assignment[r] = c
                if not assignment_consistent(regions, N, assignment):
                    return False, forced
                forced += 1
                progress = True
            continue

        col_forced = column_singletons(domains)
        if col_forced:
            for r, c in col_forced:
                if r in assignment:
                    continue
                assignment[r] = c
                if not assignment_consistent(regions, N, assignment):
                    return False, forced
                forced += 1
                progress = True
            continue

        # LinkedIn-style region counting logic (geometry-based): eliminate cells.
        ok, changed_cols = _apply_region_window_capacity_cols(regions, N, assignment, banned_cells, max_window=max_window)
        if not ok:
            return False, forced
        ok, changed_rows = _apply_region_window_capacity_rows(regions, N, assignment, banned_cells, max_window=max_window)
        if not ok:
            return False, forced
        if changed_cols or changed_rows:
            progress = True
            continue

        if not progress:
            break

    return True, forced

def is_zero_guess_solvable(regions, N: int, *, max_window: int = 4) -> bool:
    a: Dict[int, int] = {}
    ok, _ = propagate(regions, N, a, max_window=max_window)
    return ok and len(a) == N

def forced_score(regions, N: int, *, max_window: int = 4) -> int:
    a: Dict[int, int] = {}
    ok, f = propagate(regions, N, a, max_window=max_window)
    return f if ok else 0


def analyze_deduction(regions, N: int, *, max_window: int = 4) -> tuple[int, bool]:
    """Returns (forced_moves_count, zero_guess_solved)."""
    a: Dict[int, int] = {}
    ok, f = propagate(regions, N, a, max_window=max_window)
    return (f if ok else 0), (ok and len(a) == N)

def count_solutions(regions, N: int, limit: int = 2) -> int:
    solutions = 0

    def dfs(assignment: Dict[int, int]):
        nonlocal solutions
        if solutions >= limit:
            return

        if not assignment_consistent(regions, N, assignment):
            return

        ok, _ = propagate(regions, N, assignment)
        if not ok:
            return

        if len(assignment) == N:
            # Double-check final assignment is truly legal.
            if assignment_consistent(regions, N, assignment):
                solutions += 1
            return

        domains = compute_domains(regions, N, assignment)
        r = min(domains.keys(), key=lambda rr: len(domains[rr]))
        opts = list(domains[r])
        random.shuffle(opts)

        for c in opts:
            if solutions >= limit:
                return
            a2 = dict(assignment)
            a2[r] = c
            dfs(a2)

    dfs({})
    return solutions


def find_one_solution(regions, N: int) -> Set[Cell] | None:
    """
    Returns one valid solution as a set of (row, col), or None if unsatisfiable.
    """
    found: List[Dict[int, int]] = []

    def dfs(assignment: Dict[int, int]):
        if found:
            return
        if not assignment_consistent(regions, N, assignment):
            return

        ok, _ = propagate(regions, N, assignment)
        if not ok:
            return

        if len(assignment) == N:
            found.append(dict(assignment))
            return

        domains = compute_domains(regions, N, assignment)
        r = min(domains.keys(), key=lambda rr: len(domains[rr]))
        opts = list(domains[r])
        random.shuffle(opts)
        for c in opts:
            a2 = dict(assignment)
            a2[r] = c
            dfs(a2)

    dfs({})
    if not found:
        return None
    a = found[0]
    return {(r, c) for r, c in a.items()}


# ============================================================
# Puzzle generation (never crashes)
# ============================================================
def generate_puzzle(
    N: int,
    mode: str,
    tries: int,
    target_forced: int,
    require_zero_guess: bool,
    prefer_zero_guess: bool = False,
    *,
    deduce_window: int = 4,
):
    best_any = None   # (forced, zero, regions, solution, void_rid)
    best_zero = None  # same, but only where zero==True
    thresholds = list(range(target_forced, max(0, target_forced - 5) - 1, -1))

    cfg = DIFFICULTIES.get(mode, {})
    time_limit_s = float(cfg.get("time_limit_s", 0.0) or 0.0)
    t0 = time.perf_counter()

    for t in thresholds:
        for _ in range(tries):
            if time_limit_s > 0.0 and (time.perf_counter() - t0) >= time_limit_s:
                break
            sol = generate_solution(N)
            if sol is None:
                continue

            void_size = int(cfg.get("void", 0) or 0)
            void_rid = None
            void_blob: Set[Cell] = set()
            if void_size > 0:
                void_rid = random.randrange(N)
                seed = sol[void_rid]
                void_blob = generate_negative_space_blob(N, void_size, seed)

            regions = generate_regions(sol, N, mode, void_blob=void_blob, void_rid=void_rid)
            if regions is None:
                continue

            # Defensive: verify the seeded solution is consistent.
            a_seed = {r: c for r, c in sol}
            if len(a_seed) != N or not assignment_consistent(regions, N, a_seed):
                continue

            # NOTE: We already have a valid solution by construction (we seed
            # each region with the row's queen cell), so searching for a
            # solution again here is extremely expensive and unnecessary.

            f, zero = analyze_deduction(regions, N, max_window=deduce_window)

            # track best (prefer zero-guess for "require_zero_guess" difficulties)
            cur = (f, zero, regions, set(sol), void_rid)
            if best_any is None or (f, zero) > (best_any[0], best_any[1]):
                best_any = cur
            if zero and (best_zero is None or f > best_zero[0]):
                best_zero = cur

            if require_zero_guess and not zero:
                continue
            if f < t:
                continue

            # If we're aiming for 0-guess boards (but not strictly requiring
            # them), keep searching until we find a 0-guess candidate.
            if prefer_zero_guess and not zero:
                continue

            return regions, set(sol), f, zero, True, void_rid

        if time_limit_s > 0.0 and (time.perf_counter() - t0) >= time_limit_s:
            break

    # fallback to best found
    if require_zero_guess and best_zero is not None:
        f, zero, regions, solset, void_rid = best_zero
        return regions, solset, f, zero, False, void_rid

    if prefer_zero_guess and best_zero is not None:
        f, zero, regions, solset, void_rid = best_zero
        return regions, solset, f, zero, False, void_rid

    # If we require deduction-only solvability, do NOT return a non-zero puzzle.
    # Spend a little extra time trying to find *any* zero-guess board.
    if require_zero_guess:
        # IMPORTANT: the main loop above may have already consumed the entire
        # `time_limit_s` budget. The rescue phase should get *additional* time
        # from now, otherwise it often runs zero iterations and immediately
        # raises.
        rescue_budget_s = max(2.0, time_limit_s)
        rescue_deadline = time.perf_counter() + rescue_budget_s
        while time.perf_counter() < rescue_deadline:
            sol = generate_solution(N)
            if sol is None:
                continue

            void_size = int(cfg.get("void", 0) or 0)
            void_rid = None
            void_blob: Set[Cell] = set()
            if void_size > 0:
                void_rid = random.randrange(N)
                seed = sol[void_rid]
                void_blob = generate_negative_space_blob(N, void_size, seed)

            regions = generate_regions(sol, N, mode, void_blob=void_blob, void_rid=void_rid)
            if regions is None:
                continue

            a_seed = {r: c for r, c in sol}
            if len(a_seed) != N or not assignment_consistent(regions, N, a_seed):
                continue

            f, zero = analyze_deduction(regions, N, max_window=deduce_window)
            if zero:
                return regions, set(sol), f, True, False, void_rid

        raise RuntimeError("Could not generate a deduction-solvable puzzle. Press R to retry.")

    if best_any is not None:
        f, zero, regions, solset, void_rid = best_any
        return regions, solset, f, zero, False, void_rid

    # ultimate fallback: just return something
    sol = generate_solution(N)
    if sol is None:
        raise RuntimeError("Could not generate a puzzle. Press R to retry.")
    regions = generate_regions(sol, N, mode)
    if regions is None:
        raise RuntimeError("Could not generate regions. Press R to retry.")
    return regions, set(sol), 0, False, False, None


# ============================================================
# Player validity + solved
# ============================================================
def get_invalid_queens(queens: Set[Cell], regions, N: int) -> Set[Cell]:
    invalid = set()
    qs = list(queens)

    row = [0] * N
    col = [0] * N
    reg: Dict[int, int] = {}

    for r, c in qs:
        row[r] += 1
        col[c] += 1
        rid = regions[r][c]
        reg[rid] = reg.get(rid, 0) + 1

    for r, c in qs:
        if row[r] > 1 or col[c] > 1 or reg.get(regions[r][c], 0) > 1:
            invalid.add((r, c))

    for i in range(len(qs)):
        r1, c1 = qs[i]
        for j in range(i + 1, len(qs)):
            r2, c2 = qs[j]
            if touches(r1, c1, r2, c2):
                invalid.add((r1, c1))
                invalid.add((r2, c2))

    return invalid

def is_solved(queens: Set[Cell], regions, N: int) -> bool:
    if len(queens) != N:
        return False
    if get_invalid_queens(queens, regions, N):
        return False

    rows = [0] * N
    cols = [0] * N
    regs: Dict[int, int] = {}

    for r, c in queens:
        rows[r] += 1
        cols[c] += 1
        rid = regions[r][c]
        regs[rid] = regs.get(rid, 0) + 1

    return all(v == 1 for v in rows) and all(v == 1 for v in cols) and len(regs) == N and all(v == 1 for v in regs.values())


# ============================================================
# Drawing
# ============================================================
def draw_grid_lines(screen, N, CELL):
    top = PAD + TOP_BAR
    left = PAD
    for i in range(N + 1):
        x = left + i * CELL
        pygame.draw.line(screen, (45, 45, 45), (x, top), (x, top + N * CELL), GRID_BORDER)
        y = top + i * CELL
        pygame.draw.line(screen, (45, 45, 45), (left, y), (left + N * CELL, y), GRID_BORDER)

def draw_region_borders(screen, regions, N, CELL):
    top = PAD + TOP_BAR
    left = PAD

    for r in range(N):
        for c in range(N - 1):
            if regions[r][c] != regions[r][c + 1]:
                x = left + (c + 1) * CELL
                y = top + r * CELL
                pygame.draw.line(screen, BLACK, (x, y), (x, y + CELL), REGION_BORDER)

    for r in range(N - 1):
        for c in range(N):
            if regions[r][c] != regions[r + 1][c]:
                x = left + c * CELL
                y = top + (r + 1) * CELL
                pygame.draw.line(screen, BLACK, (x, y), (x + CELL, y), REGION_BORDER)

    pygame.draw.rect(screen, BLACK, pygame.Rect(left, top, N * CELL, N * CELL), REGION_BORDER)

def draw_board(state):
    screen = state["screen"]
    N = state["N"]
    CELL = state["CELL"]
    W = state["W"]

    regions = state["regions"]
    queens = state["queens"]
    xmarks = state["xmarks"]
    region_colors = state["region_colors"]
    void_rid = state.get("void_rid")
    face_small = state["face_small"]
    queen_pad = state["queen_pad"]

    screen.fill(BG)
    pygame.draw.rect(screen, (235, 235, 235), pygame.Rect(0, 0, W, TOP_BAR + PAD))

    for r in range(N):
        for c in range(N):
            rid = regions[r][c]
            if void_rid is not None and rid == void_rid:
                pygame.draw.rect(screen, BG, cell_rect(r, c, CELL))
            else:
                pygame.draw.rect(screen, region_colors[rid], cell_rect(r, c, CELL))

    draw_grid_lines(screen, N, CELL)
    draw_region_borders(screen, regions, N, CELL)

    for (r, c) in xmarks:
        rect = cell_rect(r, c, CELL)
        cx, cy = rect.center
        s = int(CELL * 0.25)
        # Use a high-contrast dark color and explicit integer coordinates
        pygame.draw.line(screen, (30, 30, 30), (int(cx - s), int(cy - s)), (int(cx + s), int(cy + s)), 3)
        pygame.draw.line(screen, (30, 30, 30), (int(cx - s), int(cy + s)), (int(cx + s), int(cy - s)), 3)

    invalid = get_invalid_queens(queens, regions, N)
    for (r, c) in queens:
        rect = cell_rect(r, c, CELL)
        screen.blit(face_small, (rect.x + queen_pad, rect.y + queen_pad))
        if (r, c) in invalid:
            pygame.draw.rect(screen, ILLEGAL_RED, rect, 4)

    return invalid


# ============================================================
# State
# ============================================================
def build_state(diff_name: str):
    cfg = DIFFICULTIES[diff_name]
    N = cfg["N"]
    CELL = cfg["CELL"]

    W = PAD * 2 + N * CELL
    H = PAD * 2 + TOP_BAR + N * CELL

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Queens — Sinner Edition")

    queen_raw = load_fixed_queen_surface().convert_alpha()


    queen_pad = int(CELL * 0.10)
    queen_size = CELL - 2 * queen_pad
    face_small = pygame.transform.smoothscale(queen_raw, (queen_size, queen_size))

    base_win = int(min(W, H) * 0.34)
    # Placeholder: the real win image is selected when you solve.
    face_base = pygame.transform.smoothscale(queen_raw, (base_win, base_win))

    region_colors = pastel_palette(N)

    last_err: Exception | None = None
    regions = solution = forced = zero_guess = strict = void_rid = None
    # Generation can occasionally miss within the per-attempt time limit.
    # Instead of crashing after a fixed number of attempts, retry for a
    # bounded wall-clock budget.
    gen_t0 = time.perf_counter()
    budget_s = float(cfg.get("gen_budget_s", 0.0) or 0.0)
    if budget_s <= 0.0:
        budget_s = max(12.0, float(cfg.get("time_limit_s", 0.0) or 0.0) * 4.0)

    while (time.perf_counter() - gen_t0) < budget_s:
        try:
            regions, solution, forced, zero_guess, strict, void_rid = generate_puzzle(
                N=N, mode=diff_name, tries=cfg["tries"],
                target_forced=cfg["target_forced"],
                require_zero_guess=cfg["require_zero_guess"],
                prefer_zero_guess=bool(cfg.get("prefer_zero_guess", False)),
                deduce_window=int(cfg.get("deduce_window", 4) or 4),
            )
            break
        except RuntimeError as e:
            last_err = e
            continue

    if regions is None or solution is None:
        raise RuntimeError(str(last_err) if last_err else "Could not generate puzzle. Press R to retry.")

    if void_rid is not None and 0 <= void_rid < N:
        region_colors[void_rid] = BG

    win_x = random.randint(0, max(0, W - base_win))
    win_y = random.randint(0, max(0, H - base_win))
    vx = random.choice([-6, 6])
    vy = random.choice([-5, 5])

    return {
        "diff": diff_name,
        "N": N, "CELL": CELL, "W": W, "H": H,
        "screen": screen,

        "regions": regions,
        "solution": solution,
        "forced_score": forced,
        "zero_guess": zero_guess,
        "strict_ok": strict,

        "void_rid": void_rid,

        "region_colors": region_colors,
        "queens": set(),
        "xmarks": set(),
        "solved": False,

        "queen_pad": queen_pad,
        "face_small": face_small,

        "face_base": face_base,
        "win_credit": None,
        "base_win": base_win,
        "win_scale": 1.0,
        "win_scale_dir": 1.0,
        "win_x": win_x, "win_y": win_y,
        "vx": vx, "vy": vy,

        "win_played": False,

        "last_click_time": 0,
        "last_click_cell": None,

        # Left-drag paints X's
        "left_down": False,
        "left_down_cell": None,
        "left_down_pos": None,
        "drag_paint_x": False,
        "drag_seen": set(),
    }


# ============================================================
# Main
# ============================================================
state = build_state("normal")
running = True

while running:
    clock.tick(FPS)

    N = state["N"]
    CELL = state["CELL"]
    W = state["W"]
    H = state["H"]
    screen = state["screen"]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_r:
                if WIN_SOUND is not None:
                    WIN_SOUND.stop()
                state = build_state(state["diff"])

            elif event.key == pygame.K_c:
                state["queens"].clear()
                state["xmarks"].clear()
                state["solved"] = False
                state["win_played"] = False
                if WIN_SOUND is not None:
                    WIN_SOUND.stop()

            elif event.unicode == "1":
                if WIN_SOUND is not None:
                    WIN_SOUND.stop()
                state = build_state("easy")

            elif event.unicode == "2":
                if WIN_SOUND is not None:
                    WIN_SOUND.stop()
                state = build_state("normal")

            elif event.unicode == "3":
                if WIN_SOUND is not None:
                    WIN_SOUND.stop()
                state = build_state("evil")

            elif event.unicode == "4":
                if WIN_SOUND is not None:
                    WIN_SOUND.stop()
                state = build_state("lolly")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            board_rect = pygame.Rect(PAD, PAD + TOP_BAR, N * CELL, N * CELL)
            mods = pygame.key.get_mods()
            ctrl_down = (mods & pygame.KMOD_CTRL) != 0

            if not board_rect.collidepoint(mx, my):
                continue

            c = (mx - PAD) // CELL
            r = (my - (PAD + TOP_BAR)) // CELL
            cell = (r, c)

            # Right click or Ctrl+Left: place/remove a queen immediately.
            if event.button == 3 or (event.button == 1 and ctrl_down):
                if cell in state["queens"]:
                    state["queens"].remove(cell)
                else:
                    state["queens"].add(cell)
                    state["xmarks"].discard(cell)

                was = state["solved"]
                state["solved"] = is_solved(state["queens"], state["regions"], N)
                if state["solved"] and not was:
                    try:
                        win_raw = load_fixed_win_surface().convert_alpha()
                        base = state["base_win"]
                        state["face_base"] = pygame.transform.smoothscale(win_raw, (base, base))
                        state["win_credit"] = None
                    except Exception as e:
                        print(f"Win image load failed: {e}")

                    if WIN_SOUND is not None and not state["win_played"]:
                        WIN_SOUND.play()
                    state["win_played"] = True
                    state["win_scale"] = 1.0
                    state["win_scale_dir"] = 1.0
                continue

            # Left button: start a potential drag-to-paint-X gesture.
            if event.button == 1:
                state["left_down"] = True
                state["left_down_cell"] = cell
                state["left_down_pos"] = (mx, my)
                state["drag_paint_x"] = False
                state["drag_seen"] = set()


        elif event.type == pygame.MOUSEMOTION:
            # Dragging with left mouse paints X's.
            if not state.get("left_down"):
                continue
            if not getattr(event, "buttons", (0, 0, 0))[0]:
                continue

            mx, my = event.pos
            board_rect = pygame.Rect(PAD, PAD + TOP_BAR, N * CELL, N * CELL)
            if not board_rect.collidepoint(mx, my):
                continue

            c = (mx - PAD) // CELL
            r = (my - (PAD + TOP_BAR)) // CELL
            cell = (r, c)

            # Enter drag paint mode if we moved enough or changed cells.
            if not state.get("drag_paint_x"):
                down_pos = state.get("left_down_pos") or (mx, my)
                down_cell = state.get("left_down_cell")
                if down_cell != cell or abs(mx - down_pos[0]) + abs(my - down_pos[1]) >= 6:
                    state["drag_paint_x"] = True

            if not state.get("drag_paint_x"):
                continue

            if cell not in state["drag_seen"]:
                state["drag_seen"].add(cell)
                # Drag should not replace an existing queen.
                if cell not in state["queens"]:
                    state["xmarks"].add(cell)
                state["solved"] = is_solved(state["queens"], state["regions"], N)


        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button != 1:
                continue

            if not state.get("left_down"):
                continue

            cell = state.get("left_down_cell")
            state["left_down"] = False
            state["left_down_cell"] = None
            state["left_down_pos"] = None

            # If we were dragging, we already painted X's; don't toggle on release.
            if state.get("drag_paint_x"):
                state["drag_paint_x"] = False
                state["drag_seen"] = set()
                continue

            state["drag_seen"] = set()

            # Treat as a normal click (supports double-click-to-queen).
            if cell is None:
                continue

            now = pygame.time.get_ticks()
            is_double = (state["last_click_cell"] == cell and (now - state["last_click_time"]) <= DOUBLE_CLICK_MS)

            if is_double:
                if cell in state["queens"]:
                    state["queens"].remove(cell)
                else:
                    state["queens"].add(cell)
                    state["xmarks"].discard(cell)
            else:
                if cell in state["xmarks"]:
                    state["xmarks"].remove(cell)
                else:
                    state["xmarks"].add(cell)
                    state["queens"].discard(cell)

            state["last_click_cell"] = cell
            state["last_click_time"] = now

            was = state["solved"]
            state["solved"] = is_solved(state["queens"], state["regions"], N)

            if state["solved"] and not was:
                try:
                    win_raw = load_fixed_win_surface().convert_alpha()
                    base = state["base_win"]
                    state["face_base"] = pygame.transform.smoothscale(win_raw, (base, base))
                    state["win_credit"] = None
                except Exception as e:
                    print(f"Win image load failed: {e}")

                if WIN_SOUND is not None and not state["win_played"]:
                    WIN_SOUND.play()
                state["win_played"] = True
                state["win_scale"] = 1.0
                state["win_scale_dir"] = 1.0

    invalid = draw_board(state)

    draw_info_panel(state)
    # hint = "0-guess" if state["zero_guess"] else "may-guess"
    # strict = "strict" if state["strict_ok"] else "relaxed"
    # draw_text(
    #     screen,
    #     f"{state['diff'].upper()}  {N}×{N}  queens {len(state['queens'])}/{N} ({hint}, {strict})",
    #     PAD, 70, font, TEXT
    # )
    if invalid and not state["solved"]:
        draw_text(screen, "Illegal placement", PAD, TOP_BAR + 8, font, ILLEGAL_RED)

    if state["solved"]:
        # Fixed-size bouncing win image drawn on top of the board.
        size = int(state["base_win"])
        face_big = pygame.transform.smoothscale(state["face_base"], (size, size))

        x = state["win_x"] + state["vx"]
        y = state["win_y"] + state["vy"]

        if x <= 0 or x + size >= W:
            state["vx"] *= -1
            x = max(0, min(W - size, x))
        if y <= 0 or y + size >= H:
            state["vy"] *= -1
            y = max(0, min(H - size, y))

        state["win_x"], state["win_y"] = x, y
        screen.blit(face_big, (x, y))

    pygame.display.flip()

pygame.quit()


