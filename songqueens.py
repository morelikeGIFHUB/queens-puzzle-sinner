import os
import random
import pygame
import colorsys
from collections import deque
from typing import Dict, Set, Tuple, List

pygame.init()
clock = pygame.time.Clock()

# ---------------- Audio ----------------
SOUND_ON = True
try:
    pygame.mixer.init()
except pygame.error:
    SOUND_ON = False

WIN_SOUND = None
if SOUND_ON:
    win_path = os.path.join(os.path.dirname(__file__), "win.wav")
    if os.path.exists(win_path):
        try:
            WIN_SOUND = pygame.mixer.Sound(win_path)
        except pygame.error:
            WIN_SOUND = None

# ---------------- Visuals ----------------
FPS = 60
DOUBLE_CLICK_MS = 420

PAD = 24
TOP_BAR = 96
GRID_BORDER = 1
REGION_BORDER = 7

BG = (245, 245, 245)
TEXT = (20, 20, 20)
BLACK = (0, 0, 0)
X_COLOR = (0, 0, 0)  # Hot Pink
ILLEGAL_RED = (220, 40, 40)

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

IMG_FILENAME = "jannik_sinner_photos/Jannik Sinner(2).jpg"
IMG_PATH = os.path.join(os.path.dirname(__file__), IMG_FILENAME)
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(
        f"Couldn't find image file:\n  {IMG_PATH}\n\n"
        f"Put '{IMG_FILENAME}' in the same folder as songqueens.py"
    )
RAW_FACE_UNCONVERTED = pygame.image.load(IMG_PATH)

# Difficulty: mainly size + generation strictness
# (We still bias towards "more forced" but never crash.)
DIFFICULTIES = {
    "easy":   {"N": 6, "CELL": 84, "tries": 350, "target_forced": 6, "require_zero_guess": True},
    "normal": {"N": 7, "CELL": 78, "tries": 450, "target_forced": 6, "require_zero_guess": True},
    "evil":   {"N": 8, "CELL": 72, "tries": 650, "target_forced": 6, "require_zero_guess": True},
}

font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 46)

Cell = Tuple[int, int]


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

    for _ in range(3500):
        res = backtrack(0, [])
        if res is not None:
            return res
    return None


# ============================================================
# Region generator (GUARANTEED assignment)
#   - Assign each cell to a seed via randomized BFS growth
#   - Then do local swaps to create more jagged/creative borders
#   - Keeps all regions connected by construction
# ============================================================
def generate_regions(seed_cells: List[Cell], N: int, mode: str):
    # Start each region at its seed
    regions = [[-1 for _ in range(N)] for _ in range(N)]
    fronts = [deque() for _ in range(N)]

    for rid, (r, c) in enumerate(seed_cells):
        regions[r][c] = rid
        fronts[rid].append((r, c))

    unassigned = {(r, c) for r in range(N) for c in range(N) if regions[r][c] == -1}

    # Multi-source growth: randomly pick which region expands each step.
    # This guarantees all cells are assigned if the grid is connected.
    while unassigned:
        expandable = [rid for rid in range(N) if fronts[rid]]
        if not expandable:
            # Should not happen, but just in case: restart
            return None

        # Bias growth: encourage varied sizes (LinkedIn feel)
        # easy/normal: more variance; evil: slightly more even
        if mode == "easy":
            weights = [random.random() ** 2.0 for _ in expandable]
        elif mode == "normal":
            weights = [random.random() ** 1.7 for _ in expandable]
        else:
            weights = [random.random() ** 1.4 for _ in expandable]
        rid = random.choices(expandable, weights=weights, k=1)[0]

        r, c = fronts[rid].popleft()

        nbrs = []
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc, N) and regions[rr][cc] == -1:
                nbrs.append((rr, cc))

        random.shuffle(nbrs)
        for rr, cc in nbrs[: max(1, len(nbrs))]:
            if (rr, cc) in unassigned:
                regions[rr][cc] = rid
                unassigned.remove((rr, cc))
                fronts[rid].append((rr, cc))

    # Post-pass: make borders more creative with swaps (connectivity-safe)
    # Because the assignment is already connected, small swaps that keep both sides connected are ok.
    jitter_steps = 160 if mode == "easy" else (220 if mode == "normal" else 300)

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

def propagate(regions, N: int, assignment: Dict[int, int]) -> Tuple[bool, int]:
    forced = 0
    while True:
        domains = compute_domains(regions, N, assignment)
        if any(len(v) == 0 for v in domains.values()):
            return False, forced

        progress = False

        row_forced = [(r, next(iter(cols))) for r, cols in domains.items() if len(cols) == 1]
        if row_forced:
            for r, c in row_forced:
                if r not in assignment:
                    assignment[r] = c
                    forced += 1
                    progress = True
            continue

        reg_forced = region_singletons(domains, regions, assignment)
        if reg_forced:
            for r, c in reg_forced:
                if r not in assignment:
                    assignment[r] = c
                    forced += 1
                    progress = True
            continue

        col_forced = column_singletons(domains)
        if col_forced:
            for r, c in col_forced:
                if r not in assignment:
                    assignment[r] = c
                    forced += 1
                    progress = True
            continue

        if not progress:
            break

    return True, forced

def is_zero_guess_solvable(regions, N: int) -> bool:
    a: Dict[int, int] = {}
    ok, _ = propagate(regions, N, a)
    return ok and len(a) == N

def forced_score(regions, N: int) -> int:
    a: Dict[int, int] = {}
    ok, f = propagate(regions, N, a)
    return f if ok else 0

def count_solutions(regions, N: int, limit: int = 2) -> int:
    solutions = 0

    def dfs(assignment: Dict[int, int]):
        nonlocal solutions
        if solutions >= limit:
            return

        ok, _ = propagate(regions, N, assignment)
        if not ok:
            return

        if len(assignment) == N:
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


# ============================================================
# Puzzle generation (never crashes)
# ============================================================
def generate_puzzle(N: int, mode: str, tries: int, target_forced: int, require_zero_guess: bool):
    best = None  # (forced, zero, regions, solution)
    thresholds = list(range(target_forced, max(0, target_forced - 5) - 1, -1))

    for t in thresholds:
        for _ in range(tries):
            sol = generate_solution(N)
            if sol is None:
                continue

            regions = generate_regions(sol, N, mode)
            if regions is None:
                continue

            # prefer unique
            if count_solutions(regions, N, limit=2) != 1:
                continue

            zero = is_zero_guess_solvable(regions, N)
            f = forced_score(regions, N)

            # track best
            if best is None or (f, zero) > (best[0], best[1]):
                best = (f, zero, regions, set(sol))

            if require_zero_guess and not zero:
                continue
            if f < t:
                continue

            return regions, set(sol), f, zero, True

    # fallback to best found
    if best is not None:
        f, zero, regions, solset = best
        return regions, solset, f, zero, False

    # ultimate fallback: just return something
    sol = generate_solution(N)
    if sol is None:
        raise RuntimeError("Could not generate a puzzle. Press R to retry.")
    regions = generate_regions(sol, N, mode)
    if regions is None:
        raise RuntimeError("Could not generate regions. Press R to retry.")
    return regions, set(sol), 0, False, False


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
    face_small = state["face_small"]
    queen_pad = state["queen_pad"]

    screen.fill(BG)
    pygame.draw.rect(screen, (235, 235, 235), pygame.Rect(0, 0, W, TOP_BAR + PAD))

    for r in range(N):
        for c in range(N):
            rid = regions[r][c]
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

    raw = RAW_FACE_UNCONVERTED.convert_alpha()

    queen_pad = int(CELL * 0.10)
    queen_size = CELL - 2 * queen_pad
    face_small = pygame.transform.smoothscale(raw, (queen_size, queen_size))

    base_win = int(min(W, H) * 0.34)
    face_base = pygame.transform.smoothscale(raw, (base_win, base_win))

    region_colors = pastel_palette(N)

    sol = generate_solution(N)
    if sol is None:
        raise RuntimeError("Could not generate a solution. Press R to retry.")

    regions, solution, forced, zero_guess, strict = generate_puzzle(
        N=N, mode=diff_name, tries=cfg["tries"],
        target_forced=cfg["target_forced"],
        require_zero_guess=cfg["require_zero_guess"]
    )

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

        "region_colors": region_colors,
        "queens": set(),
        "xmarks": set(),
        "solved": False,

        "queen_pad": queen_pad,
        "face_small": face_small,

        "face_base": face_base,
        "base_win": base_win,
        "win_scale": 1.0,
        "win_scale_dir": 1.0,
        "win_x": win_x, "win_y": win_y,
        "vx": vx, "vy": vy,

        "win_played": False,

        "last_click_time": 0,
        "last_click_cell": None,
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

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            board_rect = pygame.Rect(PAD, PAD + TOP_BAR, N * CELL, N * CELL)
            if board_rect.collidepoint(mx, my):
                c = (mx - PAD) // CELL
                r = (my - (PAD + TOP_BAR)) // CELL
                cell = (r, c)

                mods = pygame.key.get_mods()
                ctrl_down = (mods & pygame.KMOD_CTRL) != 0

                if event.button == 3 or (event.button == 1 and ctrl_down):
                    if cell in state["queens"]:
                        state["queens"].remove(cell)
                    else:
                        state["queens"].add(cell)
                        state["xmarks"].discard(cell)

                elif event.button == 1:
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
                    if WIN_SOUND is not None and not state["win_played"]:
                        WIN_SOUND.play()
                    state["win_played"] = True
                    state["win_scale"] = 1.0
                    state["win_scale_dir"] = 1.0

    invalid = draw_board(state)

    draw_text(
        screen,
        "Left click=X   Right/Ctrl/Double=Queen   R=new   C=clear   1=Easy  2=Normal  3=Evil   ESC=quit",
        PAD, 14, font, TEXT
    )
    hint = "0-guess" if state["zero_guess"] else "may-guess"
    strict = "strict" if state["strict_ok"] else "relaxed"
    draw_text(
        screen,
        f"{state['diff'].upper()}  {N}×{N}  queens {len(state['queens'])}/{N}  forced={state['forced_score']} ({hint}, {strict})",
        PAD, 42, font, TEXT
    )
    if invalid and not state["solved"]:
        draw_text(screen, "Illegal placement", PAD, TOP_BAR + 8, font, ILLEGAL_RED)

    if state["solved"]:
        state["win_scale"] += 0.010 * state["win_scale_dir"]
        if state["win_scale"] >= 1.55:
            state["win_scale"] = 1.55
            state["win_scale_dir"] = -1.0
        elif state["win_scale"] <= 1.00:
            state["win_scale"] = 1.00
            state["win_scale_dir"] = 1.0

        base = state["base_win"]
        size = int(base * state["win_scale"])
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

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 70))
        screen.blit(overlay, (0, 0))

        screen.blit(face_big, (x, y))
        draw_text(screen, "Solved!", PAD, TOP_BAR + 8, big_font, (255, 255, 255))

    pygame.display.flip()

pygame.quit()


