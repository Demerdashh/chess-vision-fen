"""Generate a held-out synthetic chess test set (images + exact ground-truth FEN).
Positions: random legal games (python-chess). Rendering: python-chess SVG -> PNG,
pasted at a random size/position on a plain background, to resemble a screenshot.
Fixed seed => fully reproducible. Includes both orientations and several board colors."""
import csv, io, random
from pathlib import Path
import chess, chess.svg, cairosvg
import numpy as np, cv2

SEED, N = 2026, 150
OUT = Path("fen_testset"); OUT.mkdir(exist_ok=True)
rng = random.Random(SEED)

THEMES = {  # light, dark square colors
    "brown":  ("#f0d9b5", "#b58863"),
    "green":  ("#eeeed2", "#769656"),
    "blue":   ("#dee3e6", "#8ca2ad"),
    "gray":   ("#e0e0e0", "#9a9a9a"),
}
BGS = [(255,255,255), (245,245,245), (40,40,40), (230,235,240)]

def random_position():
    b = chess.Board()
    for _ in range(rng.randint(10, 90)):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over(): break
        b.push(rng.choice(moves))
    return b

rows = []
for i in range(N):
    b = random_position()
    theme = rng.choice(list(THEMES)); light, dark = THEMES[theme]
    flipped = rng.random() < 0.5
    coords = rng.random() < 0.5
    svg = chess.svg.board(b, size=640, coordinates=coords,
        orientation=chess.BLACK if flipped else chess.WHITE,
        colors={"square light": light, "square dark": dark,
                "margin": "#ffffff"})
    png = cairosvg.svg2png(bytestring=svg.encode())
    board = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    side = rng.randint(380, 640)
    board = cv2.resize(board, (side, side), interpolation=cv2.INTER_AREA)
    W, H = rng.randint(side + 120, 1280), rng.randint(side + 80, 900)
    canvas = np.full((H, W, 3), rng.choice(BGS)[::-1], np.uint8)
    x, y = rng.randint(20, W - side - 20), rng.randint(20, H - side - 20)
    canvas[y:y+side, x:x+side] = board
    name = f"board_{i:03d}.png"
    cv2.imwrite(str(OUT / name), canvas)
    rows.append([name, b.board_fen(), "black_bottom" if flipped else "white_bottom", theme, int(coords)])

with open(OUT / "labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "fen", "orientation", "theme", "coords"]); w.writerows(rows)
print("wrote", len(rows), "images to", OUT)
