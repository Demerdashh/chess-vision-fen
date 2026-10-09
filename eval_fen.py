"""End-to-end exact-FEN evaluation for chess-vision-fen on a held-out test set.
Run from the repo root (next to main.py):
    python eval_fen.py --data fen_testset
Reports: board-detection rate, exact-FEN accuracy, per-square accuracy,
accuracy by orientation / theme, and median CPU-or-GPU latency per board."""

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from main import CLASSIFIER_WEIGHTS, DETECTOR_WEIGHTS, load_models, process_image_to_fen


def expand(placement: str) -> str:
    """'rn2k...' -> 64-char string with '.' for empty squares."""
    out = []
    for ch in placement.replace("/", ""):
        out.append("." * int(ch) if ch.isdigit() else ch)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fen_testset")
    ap.add_argument("--show-failures", type=int, default=10)
    a = ap.parse_args()
    data = Path(a.data)
    rows = list(csv.DictReader(open(data / "labels.csv", newline="")))

    detector, classifier = load_models(DETECTOR_WEIGHTS, CLASSIFIER_WEIGHTS)
    process_image_to_fen(str(data / rows[0]["file"]), detector, classifier)  # warm-up

    n_ok = n_exact = n_det = 0
    sq_correct = sq_total = 0
    times, fails = [], []
    by = defaultdict(lambda: [0, 0])  # key -> [exact, total]

    for r in rows:
        t0 = time.perf_counter()
        try:
            pred = process_image_to_fen(
                str(data / r["file"]), detector, classifier
            ).split()[0]
            n_det += 1
        except Exception as e:  # no board detected -> counts as a miss
            pred = None
            err = str(e)
        times.append((time.perf_counter() - t0) * 1000)

        truth = r["fen"]
        exact = pred == truth
        n_exact += exact
        for key in (f"orientation={r['orientation']}", f"theme={r['theme']}"):
            by[key][0] += exact
            by[key][1] += 1
        if pred:
            p, t = expand(pred), expand(truth)
            if len(p) == 64:
                sq_correct += sum(x == y for x, y in zip(p, t))
                sq_total += 64
        if not exact:
            fails.append(
                (r["file"], r["orientation"], pred or f"NO BOARD ({err})", truth)
            )

    n = len(rows)
    print(f"\nBoards evaluated      : {n}")
    print(f"Board detected        : {n_det}/{n} ({100 * n_det / n:.1f}%)")
    print(f"Exact-FEN accuracy    : {n_exact}/{n} ({100 * n_exact / n:.1f}%)")
    if sq_total:
        print(
            f"Per-square accuracy   : {100 * sq_correct / sq_total:.2f}%  (over detected boards)"
        )
    print(
        f"Latency per board     : median {statistics.median(times):.0f} ms, "
        f"p95 {sorted(times)[int(0.95 * len(times)) - 1]:.0f} ms  (incl. image read)"
    )
    print("\nBreakdown (exact-FEN):")
    for k in sorted(by):
        e, t = by[k]
        print(f"  {k:<28} {e}/{t} ({100 * e / t:.0f}%)")
    if fails:
        print(f"\nFirst {min(a.show_failures, len(fails))} failures:")
        for f in fails[: a.show_failures]:
            print(f"  {f[0]} [{f[1]}]\n    pred : {f[2]}\n    truth: {f[3]}")

    import json

    import torch

    Path("results").mkdir(exist_ok=True)
    Path("results/eval.json").write_text(
        json.dumps(
            {
                "test_set": f"{n} synthetic boards",
                "n": n,
                "exact": n_exact,
                "exact_fen_pct": 100 * n_exact / n,
                "per_square_pct": 100 * sq_correct / sq_total,
                "median_ms": statistics.median(times),
                "device": "GPU" if torch.cuda.is_available() else "CPU",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
