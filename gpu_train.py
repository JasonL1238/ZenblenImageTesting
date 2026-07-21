#!/usr/bin/env python
"""One-shot remote GPU training: pull -> train -> push weights back.

Cross-platform (Windows / Linux / Mac) — no bash needed. Run from anywhere in
the repo:

    python gpu_train.py                 # trains 'blended' on GPU 0
    python gpu_train.py chunk           # any mode
    python gpu_train.py blended 0       # mode + device (0 = first CUDA GPU, or 'cpu')

Modes: standard | spill | logo | chunk | unmixed | blended
Then on the Mac:  git pull  and copy the printed best.pt into active_pipeline/checkpoints/
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run(cmd, cwd=REPO, check=True):
    print(f">> {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "blended"
    device = sys.argv[2] if len(sys.argv) > 2 else "0"
    run_name = f"{mode}-nano-v1"
    best = REPO / "training" / "runs" / f"{mode}-seg" / run_name / "weights" / "best.pt"

    print(f">> [1/3] pull latest dataset + code")
    run(["git", "pull", "--ff-only"])

    print(f">> [2/3] train {mode} on device {device}")
    run([sys.executable, "train_multi.py", "--mode", mode, "--device", device],
        cwd=REPO / "training")

    if not best.exists():
        sys.exit(f"!! expected weights not found: {best}\n"
                 f"!! (did training finish? check training/runs/{mode}-seg/)")

    print(f">> [3/3] push weights back")
    run(["git", "add", str(best)])
    # commit returns nonzero if there is nothing to commit — that's fine.
    if run(["git", "commit", "-m", f"Train {mode}-seg on GPU (device {device})"],
           check=False).returncode != 0:
        print("nothing to commit (weights unchanged)")
        return
    run(["git", "push"])

    rel = best.relative_to(REPO).as_posix()
    print(f">> done. On the Mac:  git pull  then copy:\n"
          f"   {rel}  ->  active_pipeline/checkpoints/yolo_{mode}_seg.pt")


if __name__ == "__main__":
    main()
