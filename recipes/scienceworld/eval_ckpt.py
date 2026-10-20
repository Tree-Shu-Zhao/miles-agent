"""Eval-only re-decoding of a saved checkpoint, or of the HF weights with --eval-base, under this recipe's protocol
(see miles_agent.eval_ckpt). RUN_SCRIPT=eval_ckpt.py EXTRA_ARGS="--recipe-script opd.py --eval-load-tag <arm> ..."."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root -> miles_agent, as run.py does
from miles_agent import eval_ckpt  # noqa: E402

if __name__ == "__main__":
    eval_ckpt.main(Path(__file__).resolve().parent)
