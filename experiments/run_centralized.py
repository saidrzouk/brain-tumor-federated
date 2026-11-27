"""Run the existing centralized baseline from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.train_centralized import main 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the centralized VGG16 baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)
