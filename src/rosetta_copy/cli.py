from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import RosettaEncoder, download_weights, list_variants


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosetta-copy",
        description="Rosetta NP compact image-copy descriptors",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download one checkpoint")
    download.add_argument("variant", choices=list_variants())
    download.add_argument("--cache-dir", type=Path)
    download.add_argument("--force", action="store_true")

    embed = subparsers.add_parser("embed", help="encode one or more images")
    embed.add_argument("variant", choices=list_variants())
    embed.add_argument("images", nargs="+", type=Path)
    embed.add_argument("--weights", type=Path)
    embed.add_argument("--cache-dir", type=Path)
    embed.add_argument("--device", default="cpu")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        path = download_weights(args.variant, args.cache_dir, force=args.force)
        print(path)
        return

    encoder = RosettaEncoder.from_pretrained(
        args.variant,
        weights=args.weights,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    vectors = encoder.encode(args.images, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors)
    print(f"saved {vectors.shape} {vectors.dtype} -> {args.output}")


if __name__ == "__main__":
    main()

