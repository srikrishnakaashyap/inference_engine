"""Command-line entry point: `inference-engine` or `python -m inference_engine`."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate text from a prompt")
    gen.add_argument("--model-dir", required=True, help="Local model directory")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=int, default=64)
    gen.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        # TODO(you): construct the engine and run generation here.
        raise NotImplementedError("Engine not built yet (Phase 1+).")


if __name__ == "__main__":
    main()
