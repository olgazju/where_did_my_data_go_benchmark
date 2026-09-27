"""Command-line interface: `python -m poc benchmark <command>`.

Commands
--------
benchmark build    Offline: validation gates + task matrix + cost estimate (ZERO API calls).
benchmark run      Live: execute the benchmark (opt-in; requires ANTHROPIC_API_KEY).
benchmark replay   Offline: re-score preserved responses (ZERO API calls).
"""
from __future__ import annotations

import argparse

from .benchmark import cmd_build, cmd_run, cmd_replay
from .config import BenchmarkConfig

DEFAULT_CONFIG = "configs/benchmark.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poc", description="Where Did My Data Go? — benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="official P2 benchmark")
    bsub = p_bench.add_subparsers(dest="subcommand", required=True)
    for name, help_ in (("build", "offline validation + matrix + cost (zero API)"),
                        ("run", "live run (opt-in; needs ANTHROPIC_API_KEY)"),
                        ("replay", "re-score saved responses (zero API)")):
        sp = bsub.add_parser(name, help=help_)
        sp.add_argument("--config", default=DEFAULT_CONFIG)

    args = parser.parse_args(argv)
    cfg = BenchmarkConfig.load(args.config)
    return {"build": cmd_build, "run": cmd_run, "replay": cmd_replay}[args.subcommand](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
