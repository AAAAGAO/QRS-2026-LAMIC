from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import AppConfig
from .utils import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HARK project runner")
    parser.add_argument("command", choices=["train", "rq", "ui"], help="Command to run")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--library", default=None)
    parser.add_argument("--trained-output-dir", default=None)
    parser.add_argument("--rq-id", default=None)
    parser.add_argument("--rq4-query-library", default=None)
    parser.add_argument("--rq4-pool-library", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default="deepseek-chat")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--order-strategy", choices=["nearest_last", "nearest_first"], default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        library=args.library,
        trained_output_dir=Path(args.trained_output_dir) if args.trained_output_dir else None,
        rq_id=args.rq_id,
        rq4_query_library=args.rq4_query_library,
        rq4_pool_library=args.rq4_pool_library,
        rq_max_folds=args.max_folds,
        device=args.device,
    )
    config.icl.api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    config.icl.model_name = args.model_name
    config.icl.top_k = args.top_k
    if args.order_strategy is not None:
        config.icl.order_strategy = args.order_strategy
    config.icl.max_queries = args.max_queries
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.grad_accumulation_steps = args.grad_accumulation_steps
    config.training.seed = args.seed
    if args.n_splits is not None:
        config.split.n_splits = args.n_splits
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    if args.command == "train":
        from .experiments import train_retriever

        result = train_retriever(config)
        dump_json(result["summary"], config.output_dir / "train_summary.json")
        return
    if args.command == "rq":
        from .experiments import run_rq_experiment

        run_rq_experiment(config)
        return
    if args.command == "ui":
        webui_path = Path(__file__).with_name("webui.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(webui_path)], check=True)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 0:
                raise RuntimeError(
                    "UI 启动失败。请先确认当前解释器可以导入 streamlit。\n"
                    f"当前解释器: {sys.executable}\n"
                    f"可尝试执行: {sys.executable} -m pip install streamlit"
                ) from exc
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
