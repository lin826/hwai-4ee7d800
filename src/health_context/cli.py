"""Command line entry point: ``python -m health_context <command> ...``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .index import PatientIndex
from .pipeline import load_bundle
from .tools import catalog, run_tool


def _patient_files(data_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in data_dir.glob("*.json")
        if not p.name.startswith(("hospitalInformation", "practitionerInformation"))
    )


def _expand(paths: list[Path]) -> list[Path]:
    return [f for p in paths for f in (_patient_files(p) if p.is_dir() else [p])]


def _patient_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("patient", help="bundle file, or patient id with --store")
    p.add_argument("--store", type=Path, help="read from an ingested store")


def _llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", help="OpenAI-compatible server (env LLM_BASE_URL)")
    p.add_argument("--model", help="model name on that server (env LLM_MODEL)")


def _sample(files: list[Path], n: int) -> list[Path]:
    """Deterministic patient sample that always includes the README's patient."""
    ranked = sorted(
        files,
        key=lambda f: (
            not f.name.startswith("Merlene950_"),
            hashlib.sha256(f.name.encode()).hexdigest(),
        ),
    )
    return sorted(ranked[:n])


def _store_loader(store_path: Path):
    from .store import Store

    store = Store(store_path)
    return lambda path: store.load_index(store.patient_for_source(path))


def _load(args: argparse.Namespace) -> PatientIndex:
    if args.store:
        from .store import Store

        return Store(args.store).load_index(args.patient)
    return PatientIndex(load_bundle(args.patient))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="health_context")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalog", help="print the always-in-context patient catalog")
    _patient_arg(p)

    p = sub.add_parser("search", help="run search_records against one patient")
    _patient_arg(p)
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)

    p = sub.add_parser("timeline", help="run get_timeline for one concept id")
    _patient_arg(p)
    p.add_argument("concept_id")
    p.add_argument("--since")
    p.add_argument("--until")

    p = sub.add_parser("ask", help="answer a question with the local LLM and tools")
    _patient_arg(p)
    p.add_argument("question")
    _llm_args(p)

    p = sub.add_parser(
        "qa-eval", help="end-to-end eval: LLM answers known-answer questions"
    )
    p.add_argument("paths", type=Path, nargs="*", default=[Path("data")])
    p.add_argument("--patients", type=int, default=10, help="deterministic sample size")
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--store", type=Path)
    p.add_argument("--json", type=Path, help="write answers, tool traces and grades")
    _llm_args(p)

    p = sub.add_parser(
        "count", help="count catalogs with Anthropic count_tokens (claude-opus-5)"
    )
    p.add_argument("paths", type=Path, nargs="*", default=[Path("data")])

    p = sub.add_parser("ingest", help="index bundles into a Parquet store")
    p.add_argument("paths", type=Path, nargs="+", help="bundle files or directories")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--batch-mb", type=int, default=256)

    p = sub.add_parser(
        "sql", help="run SQL over a store (views: manifest, events, notes, resources)"
    )
    p.add_argument("query")
    p.add_argument("--store", type=Path, required=True)

    p = sub.add_parser(
        "rag-eval", help="retrieval eval over generated known-answer questions"
    )
    p.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=[Path("data")],
        help="bundle files or a data directory",
    )
    p.add_argument("--per-category", type=int, default=3)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--store", type=Path, help="score indexes loaded from this store")
    p.add_argument(
        "--json", type=Path, help="also write the full report, including failures"
    )

    args = parser.parse_args(argv)
    if args.command == "ingest":
        from .store import ingest

        report = ingest(_expand(args.paths), args.store, args.workers, args.batch_mb)
        print(json.dumps(report, indent=2))
        return
    if args.command == "count":
        from .counting import MAX_TOKENS, AnthropicCounter

        counter = AnthropicCounter()
        counts = []
        for path in _expand(args.paths):
            tokens = counter(catalog(PatientIndex(load_bundle(path))))
            counts.append(tokens)
            print(
                f"{tokens:>8}  {'ok' if tokens <= MAX_TOKENS else 'OVER'}  {path.name}"
            )
        print(
            f"{len(counts)} catalogs, max {max(counts)}, limit {MAX_TOKENS}, method {counter.method}"
        )
        return
    if args.command == "sql":
        from .store import Store

        for row in Store(args.store).sql(args.query):
            print("\t".join("" if v is None else str(v) for v in row))
        return
    if args.command == "rag-eval":
        from .rag_eval import format_report, run

        index_for = _store_loader(args.store) if args.store else None
        report = run(_expand(args.paths), args.per_category, args.k, index_for)
        print(format_report(report))
        if args.json:
            args.json.write_text(json.dumps(report, indent=2))
        return

    if args.command == "qa-eval":
        from .llm import ChatClient
        from .qa_eval import run as run_qa
        from .rag_eval import format_report

        client = ChatClient(args.base_url, args.model)
        files = _sample(_expand(args.paths), args.patients)
        index_for = _store_loader(args.store) if args.store else None
        report = run_qa(files, client, args.per_category, index_for, progress=print)
        print(f"model {report['model']} at {report['base_url']}")
        print(format_report(report))
        if args.json:
            args.json.write_text(json.dumps(report, indent=2))
        return

    index = _load(args)
    if args.command == "ask":
        from .agent import answer
        from .llm import ChatClient

        result = answer(index, args.question, ChatClient(args.base_url, args.model))
        for step in result.steps:
            print(f"> {step['name']}({json.dumps(step['arguments'])})")
        print(result.text)
        return
    if args.command == "catalog":
        print(catalog(index))
    elif args.command == "search":
        print(run_tool(index, "search_records", {"query": args.query, "k": args.k}))
    elif args.command == "timeline":
        print(
            run_tool(
                index,
                "get_timeline",
                {
                    "concept_id": args.concept_id,
                    "since": args.since,
                    "until": args.until,
                },
            )
        )
