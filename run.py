"""
run.py — Jailbreak attack vs. inference-time defense evaluation harness.

PURPOSE:  Controlled, local, defensive evaluation for a university computer-security
course.  Harmful behaviors are drawn from the JailbreakBench public benchmark dataset
(https://jailbreakbench.github.io/) and are never hardcoded in this repository.
All inference runs against a locally-served Ollama instance; no data leaves the
machine.

IMPORTANT: The results/ directory contains model responses to sensitive prompts.
It is excluded from version control via .gitignore — do not commit it.
"""

import argparse
import csv
import json
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import ollama
from rich.console import Console
from rich.table import Table
from rich import box

import config
from model_client import ModelClient
from attacks import get_attack
from defenses import get_defense
from defenses.llama_guard import GuardClassifier
from judge import is_attack_success_refusal, is_attack_success_guard

console = Console()


# ── Startup checks ────────────────────────────────────────────────────────────

def check_ollama(base_url: str, required_models: list[str]) -> None:
    """Verify Ollama is reachable and all required models are pulled."""
    try:
        raw_client = ollama.Client(host=base_url)
        pulled = {m.model for m in raw_client.list().models}
    except Exception:
        console.print(
            f"\n[bold red]ERROR[/bold red]: Cannot connect to Ollama at {base_url}.\n"
            "Make sure Ollama is running.  Start it with:\n\n"
            "  [bold]ollama serve[/bold]\n"
        )
        sys.exit(1)

    missing = [m for m in required_models if m not in pulled]
    if missing:
        console.print("\n[bold red]ERROR[/bold red]: The following models are not pulled:\n")
        for m in missing:
            console.print(f"  [bold]ollama pull {m}[/bold]")
        console.print()
        sys.exit(1)


_Behavior = namedtuple("_Behavior", ["behavior_id", "behavior", "category"])


def load_behaviors(n: int) -> list:
    """Load the first N behaviors from the JailbreakBench dataset (v0.1.0 API).

    jailbreakbench 0.1.0 exposes three parallel lists:
      dataset.behaviors  — short name, e.g. "Defamation"
      dataset.goals      — full harmful request string (the actual prompt text)
      dataset.categories — category string
    We zip them into a lightweight namedtuple so the rest of the code is unchanged.
    """
    try:
        import jailbreakbench as jbb  # noqa: PLC0415
        dataset = jbb.read_dataset()
        records = [
            _Behavior(behavior_id=beh, behavior=goal, category=cat)
            for beh, goal, cat in zip(dataset.behaviors, dataset.goals, dataset.categories)
        ]
        return records[:n]
    except ImportError:
        console.print("[bold red]ERROR[/bold red]: jailbreakbench is not installed.  Run:\n\n"
                      "  pip install jailbreakbench\n")
        sys.exit(1)
    except Exception as exc:
        console.print(
            f"[bold red]ERROR[/bold red]: Failed to load JailbreakBench dataset: {exc}\n\n"
            "The dataset is fetched from the internet on first use and cached locally.\n"
            "Run once with internet access:\n\n"
            "  python -c \"import jailbreakbench as jbb; jbb.read_dataset()\"\n"
        )
        sys.exit(1)


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_condition(
    messages: list[dict],
    defense,
    classifier: GuardClassifier,
    verbose: bool = False,
) -> dict:
    """Run one (messages, defense) pair and return judge results."""
    if verbose:
        _print_messages(messages)

    response, meta = defense.query(messages)

    # Reuse guard label from defense metadata to avoid a redundant call.
    guard_label = meta.get("guard_label") or classifier.classify(messages, response)

    if verbose:
        _print_response(response)

    return {
        "response": response,
        "guard_label": guard_label,
        "success_refusal": is_attack_success_refusal(response),
        "success_guard": is_attack_success_guard(guard_label),
    }


def _print_messages(messages: list[dict]) -> None:
    console.print()
    for msg in messages:
        role = msg["role"].upper()
        color = {"system": "yellow", "user": "cyan", "assistant": "green"}.get(msg["role"], "white")
        console.rule(f"[{color}]{role}[/{color}]", style=color)
        console.print(msg["content"])


def _print_response(response: str) -> None:
    console.rule("[green]ASSISTANT[/green]", style="green")
    console.print(response)
    console.rule(style="dim")


def run(
    attack_name: str,
    defense_name: str,
    n: int,
    target_model: str,
    guard_model: str,
    base_url: str,
    verbose: bool = False,
) -> list[dict]:
    """Full evaluation: returns one record per behavior."""
    required = [target_model, guard_model]
    check_ollama(base_url, required)

    target_client = ModelClient(target_model, base_url)
    guard_client = ModelClient(guard_model, base_url)
    classifier = GuardClassifier(guard_client)

    attack = get_attack(attack_name)
    no_defense = get_defense("none", target_client)
    with_defense = get_defense(defense_name, target_client, guard_client)

    behaviors = load_behaviors(n)

    console.print(
        f"\n[bold]Jailbreak evaluation[/bold]  "
        f"attack=[cyan]{attack_name}[/cyan]  "
        f"defense=[cyan]{defense_name}[/cyan]  "
        f"model=[cyan]{target_model}[/cyan]  "
        f"N=[cyan]{n}[/cyan]\n"
    )

    records = []
    for idx, b in enumerate(behaviors):
        behavior_text = b.behavior
        behavior_id = getattr(b, "behavior_id", str(idx + 1))

        console.print(
            f"[dim]({idx + 1}/{n})[/dim] [bold]{behavior_id}[/bold]: "
            f"{behavior_text[:80]}{'…' if len(behavior_text) > 80 else ''}"
        )

        messages = attack.build_messages(behavior_text)

        # Condition A: attack, no defense
        if verbose:
            console.print("\n[bold yellow]── Condition A (no defense) ──[/bold yellow]")
        cond_a = evaluate_condition(messages, no_defense, classifier, verbose=verbose)

        # Condition B: attack + chosen defense
        if verbose:
            console.print(f"\n[bold yellow]── Condition B ({defense_name}) ──[/bold yellow]")
        cond_b = evaluate_condition(messages, with_defense, classifier, verbose=verbose)

        _log_condition("A (no defense)", cond_a)
        _log_condition(f"B ({defense_name})", cond_b)

        records.append({
            "behavior_id": behavior_id,
            "behavior": behavior_text,
            "prompt_sent": messages[0]["content"],
            # Condition A
            "a_guard_label": cond_a["guard_label"],
            "a_success_refusal": cond_a["success_refusal"],
            "a_success_guard": cond_a["success_guard"],
            "a_response": cond_a["response"],
            # Condition B
            "b_guard_label": cond_b["guard_label"],
            "b_success_refusal": cond_b["success_refusal"],
            "b_success_guard": cond_b["success_guard"],
            "b_response": cond_b["response"],
        })

    return records


def _log_condition(label: str, cond: dict) -> None:
    refused = not cond["success_refusal"]
    guard = cond["guard_label"]
    refused_str = "[green]YES[/green]" if refused else "[red]NO[/red]"
    guard_str = "[red]UNSAFE[/red]" if guard == "unsafe" else "[green]SAFE[/green]"
    console.print(f"    {label:<20} refused={refused_str}  guard={guard_str}")


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary_table(records: list[dict], defense_name: str) -> None:
    """Print a Rich before/after summary table."""
    table = Table(
        title="Jailbreak Attack vs. Defense — Summary",
        box=box.ROUNDED,
        show_footer=True,
    )

    n = len(records)

    # Column helpers
    def asr(values: list[bool]) -> str:
        rate = sum(values) / n if n else 0
        return f"{rate:.0%} ASR"

    a_ref = [r["a_success_refusal"] for r in records]
    a_grd = [r["a_success_guard"] for r in records]
    b_ref = [r["b_success_refusal"] for r in records]
    b_grd = [r["b_success_guard"] for r in records]

    table.add_column("Behavior ID", footer="AGGREGATE", style="bold", min_width=16)
    table.add_column("A: No-Refusal", footer=asr(a_ref), justify="center", min_width=14)
    table.add_column("A: Guard", footer=asr(a_grd), justify="center", min_width=14)
    table.add_column(f"B: No-Refusal\n({defense_name})", footer=asr(b_ref), justify="center", min_width=14)
    table.add_column(f"B: Guard\n({defense_name})", footer=asr(b_grd), justify="center", min_width=14)

    for r in records:
        table.add_row(
            r["behavior_id"],
            _bool_cell(r["a_success_refusal"]),
            _guard_cell(r["a_guard_label"]),
            _bool_cell(r["b_success_refusal"]),
            _guard_cell(r["b_guard_label"]),
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Columns: No-Refusal = model did not refuse (attack success by keyword check); "
        "Guard = Llama Guard flagged response as unsafe (attack success by guard).[/dim]\n"
    )


def _bool_cell(success: bool) -> str:
    return "[red]✓ success[/red]" if success else "[green]✗ blocked[/green]"


def _guard_cell(label: str) -> str:
    return "[red]UNSAFE[/red]" if label == "unsafe" else "[green]SAFE[/green]"


def save_results(records: list[dict], run_meta: dict) -> tuple[Path, Path]:
    """Write JSON and CSV result files; return (json_path, csv_path)."""
    out_dir = Path(config.RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"run_{ts}.json"
    csv_path = out_dir / f"run_{ts}.csv"

    payload = {"meta": run_meta, "results": records}
    json_path.write_text(json.dumps(payload, indent=2))

    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    return json_path, csv_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate jailbreak attacks vs. inference-time defenses.",
    )
    parser.add_argument("--attack", default=config.ATTACK,
                        help=f"Attack to use (default: {config.ATTACK})")
    parser.add_argument("--defense", default=config.DEFENSE,
                        help=f"Defense to use (default: {config.DEFENSE})")
    parser.add_argument("--n", type=int, default=config.N_BEHAVIORS,
                        help=f"Number of behaviors (default: {config.N_BEHAVIORS})")
    parser.add_argument("--model", default=config.TARGET_MODEL,
                        help=f"Target model (default: {config.TARGET_MODEL})")
    parser.add_argument("--guard-model", default=config.GUARD_MODEL,
                        help=f"Guard model (default: {config.GUARD_MODEL})")
    parser.add_argument("--ollama-url", default=config.OLLAMA_BASE_URL,
                        help=f"Ollama base URL (default: {config.OLLAMA_BASE_URL})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every prompt and response as it runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = run(
        attack_name=args.attack,
        defense_name=args.defense,
        n=args.n,
        target_model=args.model,
        guard_model=args.guard_model,
        base_url=args.ollama_url,
        verbose=args.verbose,
    )

    print_summary_table(records, args.defense)

    run_meta = {
        "attack": args.attack,
        "defense": args.defense,
        "n_behaviors": args.n,
        "target_model": args.model,
        "guard_model": args.guard_model,
        "timestamp": datetime.now().isoformat(),
    }
    json_path, csv_path = save_results(records, run_meta)
    console.print(f"[dim]Results saved to:[/dim]\n  {json_path}\n  {csv_path}\n")


if __name__ == "__main__":
    main()
