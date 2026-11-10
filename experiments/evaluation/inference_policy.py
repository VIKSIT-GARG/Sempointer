"""Preserve one resource allowance when deriving an inference configuration."""
from pathlib import Path
import re


def shared_allowance_path(mount,manifest):
    ident=manifest.get("budget_account_run_id",manifest["run_id"])
    if not re.fullmatch(r"[0-9a-f]{20}",ident):raise ValueError("Invalid budget-account run ID")
    return Path(mount)/"runs"/ident/"allowance.json"


def development_spend(ledger):
    # Precision is charged to the total allowance, but does not consume the
    # separate $3 setup/development stop gate intended for the main suite.
    return sum(row["amount"] for key,row in ledger.data["attempts"].items()
               if not key.startswith("precision-"))/1_000_000
