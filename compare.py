"""compare.py - measure how often the tool's labels match labels written by hand.

Reads data/labeling_sheet.csv (id, comment, my_label) and data/tool_labels.csv
(id, tool_label, and optionally tool_source). Prints overall agreement, agreement
per category, a confusion matrix and every mismatch.

This script only reads files. It never calls an API and never changes a label.

Rows where either label is blank or not one of the 6 categories are flagged and
left out of the scores, so a typo cannot quietly count as a disagreement.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["supportive", "question", "critical", "toxic", "generic", "spam"]


# ---------- loading and pairing ----------

def _read(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
    df["id"] = df["id"].str.strip()
    return df


def _problem(side: str, raw: str) -> str:
    value = raw.strip().lower()
    if value == "":
        return f"{side} is blank"
    if value not in CATEGORIES:
        return f"{side} '{raw.strip()}' is not one of the 6 categories"
    return ""


def load_pairs(sheet_path: Path, tool_path: Path) -> dict:
    """Pair the two files by id.

    Returns {"scored": DataFrame, "flagged": [(id, [reasons])],
             "only_in_sheet": [ids], "only_in_tool": [ids]}.
    Labels are trimmed and lowercased before they are checked.
    """
    sheet = _read(Path(sheet_path), ["id", "comment", "my_label"])
    tool = _read(Path(tool_path), ["id", "tool_label"])
    if "tool_source" not in tool.columns:
        tool["tool_source"] = ""

    for name, df in (("labeling sheet", sheet), ("tool labels", tool)):
        dups = sorted(df.loc[df["id"].duplicated(), "id"].unique())
        if dups:
            raise ValueError(f"Duplicate ids in the {name}: {', '.join(dups)}")

    tool_ids, sheet_ids = set(tool["id"]), set(sheet["id"])
    only_in_sheet = [i for i in sheet["id"] if i not in tool_ids]
    only_in_tool = [i for i in tool["id"] if i not in sheet_ids]
    # An inner merge keeps the order of the sheet, so mismatches print in sheet order.
    both = sheet[["id", "comment", "my_label"]].merge(
        tool[["id", "tool_label", "tool_source"]], on="id", how="inner").reset_index(drop=True)

    flagged, keep = [], []
    for _, row in both.iterrows():
        reasons = [r for r in (_problem("my_label", row["my_label"]),
                               _problem("tool_label", row["tool_label"])) if r]
        if reasons:
            flagged.append((row["id"], reasons))
        keep.append(not reasons)

    scored = both[keep].copy().reset_index(drop=True)
    scored["my_label"] = scored["my_label"].str.strip().str.lower()
    scored["tool_label"] = scored["tool_label"].str.strip().str.lower()
    return {"scored": scored, "flagged": flagged,
            "only_in_sheet": only_in_sheet, "only_in_tool": only_in_tool,
            "n_paired": len(both)}


# ---------- statistics ----------

def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k out of n."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohens_kappa(mine: list[str], tool: list[str]):
    """Agreement beyond chance. None when chance agreement is already 100%."""
    n = len(mine)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(mine, tool)) / n
    chance = sum((mine.count(c) / n) * (tool.count(c) / n) for c in CATEGORIES)
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)


def score(scored: pd.DataFrame) -> dict:
    n = len(scored)
    matrix = (pd.crosstab(scored["my_label"], scored["tool_label"])
              .reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0).astype(int))
    agree = int((scored["my_label"] == scored["tool_label"]).sum())
    low, high = wilson_interval(agree, n)

    per_category = {}
    for c in CATEGORIES:
        mine, theirs, hit = int(matrix.loc[c].sum()), int(matrix[c].sum()), int(matrix.loc[c, c])
        per_category[c] = {
            "mine": mine, "tool": theirs, "agree": hit,
            "recall": hit / mine if mine else None,        # of what I labeled c, tool also said c
            "precision": hit / theirs if theirs else None,  # of what tool labeled c, I also said c
        }

    return {
        "n": n,
        "agree": agree,
        "overall": agree / n if n else None,
        "interval": (low, high),
        "kappa": cohens_kappa(scored["my_label"].tolist(), scored["tool_label"].tolist()),
        "per_category": per_category,
        "matrix": matrix,
    }


def find_mismatches(scored: pd.DataFrame, llm_chars: int | None = None) -> list[dict]:
    """Every row where the labels differ. 'cut' marks a comment the model only saw part of."""
    out = []
    for _, r in scored[scored["my_label"] != scored["tool_label"]].iterrows():
        cut = bool(llm_chars) and r["tool_source"] == "llm" and len(r["comment"]) > llm_chars
        out.append({"id": r["id"], "comment": r["comment"], "mine": r["my_label"],
                    "tool": r["tool_label"], "source": r["tool_source"], "cut": cut})
    return out


# ---------- printing ----------

def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def format_report(pairs: dict, result: dict, mismatches: list[dict], llm_chars: int | None = None) -> str:
    lines = []
    n_paired = pairs["n_paired"]

    if pairs["only_in_sheet"]:
        lines.append("Ids in the labeling sheet but not in tool_labels.csv (not scored): "
                     + ", ".join(pairs["only_in_sheet"]))
    if pairs["only_in_tool"]:
        lines.append("Ids in tool_labels.csv but not in the labeling sheet (not scored): "
                     + ", ".join(pairs["only_in_tool"]))
    if pairs["flagged"]:
        lines.append(f"FLAGGED ({len(pairs['flagged'])}, left out of the scores):")
        for cid, reasons in pairs["flagged"]:
            lines.append(f"  {cid}: " + "; ".join(reasons))
    if lines:
        lines.append("")

    n = result["n"]
    lines.append(f"Scored {n} of {n_paired} paired comments.")
    if n == 0:
        lines.append("Nothing to score yet. Fill in every my_label first.")
        return "\n".join(lines)

    low, high = result["interval"]
    kappa = result["kappa"]
    lines += [
        "",
        f"Overall agreement: {_pct(result['overall'])} ({result['agree']} of {n})",
        f"95% interval (Wilson): {_pct(low)} to {_pct(high)}",
        "Cohen's kappa: " + ("n/a" if kappa is None else f"{kappa:.2f}"),
        "",
        "Agreement per category",
        "  recall = of the comments I labeled X, share the tool also labeled X",
        "  precision = of the comments the tool labeled X, share I also labeled X",
        f"  {'category':<11}{'I labeled':>10}{'tool labeled':>14}{'both':>6}{'recall':>9}{'precision':>11}",
    ]
    for c in CATEGORIES:
        s = result["per_category"][c]
        lines.append(f"  {c:<11}{s['mine']:>10}{s['tool']:>14}{s['agree']:>6}"
                     f"{_pct(s['recall']):>9}{_pct(s['precision']):>11}")

    m = result["matrix"]
    lines += ["", "Confusion matrix (rows = my label, columns = tool label)",
              "  " + " " * 11 + "".join(f"{c:>11}" for c in CATEGORIES) + f"{'total':>8}"]
    for c in CATEGORIES:
        lines.append(f"  {c:<11}" + "".join(f"{int(m.loc[c, d]):>11}" for d in CATEGORIES)
                     + f"{int(m.loc[c].sum()):>8}")
    lines.append("  " + f"{'total':<11}" + "".join(f"{int(m[d].sum()):>11}" for d in CATEGORIES)
                 + f"{int(m.values.sum()):>8}")

    lines += ["", f"Mismatches ({len(mismatches)})"]
    for mm in mismatches:
        tags = []
        if mm["source"]:
            tags.append(f"tool source: {mm['source']}")
        if mm["cut"]:
            tags.append(f"model saw only the first {llm_chars} characters")
        tag = f"  [{'; '.join(tags)}]" if tags else ""
        lines.append(f"  {mm['id']}  me: {mm['mine']}  tool: {mm['tool']}{tag}")
        lines.append(f"      {mm['comment']}")
    return "\n".join(lines)


def _llm_chars_from_meta(meta_path: Path):
    try:
        return int(json.loads(Path(meta_path).read_text(encoding="utf-8"))["max_comment_chars_llm"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare hand labels with the tool's labels.")
    p.add_argument("--sheet", type=Path, default=ROOT / "data" / "labeling_sheet.csv")
    p.add_argument("--tool", type=Path, default=ROOT / "data" / "tool_labels.csv")
    p.add_argument("--meta", type=Path, default=ROOT / "data" / "run_meta.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Comments contain emoji; some Windows consoles cannot print them by default.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    try:
        pairs = load_pairs(args.sheet, args.tool)
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    llm_chars = _llm_chars_from_meta(args.meta)
    result = score(pairs["scored"])
    mismatches = find_mismatches(pairs["scored"], llm_chars)
    print(format_report(pairs, result, mismatches, llm_chars))
    has_problems = bool(pairs["flagged"] or pairs["only_in_sheet"] or pairs["only_in_tool"])
    return 1 if has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
