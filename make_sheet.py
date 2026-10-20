"""make_sheet.py - build a hand-labeling sheet and record the tool's labels for the same comments.

Writes three files into data/ (which is git-ignored):
  labeling_sheet.csv  id, comment, my_label (empty)   <- you fill in my_label
  tool_labels.csv     id, tool_label, tool_source     <- do not open until you have labeled
  run_meta.json       how the sample was made (no comment text)

The tool's labels come from the same steps the app runs (clean, rule-based spam,
then one model call). The classifier and its prompt are not changed.

Keys come from environment variables only:
  YOUTUBE_API_KEY    (not needed with --csv)
  ANTHROPIC_API_KEY

Examples:
  python scripts/make_sheet.py --video VIDEO_LINK_OR_ID
  python scripts/make_sheet.py --csv path/to/comments.csv
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import scraper    # noqa: E402
import sentiment  # noqa: E402
import utils      # noqa: E402

DEFAULT_N = 50
DEFAULT_SEED = 42
POOL_SIZE = 500   # the most get_comments will return
VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")

SHEET_NAME = "labeling_sheet.csv"
TOOL_NAME = "tool_labels.csv"
META_NAME = "run_meta.json"


class SheetError(Exception):
    pass


def parse_video_id(ref: str) -> str:
    """Accept a youtu.be link, a watch, shorts, embed or live link, or a bare 11 character id."""
    ref = (ref or "").strip()
    if VIDEO_ID_RE.match(ref):
        return ref

    parsed = urlparse(ref if "://" in ref else "https://" + ref)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    candidate = None
    if host == "youtu.be" and parts:
        candidate = parts[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parts and parts[0] == "watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        elif len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            candidate = parts[1]

    if candidate and VIDEO_ID_RE.match(candidate):
        return candidate
    raise SheetError("That does not look like a YouTube video link or id.")


def load_pool(video_id: str | None, csv_path: str | None) -> pd.DataFrame:
    """Same loading and cleaning the app uses."""
    if csv_path:
        comments = scraper.load_comments_csv(Path(csv_path).read_bytes())
    else:
        comments = scraper.get_comments(video_id, POOL_SIZE)
    return utils.clean_comments(comments)


def make_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Seeded random sample with fresh ids. Authors and platform ids are not carried over."""
    k = min(n, len(pool))
    sample = pool.sample(n=k, random_state=seed).reset_index(drop=True)
    width = max(2, len(str(k)))
    sample["id"] = [f"c{i + 1:0{width}d}" for i in range(k)]
    return sample


def classify_like_the_app(sample: pd.DataFrame) -> dict:
    """Return {row index: (label, source)}, following sentiment.analyze_comments step by step.

    Rule-based spam is labeled without a model call. Everything else goes to the model in
    one call. A comment the model did not label gets an empty label.
    """
    feats = utils.add_text_features(sample)
    rule_spam = feats["is_spam"].astype(bool)

    results = {idx: ("spam", "rule") for idx in feats.index[rule_spam]}
    to_llm = feats[~rule_spam]
    if not to_llm.empty:
        labels, _themes = sentiment._call_model(to_llm)
        for idx in to_llm.index:
            category = labels.get(idx, ("", ""))[0]
            results[idx] = (category, "llm")
    return results


def write_files(sample: pd.DataFrame, results: dict, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame({"id": sample["id"], "comment": sample["text"], "my_label": ""})
    tool = pd.DataFrame({
        "id": sample["id"],
        "tool_label": [results[i][0] for i in sample.index],
        "tool_source": [results[i][1] for i in sample.index],
    })
    # utf-8-sig so Excel shows emoji and accents correctly; compare.py reads it back the same way.
    sheet.to_csv(out_dir / SHEET_NAME, index=False, encoding="utf-8-sig")
    tool.to_csv(out_dir / TOOL_NAME, index=False, encoding="utf-8-sig")
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="YouTube link or video id")
    src.add_argument("--csv", help="CSV of comments (needs a text or comment column)")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="comments to sample (default 50)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default 42)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data", help="output folder (default data/)")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing files, INCLUDING a sheet you have already labeled")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.n < 1:
            raise SheetError("--n must be at least 1.")

        existing = [n for n in (SHEET_NAME, TOOL_NAME, META_NAME) if (args.out_dir / n).exists()]
        if existing and not args.force:
            raise SheetError("Already exists: " + ", ".join(existing)
                             + ". Re-running would replace your sheet and any labels in it. "
                               "Use --force only if you really want that.")

        if not args.csv and not os.getenv("YOUTUBE_API_KEY"):
            raise SheetError("Set the YOUTUBE_API_KEY environment variable first.")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SheetError("Set the ANTHROPIC_API_KEY environment variable first.")

        video_id = None if args.csv else parse_video_id(args.video)
        pool = load_pool(video_id, args.csv)
        if pool.empty:
            raise SheetError("No usable comments were found (comments may be turned off).")

        sample = make_sample(pool, args.n, args.seed)
        results = classify_like_the_app(sample)   # nothing is written until this succeeds
    except (SheetError, scraper.DataSourceError, sentiment.AnalysisError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    llm_chars = sentiment.MAX_COMMENT_CHARS_LLM
    long_for_model = int(sum(1 for i in sample.index
                             if results[i][1] == "llm" and len(sample.loc[i, "text"]) > llm_chars))
    missing = sum(1 for i in sample.index if not results[i][0])
    meta = {
        "source": f"youtube:{video_id}" if video_id else f"csv:{Path(args.csv).name}",
        "seed": args.seed,
        "n_requested": args.n,
        "n_sampled": len(sample),
        "pool_size": len(pool),
        "pool_cap": POOL_SIZE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": sentiment.MODEL,
        "max_comment_chars_llm": llm_chars,
        "comments_longer_than_llm_limit": long_for_model,
        "pandas_version": pd.__version__,
    }
    write_files(sample, results, meta, args.out_dir)

    print(f"Sampled {len(sample)} of {len(pool)} comments (seed {args.seed}).")
    if len(sample) < args.n:
        print(f"Note: the pool had fewer than {args.n} comments, so all of them were used.")
    print(f"Wrote {args.out_dir / SHEET_NAME}: fill in the my_label column only.")
    print(f"Wrote {args.out_dir / TOOL_NAME}: do not open it until every comment is labeled.")
    print(f"Wrote {args.out_dir / META_NAME}: no comment text.")
    print(f"{long_for_model} comments are longer than the {llm_chars} characters the model reads.")
    if missing:
        print(f"{missing} comments got no tool label. compare.py will flag them.")
    print("When you are done labeling, run: python scripts/compare.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
