"""Tests for scripts/make_sheet.py. All comments are made up and nothing touches the network."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_sheet  # noqa: E402
import scraper  # noqa: E402
import sentiment  # noqa: E402


def fake_pool(n=80, with_spam=True, with_long=True):
    rows = []
    for i in range(n):
        rows.append({"comment_id": f"yt-id-{i}", "video_id": "vid", "author_hash": f"author{i:04d}hash",
                     "text": f"made up comment number {i}", "like_count": i, "published_at": "2026-01-01T00:00:00Z"})
    if with_spam:
        rows.append({"comment_id": "yt-id-spam", "video_id": "vid", "author_hash": "authorspamhash",
                     "text": "made up promo see https://example.com now", "like_count": 0,
                     "published_at": "2026-01-01T00:00:00Z"})
    if with_long:
        rows.append({"comment_id": "yt-id-long", "video_id": "vid", "author_hash": "authorlonghash",
                     "text": "made up long comment " + "word " * 80, "like_count": 0,
                     "published_at": "2026-01-01T00:00:00Z"})
    return pd.DataFrame(rows, columns=scraper.COMMENT_COLUMNS)


@pytest.fixture
def fakes(monkeypatch):
    """Replace the YouTube call and the model call. Records what the fakes were given."""
    calls = {"video_ids": [], "limits": [], "model_rows": []}

    def fake_get_comments(video_id, max_comments=100):
        calls["video_ids"].append(video_id)
        calls["limits"].append(max_comments)
        return calls.get("pool", fake_pool())

    def fake_call_model(rows):
        calls["model_rows"].append(rows.copy())
        skip = calls.get("skip_index")
        labels = {i: ("generic", "neutral") for i in rows.index if i != skip}
        return labels, []

    monkeypatch.setattr(make_sheet.scraper, "get_comments", fake_get_comments)
    monkeypatch.setattr(make_sheet.sentiment, "_call_model", fake_call_model)
    monkeypatch.setenv("YOUTUBE_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    return calls


def run(tmp_path, *extra):
    return make_sheet.main(["--video", "8VL66ifp1Xc", "--out-dir", str(tmp_path / "data"), *extra])


# ---------- video link parsing ----------

@pytest.mark.parametrize("ref", [
    "https://youtu.be/8VL66ifp1Xc?si=SOMETRACKINGVALUE",
    "youtu.be/8VL66ifp1Xc",
    "https://www.youtube.com/watch?v=8VL66ifp1Xc&feature=youtu.be",
    "https://m.youtube.com/watch?v=8VL66ifp1Xc",
    "https://www.youtube.com/shorts/8VL66ifp1Xc",
    "https://www.youtube.com/embed/8VL66ifp1Xc",
    "https://www.youtube.com/live/8VL66ifp1Xc?feature=share",
    "8VL66ifp1Xc",
    "  8VL66ifp1Xc  ",
])
def test_parse_video_id_accepts_common_forms(ref):
    assert make_sheet.parse_video_id(ref) == "8VL66ifp1Xc"


@pytest.mark.parametrize("ref", ["", "not a link", "https://example.com/watch?v=8VL66ifp1Xc",
                                 "https://www.youtube.com/watch", "https://www.youtube.com/@somehandle",
                                 "https://youtu.be/short"])
def test_parse_video_id_rejects_other_input(ref):
    with pytest.raises(make_sheet.SheetError):
        make_sheet.parse_video_id(ref)


# ---------- sampling ----------

def test_sample_is_repeatable_with_the_same_seed_and_differs_with_another():
    pool = fake_pool()
    a = make_sheet.make_sample(pool, 50, 42)
    b = make_sheet.make_sample(pool, 50, 42)
    c = make_sheet.make_sample(pool, 50, 7)
    assert list(a["text"]) == list(b["text"])
    assert list(a["text"]) != list(c["text"])


def test_sample_ids_are_sequential_unique_and_padded():
    sample = make_sheet.make_sample(fake_pool(), 50, 42)
    assert list(sample["id"]) == [f"c{i:02d}" for i in range(1, 51)]
    assert sample["text"].is_unique


def test_sample_is_capped_at_the_pool_size():
    pool = fake_pool(n=10, with_spam=False, with_long=False)
    assert len(make_sheet.make_sample(pool, 50, 42)) == 10


# ---------- full run with fakes ----------

def test_run_writes_a_clean_sheet_and_hidden_tool_labels(tmp_path, fakes, capsys):
    assert run(tmp_path) == 0
    data = tmp_path / "data"

    sheet = pd.read_csv(data / "labeling_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(sheet.columns) == ["id", "comment", "my_label"]
    assert len(sheet) == 50
    assert (sheet["my_label"] == "").all()

    # No authors and no YouTube ids anywhere in the sheet.
    raw = (data / "labeling_sheet.csv").read_text(encoding="utf-8-sig")
    assert "author" not in raw.replace("authors", "")
    assert "yt-id-" not in raw

    tool = pd.read_csv(data / "tool_labels.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(tool.columns) == ["id", "tool_label", "tool_source"]
    assert list(tool["id"]) == list(sheet["id"])

    # The video link was reduced to its id, and the full pool size was requested.
    assert fakes["video_ids"] == ["8VL66ifp1Xc"]
    assert fakes["limits"] == [500]

    # The console output must not reveal any label to the person who is about to label.
    out = capsys.readouterr().out
    for category in sentiment.CATEGORIES:
        assert category not in out


def test_rule_spam_is_labeled_by_rules_and_skips_the_model(tmp_path, fakes):
    fakes["pool"] = fake_pool(n=60)
    assert run(tmp_path, "--n", "70") == 0   # 62 comments in the pool, so all of them are used
    data = tmp_path / "data"
    sheet = pd.read_csv(data / "labeling_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    tool = pd.read_csv(data / "tool_labels.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")

    spam_id = sheet.loc[sheet["comment"].str.contains("https://example.com"), "id"].iloc[0]
    row = tool[tool["id"] == spam_id].iloc[0]
    assert (row["tool_label"], row["tool_source"]) == ("spam", "rule")

    sent_to_model = pd.concat([f["text"] for f in fakes["model_rows"]])
    assert not sent_to_model.str.contains("https://example.com").any()
    assert (tool[tool["id"] != spam_id]["tool_source"] == "llm").all()


def test_same_seed_gives_the_same_sheet_in_two_runs(tmp_path, fakes):
    assert make_sheet.main(["--video", "8VL66ifp1Xc", "--out-dir", str(tmp_path / "one")]) == 0
    assert make_sheet.main(["--video", "8VL66ifp1Xc", "--out-dir", str(tmp_path / "two")]) == 0
    assert ((tmp_path / "one" / "labeling_sheet.csv").read_bytes()
            == (tmp_path / "two" / "labeling_sheet.csv").read_bytes())


def test_meta_records_the_method_without_any_comment_text(tmp_path, fakes):
    assert run(tmp_path, "--seed", "7") == 0
    meta_text = (tmp_path / "data" / "run_meta.json").read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert meta["source"] == "youtube:8VL66ifp1Xc"
    assert meta["seed"] == 7
    assert meta["n_sampled"] == 50
    assert meta["pool_size"] == 82
    assert meta["model"] == sentiment.MODEL
    assert meta["max_comment_chars_llm"] == sentiment.MAX_COMMENT_CHARS_LLM
    assert "made up" not in meta_text


def test_rerun_does_not_overwrite_a_labeled_sheet(tmp_path, fakes, capsys):
    assert run(tmp_path) == 0
    sheet_path = tmp_path / "data" / "labeling_sheet.csv"
    sheet_path.write_text("id,comment,my_label\nc01,made up,generic\n", encoding="utf-8")

    assert run(tmp_path) == 1
    assert "Already exists" in capsys.readouterr().err
    assert sheet_path.read_text(encoding="utf-8").endswith("c01,made up,generic\n")

    assert run(tmp_path, "--force") == 0
    assert "made up,generic" not in sheet_path.read_text(encoding="utf-8")


def test_missing_keys_stop_before_any_api_call(tmp_path, fakes, monkeypatch, capsys):
    monkeypatch.delenv("YOUTUBE_API_KEY")
    assert run(tmp_path) == 1
    assert "YOUTUBE_API_KEY" in capsys.readouterr().err

    monkeypatch.setenv("YOUTUBE_API_KEY", "not-a-real-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert run(tmp_path) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err

    assert fakes["video_ids"] == []
    assert not (tmp_path / "data").exists()


def test_a_comment_the_model_skipped_gets_an_empty_tool_label(tmp_path, fakes, capsys):
    fakes["pool"] = fake_pool(n=20, with_spam=False, with_long=False)
    fakes["skip_index"] = 3
    assert run(tmp_path, "--n", "20") == 0
    tool = pd.read_csv(tmp_path / "data" / "tool_labels.csv", dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")
    assert (tool["tool_label"] == "").sum() == 1
    assert "1 comments got no tool label" in capsys.readouterr().out


def test_model_failure_writes_nothing(tmp_path, fakes, monkeypatch, capsys):
    def boom(rows):
        raise sentiment.AnalysisError("The AI service returned an error: test")

    monkeypatch.setattr(make_sheet.sentiment, "_call_model", boom)
    assert run(tmp_path) == 1
    assert "The AI service returned an error" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_csv_mode_needs_no_youtube_key(tmp_path, fakes, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY")
    src = tmp_path / "comments.csv"
    pd.DataFrame({"comment": [f"made up csv comment {i}" for i in range(12)]}).to_csv(src, index=False)

    assert make_sheet.main(["--csv", str(src), "--n", "10", "--out-dir", str(tmp_path / "data")]) == 0
    sheet = pd.read_csv(tmp_path / "data" / "labeling_sheet.csv", dtype=str, encoding="utf-8-sig")
    assert len(sheet) == 10
    assert fakes["video_ids"] == []
    meta = json.loads((tmp_path / "data" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "csv:comments.csv"


def test_long_comments_are_counted_for_the_limits_section(tmp_path, fakes):
    fakes["pool"] = fake_pool(n=5, with_spam=False, with_long=True)
    assert run(tmp_path, "--n", "10") == 0
    meta = json.loads((tmp_path / "data" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["comments_longer_than_llm_limit"] == 1
