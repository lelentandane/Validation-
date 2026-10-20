"""Tests for scripts/compare.py. All data here is made up."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compare  # noqa: E402
import sentiment  # noqa: E402


def write_csv(path, header, rows):
    def cell(v):
        return '"' + str(v).replace('"', '""') + '"'
    lines = [",".join(header)] + [",".join(cell(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def make_files(tmp_path, mine, tool, sources=None, comments=None):
    """mine and tool are lists of labels. Ids are c1, c2, ..."""
    n = len(mine)
    comments = comments or [f"made up comment {i + 1}" for i in range(n)]
    sheet = tmp_path / "sheet.csv"
    tool_file = tmp_path / "tool.csv"
    write_csv(sheet, ["id", "comment", "my_label"], [(f"c{i + 1}", comments[i], mine[i]) for i in range(n)])
    if sources is None:
        write_csv(tool_file, ["id", "tool_label"], [(f"c{i + 1}", tool[i]) for i in range(n)])
    else:
        write_csv(tool_file, ["id", "tool_label", "tool_source"],
                  [(f"c{i + 1}", tool[i], sources[i]) for i in range(n)])
    return sheet, tool_file


# 10 comments, 8 agree. Two disagreements: c5 (me question, tool generic) and c10 (me critical, tool toxic).
MINE = ["supportive"] * 4 + ["question", "generic", "generic", "spam", "spam", "critical"]
TOOL = ["supportive"] * 4 + ["generic", "generic", "generic", "spam", "spam", "toxic"]


def test_category_list_matches_the_classifier():
    assert compare.CATEGORIES == sentiment.CATEGORIES


def test_overall_agreement_and_mismatches(tmp_path):
    sheet, tool = make_files(tmp_path, MINE, TOOL)
    pairs = compare.load_pairs(sheet, tool)
    result = compare.score(pairs["scored"])
    assert result["n"] == 10
    assert result["agree"] == 8
    assert result["overall"] == pytest.approx(0.8)

    mismatches = compare.find_mismatches(pairs["scored"])
    # Listed in sheet order, even though "c10" sorts before "c5" as text.
    assert [(m["id"], m["mine"], m["tool"]) for m in mismatches] == [
        ("c5", "question", "generic"),
        ("c10", "critical", "toxic"),
    ]
    assert mismatches[0]["comment"] == "made up comment 5"


def test_per_category_recall_and_precision(tmp_path):
    sheet, tool = make_files(tmp_path, MINE, TOOL)
    result = compare.score(compare.load_pairs(sheet, tool)["scored"])
    per = result["per_category"]
    assert per["supportive"] == {"mine": 4, "tool": 4, "agree": 4, "recall": 1.0, "precision": 1.0}
    # I labeled 2 generic and the tool matched both; the tool labeled 3 generic and 2 were mine.
    assert per["generic"]["recall"] == pytest.approx(1.0)
    assert per["generic"]["precision"] == pytest.approx(2 / 3)
    # I labeled 1 question and the tool missed it; the tool never said question.
    assert per["question"]["recall"] == 0.0
    assert per["question"]["precision"] is None
    # Nobody used toxic as my label, so recall is undefined.
    assert per["toxic"]["recall"] is None
    assert per["toxic"]["precision"] == 0.0


def test_confusion_matrix_rows_are_my_labels(tmp_path):
    sheet, tool = make_files(tmp_path, MINE, TOOL)
    m = compare.score(compare.load_pairs(sheet, tool)["scored"])["matrix"]
    assert list(m.index) == compare.CATEGORIES
    assert list(m.columns) == compare.CATEGORIES
    assert m.loc["question", "generic"] == 1
    assert m.loc["critical", "toxic"] == 1
    assert m.loc["supportive", "supportive"] == 4
    assert int(m.values.sum()) == 10


def test_kappa_on_a_hand_checkable_case():
    mine = ["supportive"] * 5 + ["generic"] * 5
    tool = ["supportive"] * 4 + ["generic"] + ["generic"] * 4 + ["supportive"]
    # observed 0.8, chance 0.5, so kappa 0.6
    assert compare.cohens_kappa(mine, tool) == pytest.approx(0.6)
    assert compare.cohens_kappa(mine, mine) == pytest.approx(1.0)


def test_kappa_is_undefined_when_everything_is_one_category():
    assert compare.cohens_kappa(["generic"] * 4, ["generic"] * 4) is None
    assert compare.cohens_kappa([], []) is None


def test_wilson_interval():
    low, high = compare.wilson_interval(40, 50)
    assert low == pytest.approx(0.670, abs=0.005)
    assert high == pytest.approx(0.888, abs=0.005)
    assert compare.wilson_interval(0, 0) == (None, None)
    low, high = compare.wilson_interval(10, 10)
    assert high == 1.0 and low < 1.0


def test_invalid_and_blank_labels_are_flagged_and_not_scored(tmp_path):
    mine = ["supportive", "great", "", "Question ", "generic"]
    tool = ["supportive", "supportive", "generic", "question", ""]
    sheet, tool_file = make_files(tmp_path, mine, tool)
    pairs = compare.load_pairs(sheet, tool_file)

    flagged = dict(pairs["flagged"])
    assert set(flagged) == {"c2", "c3", "c5"}
    assert "'great' is not one of the 6 categories" in flagged["c2"][0]
    assert flagged["c3"] == ["my_label is blank"]
    assert flagged["c5"] == ["tool_label is blank"]

    # "Question " is trimmed and lowercased, so it is valid and it agrees.
    assert list(pairs["scored"]["id"]) == ["c1", "c4"]
    assert compare.score(pairs["scored"])["overall"] == 1.0


def test_ids_missing_from_one_file_are_reported(tmp_path):
    sheet, tool = make_files(tmp_path, ["generic"] * 3, ["generic"] * 3)
    # Drop c3 from the tool file and add an id that the sheet does not have.
    write_csv(tool, ["id", "tool_label"], [("c1", "generic"), ("c2", "generic"), ("c9", "spam")])
    pairs = compare.load_pairs(sheet, tool)
    assert pairs["only_in_sheet"] == ["c3"]
    assert pairs["only_in_tool"] == ["c9"]
    assert list(pairs["scored"]["id"]) == ["c1", "c2"]


def test_duplicate_ids_are_an_error(tmp_path):
    sheet = tmp_path / "sheet.csv"
    tool = tmp_path / "tool.csv"
    write_csv(sheet, ["id", "comment", "my_label"], [("c1", "a", "spam"), ("c1", "b", "spam")])
    write_csv(tool, ["id", "tool_label"], [("c1", "spam")])
    with pytest.raises(ValueError, match="Duplicate ids"):
        compare.load_pairs(sheet, tool)


def test_missing_column_is_an_error(tmp_path):
    sheet = tmp_path / "sheet.csv"
    tool = tmp_path / "tool.csv"
    write_csv(sheet, ["id", "comment"], [("c1", "a")])
    write_csv(tool, ["id", "tool_label"], [("c1", "spam")])
    with pytest.raises(ValueError, match="my_label"):
        compare.load_pairs(sheet, tool)


def test_mismatch_notes_when_the_model_only_saw_part_of_the_comment(tmp_path):
    comments = ["short", "x" * 30]
    sheet, tool = make_files(tmp_path, ["generic", "critical"], ["supportive", "generic"],
                             sources=["llm", "llm"], comments=comments)
    scored = compare.load_pairs(sheet, tool)["scored"]
    cut_flags = [m["cut"] for m in compare.find_mismatches(scored, llm_chars=10)]
    assert cut_flags == [False, True]
    # Rule-labeled rows never reach the model, so they are never marked as cut.
    sheet, tool = make_files(tmp_path, ["generic"], ["spam"], sources=["rule"], comments=["x" * 30])
    scored = compare.load_pairs(sheet, tool)["scored"]
    assert compare.find_mismatches(scored, llm_chars=10)[0]["cut"] is False


def test_main_prints_the_full_report(tmp_path, capsys):
    sheet, tool = make_files(tmp_path, MINE, TOOL, sources=["llm"] * 10,
                             comments=["made up comment with an emoji \U0001F600"] + [f"made up {i}" for i in range(9)])
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"max_comment_chars_llm": 250}), encoding="utf-8")

    code = compare.main(["--sheet", str(sheet), "--tool", str(tool), "--meta", str(meta)])
    out = capsys.readouterr().out

    assert code == 0
    assert "Overall agreement: 80.0% (8 of 10)" in out
    assert "Cohen's kappa:" in out
    assert "Agreement per category" in out
    assert "Confusion matrix (rows = my label, columns = tool label)" in out
    assert "Mismatches (2)" in out
    assert "c5  me: question  tool: generic" in out
    assert "made up 3" in out and "made up 8" in out   # mismatched comment text is shown


def test_main_exit_code_is_1_when_labels_are_flagged(tmp_path, capsys):
    sheet, tool = make_files(tmp_path, ["generic", "nonsense"], ["generic", "generic"])
    code = compare.main(["--sheet", str(sheet), "--tool", str(tool), "--meta", str(tmp_path / "none.json")])
    out = capsys.readouterr().out
    assert code == 1
    assert "FLAGGED (1" in out
    assert "Scored 1 of 2 paired comments." in out


def test_main_with_nothing_labeled_yet(tmp_path, capsys):
    sheet, tool = make_files(tmp_path, [""] * 3, ["generic"] * 3)
    code = compare.main(["--sheet", str(sheet), "--tool", str(tool), "--meta", str(tmp_path / "none.json")])
    out = capsys.readouterr().out
    assert code == 1
    assert "Nothing to score yet" in out


def test_main_reports_a_missing_file(tmp_path, capsys):
    code = compare.main(["--sheet", str(tmp_path / "nope.csv"), "--tool", str(tmp_path / "nope2.csv")])
    assert code == 2
    assert "file not found" in capsys.readouterr().err
