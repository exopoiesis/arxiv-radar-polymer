"""Tests for tools/filter_corpus.py — apply polymer-context filter locally."""
import json
import sys

import pytest


def test_is_polymer_paper_glass_transition():
    import filter_corpus
    assert filter_corpus.is_polymer_paper("Glass transition temperature prediction for polymers.")


def test_is_polymer_paper_molecular_dynamics():
    import filter_corpus
    assert filter_corpus.is_polymer_paper("We perform molecular dynamics simulations of polymer melts.")


def test_is_polymer_paper_copolymer():
    import filter_corpus
    assert filter_corpus.is_polymer_paper("Block copolymer self-assembly controls morphology.")


def test_is_polymer_paper_polymer_stem():
    import filter_corpus
    # 'polymerization' should match via 'polymer' stem
    assert filter_corpus.is_polymer_paper("Radical polymerization of styrene.")


def test_is_polymer_paper_bigsmiles():
    import filter_corpus
    assert filter_corpus.is_polymer_paper("Generating BigSMILES strings with a transformer.")


def test_is_polymer_paper_polymer_llm():
    """LLM paper that's actually about polymers passes the filter."""
    import filter_corpus
    text = "We propose a Large Language Model for macromolecular property prediction."
    assert filter_corpus.is_polymer_paper(text)


def test_is_not_polymer_paper_pure_ml():
    """Pure ML/CS paper that mentions LLMs but no polymer context fails."""
    import filter_corpus
    text = "We propose a Large Language Model for code generation tasks."
    assert not filter_corpus.is_polymer_paper(text)


def test_is_not_polymer_paper_robotics():
    import filter_corpus
    text = "Diffusion models for robot motion planning in cluttered environments."
    assert not filter_corpus.is_polymer_paper(text)


def test_is_not_polymer_paper_image():
    import filter_corpus
    text = "Generative adversarial networks for high-resolution image synthesis."
    assert not filter_corpus.is_polymer_paper(text)


def test_is_polymer_paper_empty_abstract():
    import filter_corpus
    assert not filter_corpus.is_polymer_paper("")


def test_is_polymer_paper_no_false_positive_on_short_subword():
    """Polymer terms as substrings of unrelated words do not trigger."""
    import filter_corpus
    assert not filter_corpus.is_polymer_paper("Random text about networks and graphs.")
    assert not filter_corpus.is_polymer_paper("The aforementioned approach.")


def test_filter_corpus_writes_kept_papers(isolated_data_dir):
    """filter_corpus.run reads data/papers-*.json, writes to out_dir keeping
    only polymer-relevant papers."""
    import filter_corpus
    import data_io
    polymer_paper = {
        "title": "Polymer Tg prediction", "first_author": "A", "authors": ["A"],
        "abstract": "We predict glass transition temperature for polymer membranes.",
        "primary_category": "cond-mat.mtrl-sci",
        "categories": ["cond-mat.mtrl-sci"],
        "published": "2025-04-01", "updated": "2025-04-05",
        "comment": None, "pdf_url": "http://arxiv.org/pdf/2504.00100",
        "topics": ["Property Prediction & QSPR"], "tags": [],
    }
    noise_paper = {
        "title": "LLM for code", "first_author": "B", "authors": ["B"],
        "abstract": "We use Large Language Models for code completion in IDEs.",
        "primary_category": "cs.LG", "categories": ["cs.LG"],
        "published": "2025-04-10", "updated": "2025-04-12",
        "comment": None, "pdf_url": "http://arxiv.org/pdf/2504.00200",
        "topics": ["Large Language Models & Materials"], "tags": [],
    }
    by_month = {"2025-04": {"2504.00100": polymer_paper, "2504.00200": noise_paper}}
    data_io.save_month(by_month, "2025-04")

    out_dir = isolated_data_dir.root / "data_filtered"
    stats = filter_corpus.run(out_dir=out_dir)

    assert stats["kept"] == 1
    assert stats["dropped"] == 1
    out_file = out_dir / "papers-2025-04.json"
    assert out_file.exists()
    kept = json.loads(out_file.read_text(encoding="utf-8"))
    assert "2504.00100" in kept
    assert "2504.00200" not in kept


def test_filter_corpus_stats_per_topic(isolated_data_dir):
    """Stats report dropped count per topic — useful to see which topic is noisy."""
    import filter_corpus
    import data_io
    base = {
        "title": "x", "first_author": "x", "authors": ["x"],
        "abstract": "We propose a Large Language Model for code generation.",
        "primary_category": "cs.LG", "categories": ["cs.LG"],
        "published": "2025-04-01", "updated": "2025-04-05",
        "comment": None, "pdf_url": "http://arxiv.org/pdf/x",
        "tags": [],
    }
    by_month = {"2025-04": {
        "p1": {**base, "topics": ["Topic A"]},
        "p2": {**base, "topics": ["Topic A"]},
        "p3": {**base, "topics": ["Topic B"]},
    }}
    data_io.save_month(by_month, "2025-04")

    out_dir = isolated_data_dir.root / "data_filtered"
    stats = filter_corpus.run(out_dir=out_dir)

    assert stats["dropped_by_topic"]["Topic A"] == 2
    assert stats["dropped_by_topic"]["Topic B"] == 1
