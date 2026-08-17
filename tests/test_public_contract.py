import json
from importlib.resources import files
from pathlib import Path

from rosetta_copy import list_variants


def test_variants_are_explicit_and_256d():
    assert list_variants() == ("s224", "s336")
    path = files("rosetta_copy").joinpath("configs", "variants.json")
    variants = json.loads(path.read_text(encoding="utf-8"))
    assert {row["input_side"] for row in variants.values()} == {224, 336}
    assert {row["descriptor_dimensions"] for row in variants.values()} == {256}
    assert all(len(row["sha256"]) == 64 for row in variants.values())
    assert all(row["url"].startswith("https://github.com/Grovanni/") for row in variants.values())


def test_published_checksum_file_matches_packaged_metadata():
    path = files("rosetta_copy").joinpath("configs", "variants.json")
    variants = json.loads(path.read_text(encoding="utf-8"))
    checksum_path = Path(__file__).parents[1] / "weights" / "SHA256SUMS.txt"
    published = {
        filename: digest
        for digest, filename in (
            line.split(maxsplit=1)
            for line in checksum_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    assert published == {
        config["filename"]: config["sha256"] for config in variants.values()
    }


def test_unknown_variant_has_an_actionable_error():
    from rosetta_copy.model import _variant_config

    try:
        _variant_config("unknown")
    except ValueError as error:
        assert "s224, s336" in str(error)
    else:
        raise AssertionError("unknown variants must be rejected")

