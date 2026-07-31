import json
from importlib.resources import files

from rosetta_copy import list_variants


def test_variants_are_explicit_and_256d():
    assert list_variants() == ("s224", "s336")
    path = files("rosetta_copy").joinpath("configs", "variants.json")
    variants = json.loads(path.read_text(encoding="utf-8"))
    assert {row["input_side"] for row in variants.values()} == {224, 336}
    assert {row["descriptor_dimensions"] for row in variants.values()} == {256}
    assert all(len(row["sha256"]) == 64 for row in variants.values())

