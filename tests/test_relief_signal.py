from src.evaluation.relief_signal_eval import RELIEF, relief_fraction


def test_relief_fraction_counts_relief_outcomes():
    outcomes = [
        "Closed with monetary relief",
        "Closed with explanation",
        "Closed with non-monetary relief",
        "Closed with explanation",
    ]
    assert relief_fraction(outcomes) == 0.5


def test_relief_fraction_empty_is_zero():
    assert relief_fraction([]) == 0.0


def test_relief_fraction_all_relief():
    assert relief_fraction(list(RELIEF)) == 1.0


def test_relief_fraction_no_relief():
    assert relief_fraction(["Closed with explanation", "Closed", "Untimely response"]) == 0.0
