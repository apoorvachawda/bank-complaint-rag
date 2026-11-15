from src.evaluation.retrieval_eval import recall_at_k, reciprocal_rank


def test_recall_denominator_caps_at_k():
    # 8 ground-truth items, k=5: perfect top-5 must score 1.0, not 5/8
    ground_truth = {str(i) for i in range(8)}
    retrieved = [str(i) for i in range(8)]
    assert recall_at_k(ground_truth, retrieved, k=5) == 1.0


def test_recall_uses_gt_size_when_smaller_than_k():
    ground_truth = {"1", "2"}
    assert recall_at_k(ground_truth, ["1", "x", "y"], k=5) == 0.5


def test_recall_empty_ground_truth_is_zero():
    assert recall_at_k(set(), ["1"], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank({"a"}, ["x", "a", "y"]) == 0.5
    assert reciprocal_rank({"a"}, ["a"]) == 1.0
    assert reciprocal_rank({"a"}, ["x", "y"]) == 0.0
