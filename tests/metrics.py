"""
Tests for metrics.py module.
"""
from metrics import is_permutation, nondecreasing


def test_permutation():
    a, b = [1, 2, 3, 4], [1, 2, 3, 4, 5]
    assert not is_permutation(a, b)
    a, b = [1, 2, 3, 4], [1, 2, 3, 4]
    assert is_permutation(a, b)
    a, b = [4, 3, 2, 1], [1, 2, 3, 4]
    assert is_permutation(a, b)
    a, b = [1, 3, 2, 4], [3, 2, 4, 1]
    assert is_permutation(a, b)
    a, b = [1, 1, 1, 2], [1, 2, 2, 2]
    assert not is_permutation(a, b)


def test_nondecreasing():
    a = [1, 2, 3, 4]
    assert nondecreasing(a) == 0
    a = [1, 1, 1, 1]
    assert nondecreasing(a) == 0
    a = [1, 2, 4, 3]
    assert nondecreasing(a) == 1
    a = [1, 3, 2, 4]
    assert nondecreasing(a) == 1
    a = [1, 4, 3, 2]
    assert nondecreasing(a) == 2


def run_tests():
    """
    Run all tests.
    """
    print("Test permutation")
    test_permutation()
    print("Test nondecreasing")
    test_nondecreasing()
