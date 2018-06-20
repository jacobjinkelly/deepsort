"""
Metrics for measuring performance of models.
"""


def is_permutation(a, b):
    """
    Returns if a and b are permutations of one another.
    """
    if len(a) != len(b):
        return False
    member = [True for _ in range(len(a))]
    for item in a:
        found = False
        for i in range(len(b)):
            if item == b[i] and member[i]:
                member[i], found = False, True
                break
        if not found:
            return False
    return not any(member)


def nondecreasing(a):
    """
    Returns # of pairs of elements of a not in nondecreasing order.
    """
    incorrect = 0
    for i in range(len(a) - 1):
        if a[i] > a[i+1]:
            incorrect += 1
    return incorrect


if __name__ == "__main__":
    from tests.metrics import run_tests
    run_tests()
