import math


def ceildiv(a: float, b: float) -> int:
    if a == 0:
        return 0
    return math.ceil(a / b)
