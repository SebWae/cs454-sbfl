def div(x: float, y: float) -> float: # divide x with y
    if y == 0:
        return 1.0
    else:
        return x / y

def my_sqrt(x: float) -> float:
    return abs(x) ** 0.5

def sbfl(ep: int, ef: int, np: int, nf: int) -> float:
    """
    My best performing risk evaluation formula for SBFL.
    """
    nominator = 2 * ef - 1
    denominator = my_sqrt(2 * nf + 2 * ep)
    
    return div(nominator, denominator)