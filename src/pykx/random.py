from typing import Any, List, Optional, Union

__all__ = [
    'random',
    'seed',
]


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q


def seed(seed: int) -> None:
    """Set random seed for PyKX random data generation

    Parameters:
        seed: Integer value defining the seed value to be set

    Returns:
        On successful invocation this function returns None
    """
    q('{system"S ",string x}', seed)


def random(dimensions: Union[int, List[int]],
           data: Any,
           seed: Optional[int] = None
) -> Any:
    """Return random data of specified dimensionality

    Parameters:
        dimensions: The dimensions of the data returned. Will produce a 1D array if single integer
        passed. Returns random data in shape of a list passed. Passing a negative value will perfom
        a kdb Deal on the data.

        data: The data from which a random sample is chosen. If an int or a float is passed,
        the random values are chosen in the range [0,data]. If a list is passed,
        the values are chosen from that list.

        seed: Denotes whether or not a seed should be used in the generation of data. Defaulted to
        None, any value passed will be used as a seed to generate the data.

    Returns:
        Randomised data in the shape specified by the 'dimensions' variable
    """

    if seed is not None:
        return q('''{pS:string system"S";
                    system"S ",string x;
                    r:.[{(0b;abs[x] # (prd x)?y)};(y;z);{(1b;x)}];
                    system"S ",pS;
                    if[r 0;'r 1];
                    r[1]}''', seed, dimensions, data)
    return q('{abs[x] # (prd x)?y}', dimensions, data)
