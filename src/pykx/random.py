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
    """Set random seed for PyKX random data generation.

    Parameters:
        seed: Integer value defining the seed value to be set

    Returns:
        On successful invocation this function returns None

    Examples:

    Set the random seed for generated data to 42 validating random generation is deterministic

    ```python
    >>> import pykx as kx
    >>> kx.random.seed(42)
    >>> kx.random.random(10, 10)
    pykx.LongVector(pykx.q('4 7 2 2 9 4 2 0 8 0'))
    >>> kx.random.random(42)
    >>> kx.random.random(10, 10)
    pykx.LongVector(pykx.q('4 7 2 2 9 4 2 0 8 0'))
    ```
    """
    q('{system"S ",string x}', seed)


def random(dimensions: Union[int, List[int]],
           data: Any,
           seed: Optional[int] = None
) -> Any:
    """Generate random data in the shape of the specified dimensions.

    Parameters:
        dimensions: The dimensions of the data returned. A a 1D array is produced if the input for
            this parameter is a single integer. A list input generates random data in the shape of
            the list. Passing a negative value performs a kdb Deal on the data.

        data: The data from which a random sample is chosen. Input an [int][int] or [float][float]
            to generate random values from the range [0,data]. Input a list to pick random values
            from that list.

        seed: Optional parameter to force randomisation to use a specific seed. Defaults to None.

    Returns:
        Randomised data in the shape specified by the 'dimensions' variable.

    Examples:

    Generate a random vector of floats between 0 and 10.5 of length 20

    ```python
    >>> import pykx as kx
    >>> kx.random.random(20, 10.5)
    pykx.FloatVector(pykx.q('5.233059 0.5785577 2.668026 4.834967 0.5733764..'))
    ```

    Generate a 1D generic list containing random values from a supplied list

    ```python
    >>> import pykx as kx
    >>> kx.random.random(20, ['a', 10, 1.5])
    pykx.List(pykx.q('
    10
    10
    1.5
    `a
    ..
    '))
    ```

    Generate a 2D generic list containing random long atoms between 0 and 100

    ```python
    >>> import pykx as kx
    >>> arr = kx.random.random([5, 5], 100)
    >>> arr
    pykx.List(pykx.q('
    67 46 30 29 61
    82 80 0  73 97
    92 75 38 28 94
    64 75 92 35 95
    81 45 44 59 49
    '))
    >>> arr[0]
    pykx.LongVector(pykx.q('67 46 30 29 61'))
    ```

    Generate a random vector of GUIDs using GUID null to generate full range with a defined seed

    ```python
    >>> import pykx as kx
    >>> kx.random.random(100, kx.GUIDAtom.null, seed=42)
    pykx.GUIDVector(pykx.q('84cf32c6-c711-79b4-2f31-6e85923decff 223..'))
    ```

    Using a negative value perform a "deal" returning non repeating values from a list of strings

    ```python
    >>> import pykx as kx
    >>> kx.random.random(-3, ['the', 'quick', 'brown', 'fox'])
    pykx.SymbolVector(pykx.q('`the`fox`brown'))
    ```
    """

    if seed is not None:
        return q('''{pS:string system"S";
                    system"S ",string x;
                    r:.[{(0b;abs[x] # (prd x)?y)};(y;z);{(1b;x)}];
                    system"S ",pS;
                    if[r 0;'r 1];
                    r[1]}''', seed, dimensions, data)
    return q('{abs[x] # (prd x)?y}', dimensions, data)
