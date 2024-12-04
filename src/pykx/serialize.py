"""
Module to help with serializing K objects without copying the data.
"""
from typing import Any, Union

from ._wrappers import _to_bytes, decref, deserialize as _deserialize
from .wrappers import K


class serialize:
    def __init__(self, obj: Any, mode: int = 6, wait: int = 0):
        """Helper class to manage making 0 copy serialized `K` objects.

        Parameters:
            obj: The object to serialize.
            mode: The [capability level](https://code.kx.com/q/basics/ipc/#handshake)
                to use for serialization, defaults to the maximum value of 6.
            wait: The message type to use, defaults to 0.

        Note: The available message types to use are 0, 1, and 2.
            - 0: async
            - 1: sync
            - 2: response\n
            More information about the serialization of `K` objects can be found
            [here](https://code.kx.com/pykx/user-guide/advanced/serialization.html).

        Note: To access the memory view of the serialized object you can use the `data` property.
            If you need a copy of the data instead you can use the `copy` method.

        Warning: Passing just the `data` property of this class to a function may invalidate data.
+            This can be avoided by passing the whole `serialize` object instead.

        Examples:

        Serializing a `K` object and copying the serialized data.

        ```python
        >>> k_obj = kx.q('til 10')
        >>> ser = kx.serialize(k_obj)
        # The 0-copy memoryview of the data can be accessed through the `data` property
        >>> ser.data
        <memory at 0x7f92f05b4dc0>
        # The underlying bytes can be copied with the `copy` method.
        >>> k_obj_copy = ser.copy()
        >>> k_obj_copy
        b'\\x01\\x00\\x00\\x00^\\x00\\x00\\x00\\x07\\x00\\n\\x00\\x00...'
        ```

        You can also directly index into the serialized object.

        ```python
        >>> k_obj = kx.q('til 10')
        >>> ser = kx.serialize(k_obj)
        >>> ser[0]
        1
        >>> bytes(ser[0:5])
        b'\\x01\\x00\\x00\\x00^'
        ```
        """
        d = _to_bytes(mode, K(obj), wait)
        self.data = d[1]
        self._ptr = d[0]

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, key):
        return self.data[key]

    def __del__(self):
        decref(self._ptr)

    def copy(self):
        """Returns a copy of the bytes making up the serialized object.

        Examples:

        Serializing a `K` object and then copying the serialized data to a new variable.

        ```python
        >>> k_obj = kx.q('til 10')
        >>> ser = kx.serialize(k_obj)
        >>> k_obj_copy = ser.copy()
        >>> k_obj_copy
        b'\\x01\\x00\\x00\\x00^\\x00\\x00\\x00\\x07\\x00\\n\\x00\\x00...'
        ```
        """
        return bytes(self.data)


def deserialize(data: Union[bytes, serialize, memoryview]):
    """Helper method to deserialize `K` objects from bytes.

    Parameters:
        data: The object to deserialize.

    Examples:

    Deserialize a serialized `K` object.

    ```
    >>> k_obj = kx.q('til 10')
    >>> ser = kx.serialize(k_obj)
    >>> kx.deserialize(ser)
    pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
    ```

    You can also directly deserialize a bytes object.

    ```python
    >>> k_obj = kx.q('til 10')
    >>> ser = kx.serialize(k_obj).copy()
    >>> ser
    b'\\x01\\x00\\x00\\x00^\\x00\\x00\\x00\\x07\\x00\\n\\x00\\x00...'
    >>> kx.deserialize(ser)
    pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
    ```
    """
    if isinstance(data, serialize):
        return _deserialize(data.data.tobytes())
    elif isinstance(data, memoryview):
        return _deserialize(data.tobytes())
    return _deserialize(data)
