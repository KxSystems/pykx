from ..wrappers import List
from . import api_return

import inspect


def _init(_q):
    global q
    q = _q


class PandasApply:

    @api_return
    def apply(self, func, *args, axis: int = 0, raw=None, result_type=None, **kwargs):
        """Apply a function along an axis of the table.

        Parameters:
            func (callable): The function to apply to each column or row, which is passed as a
                `pykx` list.
            *args (Any): Positional arguments passed to `func` after the list.
            axis: 0 applies the function to each column and 1 to each row.
            raw (None): Not yet implemented, must be `None`.
            result_type (None): Not yet implemented, must be `None`.
            **kwargs (Any): Keyword arguments passed to `func`. Not supported when `func` is a q
                function.

        Returns:
            (List, Dictionary or Table): The result of `func` for each column or row, as a list,
                dictionary or table.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.apply(kx.q.sum)
            pykx.Dictionary(pykx.q('
            a| 6
            b| 15
            '))
        """
        if raw is not None:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only available for use when the 'raw' parameter is set to None") # noqa: E501
        if result_type is not None:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only available for use when the 'result_type' parameter is set to None") # noqa: E501
        if not callable(func):
            raise RuntimeError(f"Provided value '{func}' is not callable")

        if axis == 0:
            data = q.value(q.flip(self))
        else:
            data = q.flip(q.value(q.flip(self)))

        res = q(
            '{[f; tab; args; kwargs] '
            '  func: $[.pykx.util.isw f;'
            '    f[; pyarglist args; pykwargs kwargs];'
            '    ['
            '      if[0<count kwargs;'
            '        \'"ERROR: Passing key word arguments to q is not supported"'
            '      ];'
            '      {[data; f; args]'
            '        r: f[data];'
            '        $[104h~type r; r . args; r]'
            '      }[; f; args]'
            '    ]'
            '  ];'
            '  func each tab'
            '}',
            func,
            data,
            args,
            kwargs
        )

        if axis == 0:
            res = q('{k:cols[x]!y;@[flip;k;{[x;y]x}[k]]}', self, res)
        else:
            if isinstance(res, List):
                res = q('{flip cols[x]!flip y}', self, res)
        return res
