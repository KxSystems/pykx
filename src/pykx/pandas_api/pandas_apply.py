from ..wrappers import List
from . import api_return


def _init(_q):
    global q
    q = _q


class PandasApply:

    @api_return
    def apply(self, func, *args, axis: int = 0, raw=None, result_type=None, **kwargs):
        if raw is not None:
            raise NotImplementedError("'raw' parameter not implemented, please set to None")
        if result_type is not None:
            raise NotImplementedError("'result_type' parameter not implemented, please set to None")
        if not callable(func):
            raise RuntimeError("Provided value 'func' is not callable")

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
