from . import api_return


def _init(_q):
    global q
    q = _q


class PandasMap:

    @api_return
    def map(self, func, na_action=None, *args, **kwargs):
        if not callable(func):
            raise TypeError("Provided value 'func' is not callable")
        if na_action is not None:
            if na_action != 'ignore':
                raise TypeError("na_action must be None or 'ignore'")
        return q(
            '{[f; tab; args; kwargs;nan] '
            '  iskeyed:99h=type tab;'
            '  tab:$[iskeyed;[kt:key tab;value tab];tab];'
            '  func: $[.pykx.util.isw f;'
            '          f[; pyarglist args; pykwargs kwargs];'
            '          ['
            '            if[0<count kwargs;'
            '              \'"ERROR: Passing key word arguments to q is not supported"'
            '              ];'
            '            {[data; f; args]'
            '              r: f[data];'
            '              $[104h~type r; r . args; r]'
            '              }[; f; args]'
            '          ]'
            '          ];'
            '   $[iskeyed;kt!;]{[f;t;n]'
            '     ct:cols t;'
            '     lambda:$[n~`ignore;{$[(all/)null y;y;x y]}[f];f];'
            '     flip ct!{x each y}[lambda]each t ct}[func;tab;nan]'
            '  }',
            func,
            self,
            args,
            kwargs,
            na_action)

    applymap = map
