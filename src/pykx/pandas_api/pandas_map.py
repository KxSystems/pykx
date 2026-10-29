from . import api_return


def _init(_q):
    global q
    q = _q


class PandasMap:

    @api_return
    def map(self, func, na_action=None, *args, **kwargs):
        """Apply a function to every element of the table.

        A vectorized operation, such as `tab * 2`, is much faster than `map` with a function.

        Parameters:
            func (callable): The function to apply to each element.
            na_action (str): If `'ignore'` nulls are returned without being passed to `func`.
            *args (Any): Positional arguments passed to `func`.
            **kwargs (Any): Keyword arguments passed to `func`. Not supported when `func` is a q
                function.

        Returns:
            (Table): The table of the results.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.map(lambda x: x * 2)
            pykx.Table(pykx.q('
            a b
            ----
            2 8
            4 10
            6 12
            '))
        """
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

    def applymap(self, func, na_action=None, *args, **kwargs):
        """Apply a function to every element of the table. This is an alias of `map`.

        Parameters:
            func (callable): The function to apply to each element.
            na_action (str): If `'ignore'` nulls are returned without being passed to `func`.
            *args (Any): Positional arguments passed to `func`.
            **kwargs (Any): Keyword arguments passed to `func`. Not supported when `func` is a q
                function.

        Returns:
            (Table): The table of the results.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.applymap(lambda x: x * 2)
            pykx.Table(pykx.q('
            a b
            ----
            2 8
            4 10
            6 12
            '))
        """
        return self.map(func, na_action, *args, **kwargs)
