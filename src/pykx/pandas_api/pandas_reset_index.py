from . import api_return
from ..exceptions import QError


def _init(_q):
    global q
    q = _q


class PandasResetIndex:
    @api_return
    def reset_index(self, levels=None, *, drop=False, inplace=False,
                    col_level=0, col_fill='', allow_duplicates=False,
                    names=None):
        """Reset keys/index of a PyKX Table"""
        if col_level != 0:
            raise QError("'col_level' not presently supported")
        if col_fill != '':
            raise QError("'col_fill' not presently supported")
        if names is not None:
            raise QError("'names' not presently supported")
        if 'Keyed' not in str(type(self)):
            return self
        if not allow_duplicates:
            if q('{any cols[key x] in cols value x}', self).py():
                raise QError('Cannot reset index due to duplicate column names')
        if drop and levels is None:
            return q.value(self)
        if levels is not None:
            intlist = False
            strlist = False
            if drop:
                drop_keys = q('{.[{(y _ key x)!value x};(x;y);{[x;y]value x}[x]]}')
            if isinstance(levels, list):
                strlist = all(isinstance(n, str) for n in levels)
                intlist = all(isinstance(n, int) for n in levels)
            if isinstance(levels, str) or strlist:
                q('''
                    {
                     if[any locs:not ((),x) in cols[key y];
                       '"Key(s) ",(", " sv string ((),x) where locs)," not found"
                       ]
                    }
                  ''',
                  levels,
                  self)
                if drop:
                    res = q('{x[y;(),z]}', drop_keys, self, levels)
                else:
                    res = q('{(cols[key x] except y) xkey x}', self, levels)
            elif isinstance(levels, int) or intlist:
                q('''
                  {
                    if[any locs:((),x)>ckeys:count cols key y;
                      '"Key level(s) ",
                       (", " sv string((),x)where locs),
                       " out of range ",
                       string ckeys
                      ]
                  }
                  ''',
                  levels,
                  self)
                if drop:
                    res = q('{x[y;(),cols[key y]z]]}', drop_keys, self, levels) # noqa: E501
                else:
                    res = q('{(cols[key x] except cols[key x]y) xkey x}', self, levels)
            else:
                raise TypeError("Unsupported type provided for 'levels'")
        else:
            res = q('0!', self)
        return res
