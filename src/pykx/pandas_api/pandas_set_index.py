from ..wrappers import BooleanVector, IntVector, List, LongVector, ShortVector, SymbolAtom, SymbolVector # noqa
from . import api_return

import inspect

import pandas as pd


def _init(_q):
    global q
    q = _q


class PandasSetIndex:
    @api_return
    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        """Set the index using existing columns.

        Examples:
            >>> tab = kx.q('([] k:`a`b`c; v:1 2 3)')
            >>> tab.set_index('k')
            pykx.KeyedTable(pykx.q('
            k| v
            -| -
            a| 1
            b| 2
            c| 3
            '))
        """
        if(pd.core.indexes.multi.MultiIndex == type(keys) and not
           all(x is None for x in keys.names)):
            keys = q('{flip x!flip y}', list(keys.names), keys.values)
        if(not drop):
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only implemented for 'drop=True'") # noqa: E501
        self = q('''{[tab;kys;drop;append;verify_integrity]
                    keyed:99h~type tab;
                    if[-11h~type kys;kys:enlist kys];
                    reskeys:$[11h~type kys;
                              kys#$[keyed;value tab;tab];
                              kys];
                    if[keyed and append;reskeys:(key tab),'reskeys];
                    if[verify_integrity;
                       if[count where >[;1] count each group reskeys;
                        '"Index has duplicate key(s)"]
                       ];
                    resvals:$[11h~type kys;
                              $[drop;kys _ $[keyed;value tab;tab];0!tab];
                              0!tab];
                    reskeys!resvals
                }''', self, keys, drop, append, verify_integrity)
        return self

    @property
    def index(self):
        """The index of the table, which is the key of a keyed table or the row numbers of a table.

        Examples:
            >>> tab = kx.q('([] k:`a`b`c; v:1 2 3)')
            >>> tab.index
            pykx.LongVector(pykx.q('0 1 2'))
            >>> tab.set_index('k').index
            pykx.Table(pykx.q('
            k
            -
            a
            b
            c
            '))
        """
        key = q('{$[99h~type x;key x;til count x]}', self)
        return key
