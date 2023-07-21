from ..wrappers import BooleanVector, IntVector, List, LongVector, ShortVector, SymbolAtom, SymbolVector # noqa
from . import api_return


import pandas as pd


def _init(_q):
    global q
    q = _q


class PandasSetIndex:
    @api_return
    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        """Set the index using existing columns."""
        if(pd.core.indexes.multi.MultiIndex == type(keys) and not
           all(x is None for x in keys.names)):
            keys = q('{flip x!flip y}', list(keys.names), keys.values)
        if(not drop):
            raise ValueError('nyi')
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
        key = q('{$[99h~type x;key x;til count x]}', self)
        return key
