from . import api_return
from ..wrappers import IntAtom, LongAtom, RealAtom, ShortAtom, SymbolAtom, SymbolVector


def _init(_q):
    global q
    q = _q


def process_keep(self, columns, n, keep, x):
    return q('''
            {[tab;cls;n;keep;x]
            cls:(),cls;
            kys:$[99h~type tab;cols key tab;`$()];
            s:$[x~`largest;xdesc;x~`smallest;xasc;'"Unknown sort option"];
            index:(::);
            if[keep~`last;
                s:$[x~`largest;xasc;xdesc];
                index:reverse;
                n:neg n;
                ];
            r:s[cls] update pykx_temp__internal_index:i from ?[tab;();0b;cls!cls];
            if[`all~keep;
                r:cls xgroup r;
                r:0!n sublist r;
                r:update pykx_temp__internal_running_sum:sums count each
                pykx_temp__internal_index from r;
                r:(1+(count[r]-1)^first where n <=r`pykx_temp__internal_running_sum) sublist r;
                i:count each r`pykx_temp__internal_index;
                :kys xkey (0!tab)raze r`pykx_temp__internal_index
            ];
            kys xkey (0!tab) index n sublist r`pykx_temp__internal_index}
            ''', self, columns, n, keep, x)


def check_column_types(self, cols):
    column_types = [SymbolAtom, SymbolVector, str]
    if isinstance(cols, list):
        if not all(type(c) in column_types for c in cols):
            raise ValueError('columns must be of type string, SymbolAtom or SymbolVector')
    elif type(cols) not in column_types:
        raise ValueError('columns must be of type string, SymbolAtom or SymbolVector')


def check_n(self, n):
    n_types = [ShortAtom, IntAtom, LongAtom, RealAtom, int]
    if type(n) not in n_types:
        raise ValueError("Only numeric values accepted for n")
    return True if n<1 else False


def nLargeSmall(self, n, order, columns=None, keep='first'):
    if check_n(self, n):
        return q.sublist(0, self)
    keep_options = ['first', 'last', 'all']
    if keep not in keep_options:
        raise ValueError('keep must be either "first", "last" or "all"')
    if keep != 'first':
        check_column_types(self, columns)
        return process_keep(self, columns, n, keep, order)
    asc = True if order == 'smallest' else False
    sorted = self.sort_values(by=columns, ascending=asc)
    return q('sublist', n, sorted)


class PandasSorting:

    @api_return
    def sort_values(self, by=None, ascending=True):
        check_column_types(self, by)
        if not isinstance(ascending, bool):
            raise ValueError(f"""For argument 'ascending' expected type bool,
                              received type {type(ascending)}.""")
        if ascending:
            if by is None:
                self = q('asc', self)
            else:
                self = q('xasc', by, self)
        else:
            if by is None:
                self = q('desc', self)
            else:
                self = q('xdesc', by, self)
        return self

    @api_return
    def nlargest(self, n, columns=None, keep='first'):
        return nLargeSmall(self, n, "largest", columns, keep)

    @api_return
    def nsmallest(self, n, columns, keep='first'):
        return nLargeSmall(self, n, "smallest", columns, keep)
