"""Mkdocs plugin to make mkdocstrings filter out elements from its generated documentation.

Older versions of mkdocstrings do not support Cython projects. Newer versions do not yet have
element filtering implemented. This plugin exists as a temporary measure to enable filtering for
the PyKX docs until mkdocstrings officially supports filtering.
"""

from griffe.dataclasses import Alias
import mkdocs.plugins
from mkdocstrings.handlers.base import CollectorItem
from mkdocstrings_handlers.python.collector import PythonCollector

import pykx as kx


original_collect = PythonCollector.collect

wrappers_keep = {
    'AppliedIterator',
    'Atom',
    'BooleanAtom',
    'BooleanVector',
    'ByteAtom',
    'ByteVector',
    'CharAtom',
    'CharVector',
    'Collection',
    'Composition',
    'DateAtom',
    'DateVector',
    'DatetimeAtom',
    'DatetimeVector',
    'Dictionary',
    'Each',
    'EachLeft',
    'EachPrior',
    'EachRight',
    'EnumAtom',
    'EnumVector',
    'FloatAtom',
    'FloatVector',
    'Foreign',
    'Function',
    'GUIDAtom',
    'GUIDVector',
    'Identity',
    'IntAtom',
    'IntVector',
    'IntegralNumericAtom',
    'IntegralNumericVector',
    'Iterator',
    'K',
    'KeyedTable',
    'Lambda',
    'List',
    'LongAtom',
    'LongVector',
    'Mapping',
    'MinuteAtom',
    'MinuteVector',
    'MonthAtom',
    'MonthVector',
    'NonIntegralNumericAtom',
    'NonIntegralNumericVector',
    'NumericAtom',
    'NumericVector',
    'Operator',
    'Over',
    'PartitionedTable',
    'Projection',
    'ProjectionNull',
    'RealAtom',
    'RealVector',
    'Scan',
    'SecondAtom',
    'SecondVector',
    'ShortAtom',
    'ShortVector',
    'SplayedTable',
    'SymbolAtom',
    'SymbolVector',
    'Table',
    'TemporalAtom',
    'TemporalFixedAtom',
    'TemporalFixedVector',
    'TemporalSpanAtom',
    'TemporalSpanVector',
    'TemporalVector',
    'TimeAtom',
    'TimeVector',
    'TimespanAtom',
    'TimespanVector',
    'TimestampAtom',
    'TimestampVector',
    'UnaryPrimitive',
    'Vector',
    'py',
    'np',
    'pd',
    'pa',
}


exclude = {
    't',
    'ArrowUUIDType',
    'PandasUUIDArray',
    'PandasUUIDType',
    'atom_to_vector',
    'vector_to_atom',
    'reserved_words',
}


def collect(self, identifier: str, config: dict) -> CollectorItem:
    def g(k, in_wrappers):
        if in_wrappers:
            return k in wrappers_keep
        if k in {'__call__', '__init__'}: # special cases that start with an underscore
            return True
        return not k.startswith('_') and k not in exclude

    def f(x, in_wrappers=False):
        if not isinstance(x, Alias) and hasattr(x, 'members'):
            x.members = {k: f(v, in_wrappers or k == 'wrappers' or k in kx.wrappers.__all__)
                         for k, v in x.members.items() if g(k, in_wrappers)}
        return x

    return f(original_collect(self, identifier, config))


PythonCollector.collect = collect


class MkdocstringsFilter(mkdocs.plugins.BasePlugin):
    config_scheme = ()
    config = {}

    def load_config(self, *args, **kwargs):
        return (), ()
