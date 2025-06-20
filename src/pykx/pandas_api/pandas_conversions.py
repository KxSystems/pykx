from . import api_return
from ..exceptions import QError

import inspect

type_number_to_pykx_k_type = {-128: 'QError',
                              -20: 'EnumAtom',
                              -19: 'TimeAtom',
                              -18: 'SecondAtom',
                              -17: 'MinuteAtom',
                              -16: 'TimespanAtom',
                              -15: 'DatetimeAtom',
                              -14: 'DateAtom',
                              -13: 'MonthAtom',
                              -12: 'TimestampAtom',
                              -11: 'SymbolAtom',
                              -10: 'CharAtom',
                              -9: 'FloatAtom',                                                     # noqa
                              -8: 'RealAtom',                                                      # noqa
                              -7: 'LongAtom',                                                      # noqa
                              -6: 'IntAtom',                                                       # noqa
                              -5: 'ShortAtom',                                                     # noqa
                              -4: 'ByteAtom',                                                      # noqa
                              -2: 'GUIDAtom',                                                      # noqa
                              -1: 'BooleanAtom',                                                   # noqa
                              0: 'List',                                                          # noqa
                              1: 'BooleanVector',                                                 # noqa
                              2: 'GUIDVector',                                                    # noqa
                              4: 'ByteVector',                                                    # noqa
                              5: 'ShortVector',                                                   # noqa
                              6: 'IntVector',                                                     # noqa
                              7: 'LongVector',                                                    # noqa
                              8: 'RealVector',                                                    # noqa
                              9: 'FloatVector',                                                   # noqa
                              10: 'CharVector',                                                    # noqa
                              11: 'SymbolVector',                                                  # noqa
                              12: 'TimestampVector',                                               # noqa
                              13: 'MonthVector',                                                   # noqa
                              14: 'DateVector',                                                    # noqa
                              15: 'DatetimeVector',                                                # noqa
                              16: 'TimespanVector',                                                # noqa
                              17: 'MinuteVector',                                                  # noqa
                              18: 'SecondVector',                                                  # noqa
                              19: 'TimeVector',                                                    # noqa
                              20: 'EnumVector',                                                    # noqa
                              77: 'Anymap',                                                        # noqa
                              98: '_k_table_type',                                                 # noqa
                              99: '_k_dictionary_type',                                            # noqa
                              100: 'Lambda',
                              101: '_k_unary_primitive',
                              102: 'Operator',
                              103: 'Iterator',
                              104: 'Projection',
                              105: 'Composition',
                              106: 'Each',
                              107: 'Over',
                              108: 'Scan',
                              109: 'EachPrior',
                              110: 'EachRight',
                              111: 'EachLeft',
                              112: 'Foreign'}

kx_type_to_type_number = {v: k for k, v in type_number_to_pykx_k_type.items()}


def _init(_q):
    global q
    q = _q


class PandasConversions:
    @api_return
    def astype(self, dtype, copy=True, errors='raise'): # noqa: max-complexity: 13

        try:
            if copy is not True:
                raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only implemented when copy is set to True") # noqa: E501

            # Check if input is scalar or str --> run q code per this input
            if isinstance(dtype, dict):
                # Check to ensure all columns provided are present in the table
                if not all([x in self.columns for x in dtype.keys()]):
                    raise ValueError('Column name passed in dictionary not present in df table')

                try:
                    dict_grab = {}
                    for k, v in dtype.copy().items():
                        dict_grab[k] = abs(kx_type_to_type_number[[x for x in
                                           kx_type_to_type_number.keys() if x in str(v)][0]])
                        strval = q('{3_first x}',
                                   q.qsql.exec(self.dtypes, 'datatypes', f'columns=`{k}'))
                        qtype = kx_type_to_type_number[strval.py().decode('utf-8')]
                        if abs(qtype) == dict_grab[k]:
                            dict_grab.pop(k)
                    if dict_grab == {}:
                        return self
                except IndexError:
                    raise QError('Value passed does not match PyKX wrapper type')

                # Check needed to ensure no non string nested cols in table to convert
                check_mixed_columns = q('''{[tab;dict]
                                            dict:5h$dict;
                                            kd:key dict;
                                            tab:?[tab;();0b;kd!kd];
                                            tabCols:cols tab;
                                            tabColTypes:value tabColTypesDict:type each flip 0#tab;
                                            if[not any 0h=tabColTypes; :0b];
                                            tabColNestedTypes:value tabColTypesDict,distinct each
                                            type each\'flip #[;tab] where 0h=tabColTypesDict;
                                            dictCols:key dict;
                                            tabBool:any tabCols in/: dictCols;
                                            tabCols:tabCols where tabBool;
                                            dict:tabCols!(dict tabCols);
                                            dictCols:tabCols;
                                            tabColTypes:tabColTypes where tabBool;
                                            tabColNestedTypes:tabColNestedTypes where tabBool;
                                            dictColTypes:value dict;
                                            nonStringToSymForNested:any
                                            (not dictColTypes=11h) & (tabColTypes=0h) &
                                            all each tabColNestedTypes~\\:\\:10h;
                                            nonNestedString: any (tabColTypes=0h) &
                                            not all each tabColNestedTypes~\\:\\:10h;
                                            nonNestedString or nonStringToSymForNested
                                            }''', self, dict_grab)
                if check_mixed_columns:
                    raise ValueError("This method can only handle casting string complex "
                                     "columns to symbols.  Other complex column data or "
                                     "casting to other data is not supported.")

                return_value = q('''{[tab;dict;typeValDict]
                                    dict:5h$dict;
                                    tabColsOrig:cols tab;
                                    tabColTypes:value type each flip tab;
                                    tabColNestedTypes:value first each type each\' flip tab;
                                    dictCols:key dict;
                                    tabBool:any tabColsOrig in/: dictCols;
                                    tabCols:tabColsOrig where tabBool;
                                    dict:tabCols!(dict tabCols);
                                    dictCols:tabCols;
                                    tabColTypes:tabColTypes where tabBool;
                                    tabColNestedTypes:tabColNestedTypes where tabBool;
                                    dictColTypes:value dict;
                                    // Check any char -> symbol casting
                                    b1:(tabColTypes=10h) & dictColTypes=11h;
                                    c1:()!();
                                    if[any b1;
                                      dCols1:dictCols where b1;
                                      f1:{(`$';x)}; c1:dCols1!(f1 each dCols1)];
                                    // Check casting to symbol, run `$string col
                                    // (also covers any symbol -> symbol cases)
                                    b2:(dictColTypes=11h) & not (b1 or tabColTypes=0h);
                                    c2:()!();
                                    if[any b2;
                                      dCols2:dictCols where b2;
                                      f2:{(`$string; x)}; c2:dCols2!(f2 each dCols2)];
                                    // Casting to string covering all cases except mixed lists
                                    b3: (dictColTypes=10h) & not tabColTypes=0h;
                                    c3:()!();
                                    if[any b3;
                                      dCols3:dictCols where b3;
                                      f3:{(string; x)}; c3:dCols3!(f3 each dCols3)];
                                    // Check mixed lists
                                    // if string column then allow cast to symbol
                                    // Check at beginning of method
                                    // should have returned error for other cases
                                    b4:(dictColTypes=11h) &
                                       (tabColTypes=0h) & tabColNestedTypes=10h;
                                    c4:()!();
                                    if[any b4;
                                      dCols4:dictCols where b4;
                                      f4:{(`$; x)}; c4:dCols4!(f4 each dCols4)];
                                    // Any matches that meet the vanilla case
                                    // and don't have additional needs --> not any (bools)
                                    b5:not any (b1;b2;b3;b4);
                                    .papi.errorList:();
                                    if[any b5;
                                    dCols5:dictCols where b5;
                                    dictColTypes5:dictColTypes where b5;
                                    f5:{[c;t;tvd]
                                        ({[cl;t;tvd]
                                            @[t$;cl;
                                            {[cl;ty;tvd;err].papi.errorList,:enlist
                                             "Not supported: Error casting ",
                                             string[tvd 7h$type cl],
                                             " to ", string[tvd 7h$ty],
                                             " with q error: ", err;}[cl;t;tvd;]]}[;t;tvd];
                                        c)};
                                    f5:f5[;;typeValDict];
                                    c5:dCols5!(f5\'[dCols5;dictColTypes5])];
                                    // Grab all cols
                                    c:c1,c2,c3,c4,c5;
                                    tableOutput:tabColsOrig xcols ![tab;();0b;c];
                                    $[count .papi.errorList;
                                    .papi.errorList;
                                    tableOutput]
                                    }''',
                                 self, dict_grab, type_number_to_pykx_k_type)
            else:
                try:
                    dtype_val = abs(kx_type_to_type_number[next(x for x
                                    in kx_type_to_type_number.keys() if x in str(dtype))])
                except StopIteration:
                    raise QError('Value passed does not match PyKX wrapper type')

                # Check needed to ensure no non string nested cols in table to convert
                check_mixed_columns = q('''{[tab;dtype]
                                            dtype:5h$dtype;
                                            tabColTypes:value tabColTypesDict:type each flip 0#tab;
                                            if[not any 0h=tabColTypes; :0b];
                                            tabColNestedTypes:value tabColTypesDict,distinct each
                                            type each\'flip #[;tab] where 0h=tabColTypesDict;
                                            nonNestedString:any (tabColTypes=0h) &
                                                        not all each tabColNestedTypes~\\:\\:10h;
                                            nonStringToSymForNested:any (not dtype=11h) &
                                                                    (tabColTypes=0h) &
                                                                    all each
                                                                    tabColNestedTypes~\\:\\:10h;
                                            nonNestedString or nonStringToSymForNested
                                            }''',
                                        self, dtype_val)
                if check_mixed_columns:
                    raise ValueError("This method can only handle casting string complex"
                                     " columns to symbols.  Other complex column data or"
                                     " casting to other data is not supported.")

                return_value = q('''{[tab;dtype;typeValDict]
                                    dtype:5h$dtype;
                                    tabCols:cols tab;
                                    tabColTypes:value type each flip 0#tab;
                                    tabColNestedTypes:value first each
                                                      type each\' flip 5#tab;
                                    // Support char->symbol conversion
                                    b1:(dtype=11h) & tabColTypes=10h;
                                    c1:()!();
                                    if[any b1;
                                    tCols1:tabCols where b1;
                                    f1:{(`$';x)}; c1:tCols1!(f1 each tCols1)
                                    ];
                                    // Support casting to symbol
                                    b2:(dtype=11h) & not (b1 or tabColTypes=0h);
                                    c2:()!();
                                    if[any b2;
                                    tCols2:tabCols where b2;
                                    f2:{(`$string@; x)}; c2:tCols2!(f2 each tCols2)
                                    ];
                                    // Support casting to string except for mixed lists
                                    b3:(dtype=10h) & not tabColTypes=0h;
                                    c3:()!();
                                    if[any b3;
                                    tCols3:tabCols where b3;
                                    f3:{(string; x)}; c3:tCols3!(f3 each tCols3)
                                    ];
                                    // For mixed lists support casting strings to symbols
                                    b4:(dtype=11h) & (tabColTypes=0h)
                                        & tabColNestedTypes=10h;
                                    c4:()!();
                                    if[any b4;
                                    tCols4:tabCols where b4;
                                    f4:{(`$; x)}; c4:tCols4!(f4 each tCols4)
                                    ];
                                    // Any other combination not matching b1-4
                                    b5:not any (b1;b2;b3;b4);
                                    .papi.errorList:();
                                    c5:()!();
                                    if[any b5;
                                    tCols5: tabCols where b5;
                                    f5:{[c;t;tvd]
                                        ({[cl;t;tvd]
                                        @[t$;cl;
                                            {[cl;ty;tvd;err].papi.errorList,:enlist
                                            "Not supported: Error casting ",
                                            string[tvd 7h$type cl], " to ",
                                            string[tvd 7h$ty], " with q error: ",
                                            err;}[cl;t;tvd;]]}[;t;tvd];
                                            c)};
                                    c5:tCols5!(f5[;dtype;typeValDict] each tCols5)
                                    ];
                                    c:c1,c2,c3,c4,c5;
                                    tableOutput:tabCols xcols ![tab;();0b;c];
                                    $[count .papi.errorList;
                                    .papi.errorList;
                                    tableOutput]
                                    }''',
                                 self, dtype_val, type_number_to_pykx_k_type)

            if return_value.t in [98, 99]:
                return return_value
            else:
                raise QError(return_value)

        except BaseException as e:
            if errors == 'raise':
                raise e
            else:
                return self

    @api_return
    def select_dtypes(self, include=None, exclude=None):
        # Check params included
        if include is None and exclude is None:
            raise ValueError('Expecting either include or exclude param to be passed')

        # Run for include
        if include is not None:
            if not isinstance(include, list):
                include = [include]

            # Convert all to str
            include_type_nums = [kx_type_to_type_number[x] for x
                                 in [x for x in kx_type_to_type_number.keys() for y
                                 in include if x in str(y)]]
            if 10 in include_type_nums:
                raise Exception("'CharVector' not supported."
                                " Use 'CharAtom' for columns of char atoms."
                                " 'kx.List' will include any columns containing"
                                " mixed list data.")
        # Run for exclude
        if exclude is not None:
            if not isinstance(exclude, list):
                exclude = [exclude]

            # Convert all to str
            exclude_type_nums = [kx_type_to_type_number[x] for x
                                 in [x for x in kx_type_to_type_number.keys() for y
                                 in exclude if x in str(y)]]
            if 10 in exclude_type_nums:
                raise Exception("'CharVector' not supported."
                                " Use 'CharAtom' for columns of char atoms."
                                " 'kx.List' will exclude any columns containing"
                                " mixed list data.")
        # Check no overlapping values
        if include is not None and exclude is not None:
            if any([x in exclude for x in include]):
                raise ValueError('Include and Exclude lists have overlapping elements')

        # Run if include is not None
        if include is not None:
            table_out = q('''{[qtab;inc]
                            tCols:cols $[99h~type qtab;value qtab;qtab];
                            inc:abs 5h$inc;
                            bList:value (type each flip 0#$[99h~type qtab;value qtab;qtab]) in inc;
                            if[not any bList;:(::)];
                            colList:tCols where bList;
                            res:?[qtab; (); 0b; colList!colList];
                            $[99h~type qtab;(key qtab)!res;res]
                          }''',
                        self, include_type_nums)  # noqa
        else:
            table_out = q('''{[qtab;exc]
                            tCols:cols $[99h~type qtab;value qtab;qtab];
                            exc:abs 5h$exc;
                            bList:value (type each flip 0#$[99h~type qtab;value qtab;qtab]) in exc;
                            if[all bList;:(::)];
                            colList:tCols where not bList;
                            res:?[qtab; (); 0b; colList!colList];
                            $[99h~type qtab;(key qtab)!res;res]
                          }''',
                        self, exclude_type_nums)  # noqa

        return table_out
