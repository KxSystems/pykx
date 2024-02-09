// dash.q - PyKX functionality for integration with KX Dashboards Integration

\d .pykx

// @private
// @desc
// Generate the Python function to retrieve the Python object
dash.util.lib:{x!{@[{.pykx.pyexec x;1b};"import ",string x;0b]} each x}`ast2json`ast

// @private
// @kind function
// @name .pykx.dash.available
// @category api
// @overview
// _Function to denote if all Python libraries required for dashboards are available_
dash.available:{all dash.util.lib}

// @private
// @desc
// Generate Python functions for parsing functions from strings
if[dash.available[];
    .pykx.pyexec"def _pykx_func_parse(py_code):\n",
                "    parse_info = ast2json.ast2json(ast.parse(py_code))\n",
                "    idx = next((i for i, item in enumerate(parse_info['body']) if item['_type'] == 'FunctionDef'), None)\n",
                "    if idx == None:\n",
                "        raise Exception('No function definition found')\n",
                "    return parse_info['body'][idx]['name']";
    ]

// @private
// @kind function
// @name .pykx.dash.util.getFunction
// @category api
// @overview
// _Functionality for the generation of a Python function to be called from code_
//
// ```q
// .pykx.dash.util.getFunction[pycode]
// ```
// **Parameters:**
//
// name           | type     | description                                                  |
// ---------------|----------|--------------------------------------------------------------|
// `pycode`       | `string` | The Python code this is to be executed for use as a function |
//
// **Returns:**
//
// type          | description |
// --------------|-------------|
// `composition` | A wrapped foreign Python object associated with the specified code
//
// **Example:**
//
// ```q
// q).pykx.dash.util.getFunction["def func(x):\n\treturn 1"]
// {[f;x].pykx.util.pykx[f;x]}[foreign]enlist
// ```
dash.util.getFunction:{[pyCode]
  if[not dash.available[];
    '"Required libraries for PyKX Dashboards integration not found"
    ];
  funcName:@[.pykx.get[`$"_pykx_func_parse";<];
    .pykx.topy pyCode;
    {[err]'err," raised when attempting to retrieve function definition"}
    ];
  @[.pykx.pyexec;
    pyCode;
    {[err]'err," raised when executing supplied Python code"}
    ];
  .pykx.get funcName
  }

// @private
// @kind function
// @name .pykx.dash.runFunction
// @category api
// @overview
// _Generate and execute a callable Python function using supplied arguments_
//
// ```q
// .pykx.dash.runFunction[pycode;args]
// ```
// **Parameters:**
//
// name     | type     | description                                                            |
// ---------|----------|------------------------------------------------------------------------|
// `pycode` | `string` | The Python code this is to be executed for use as a function           |
// `args`   | `list`   | A mixed/generic list of arguments to be used when calling the function |
//
// **Returns:**
//
// type   | description                                                            |
// -------|------------------------------------------------------------------------|
// `list` | The list of argument names associated with the user specified function |
//
// **Example:**
//
// Single argument function usage:
//
// ```q
// q).pykx.dash.runFunction["def func(x):\n\treturn x";enlist ([]5?1f;5?1f)]
// x         x1
// -------------------
// 0.9945242 0.6298664
// 0.7930745 0.5638081
// 0.2073435 0.3664924
// 0.4677034 0.9240405
// 0.4126605 0.5420167
// ```
//
// Multiple argument function usage:
//
// ```q
// q).pykx.dash.runFunction["def func(x, y):\n\treturn x*y";(2;5)]
// 10
// ```
//
// Function using Python dependencies:
//
// ```q
// q).pykx.dash.runFunction["import numpy as np\n\ndef func(x):\n\treturn np.linspace(0, x.py(), 5)";enlist 10]
// 0 2.5 5 7.5 10
// ```
dash.runFunction:{[pyCode;args]
  cache:util.defaultConv;
  .pykx.setdefault["k"];
  return:.[{.pykx.dash.util.getFunction[x][<] . (),y};(pyCode;args);{.pykx.util.defaultConv:x;'y}[cache]];
  util.defaultConv:cache;
  return}
