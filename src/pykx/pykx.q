// pykx.q - PyKX functionality for operation within a q process
//
// @namespace .pykx
// @category api
// @end

// @private
// @desc Process context prior to PyKX initialization
.pykx.util.prevCtx:system"d";

@[
  {if[not"{.pykx.pyexec x}"~string get x;
     -1"Warning: Detected invalid '.p.e' function definition expected for PyKX.\n",
     "Have you loaded another Python integration first?\n\n",
     "Please consider full installation of PyKX under q following instructions at:\n",
     "https://code.kx.com/pykx/pykx-under-q/intro.html#installation.\n"
     ]
    };
  `.p.e;
  {::}]

\d .pykx

// @private
// @overview
// For a given function retrieve the location from which the file was loaded
// 
// @return {string} the location from which this file is being loaded
util.getLoadDir:{@[{"/"sv -1_"/"vs ssr[;"\\";"/"](-3#get .z.s)0};`;""]}

// @private
// @desc Operating system within which PyKX under q is loaded
//
// @type {char}
util.os:first string .z.o;

// @private
// @desc Provided command line arguments to q process
//
// @type {dict}
util.startup:.Q.opt .z.x


// @private
// @overview
// @desc Load a file at an associated folder location, this is used
//     to allow loading of files at folder locations containing spaces
util.loadfile:{[folder;file]
  cache:system"cd";
  res:.[{system"cd ",x;res:system"l ",y;(0b;res)};(folder;file);{(1b;x)}];
  if[folder~system"cd";system"cd ",cache];
  $[res[0];'res[1];res[1]]
  }

// @private
// @desc Retrieval of PyKX initialization directory on first initialization
if[not "true"~lower getenv`PYKX_LOADED_UNDER_Q;
  util.whichPython:$[count pykxExecutable:getenv`PYKX_EXECUTABLE;pykxExecutable;()];
  util.dirCommand:"-c \"import pykx; print('PYKX_DIR: ' + str(pykx.config.pykx_dir))\"";
  if[not count pykxDir:getenv`PYKX_DIR;
    util.dirSysCall:{ret:system x," ",util.dirCommand;util.whichPython:x;ret};
    pykxDir:$[count util.whichPython;util.dirSysCall[util.whichPython];
      @[util.dirSysCall;"python";{util.dirSysCall["python3"]}]
      ];
    pykxDir:ssr[;"\\";"/"]last vs["PYKX_DIR: "]last pykxDir
    ];
  ];

// @private
// @desc
// Environment variable denoting that pythonic invocations of PyKX should the
// PyKX act as of in a q process, as q symbols are already defined in the process.
setenv[`PYKX_UNDER_Q;"true"];

// @private
// @desc Compose a list of functions
k)c:{'[y;x]}/|:

// @private
// @desc Compose using enlist for generation of variadic functions
k)ce:{'[y;x]}/enlist,|:

// @desc Print a message warning that "UNDER_PYTHON" is deprecated
if[not ""~getenv`UNDER_PYTHON;
   -1"WARN: Environment variable 'UNDER_PYTHON' is deprecated, if set locally update to use 'PYKX_UNDER_PYTHON'";
  ]

// @desc Make use of `pykx.so` logic when running under Python
if["true"~getenv`PYKX_UNDER_PYTHON;
  util.load:2:[hsym`$pykxDir,"/pykx";]
  ];

// @private
// @desc Load PyKX initialization script if not previously initialised
if[not "true"~lower getenv`PYKX_LOADED_UNDER_Q;
  util.pyEnvInfo:("None"; "None"; "");
  if[0=count getenv`PYKX_Q_LOADED_MARKER;
    @[system"l ",;
      "pykx_init.q_";
      {[x;y] util.loadfile[x;"pykx_init.q_"]}[pykxDir]
      ]
    ];
  ];

// @private
// @desc
// Dictionary containing information useful for debugging, wrapping
// within if statement is to ensure these values are not overwritten
// on multiple loads
if[not `debug in key `.pykx;
    debug: (!) . flip (
        (`pykxLoadDir    ; util.getLoadDir[]);
        (`startupOptions ; .Q.s1 util.startup);
        (`os             ; util.os);
        (`whichPython    ; util.whichPython)
        )
  ];

// @private
// @desc Initialise Python integration within q process
preinit[];

// @private
// @desc Validate that PyKX was installed, failover will initialise pykxDir to "/pykx"
if[pykxDir~"/pykx";
  '"Failed to find pykx - ensure your Python environment is properly configured and activated"
  ];

// @private
// @desc
// List mapping functions to be loaded from the C API to their names within the q API
util.CFunctions:flip `qname`cname`args!flip (
    (`util.repr      ;`repr           ;2);
    (`util.pyForeign ;`k_to_py_foreign;3);
    (`util.isf       ;`k_check_python ;1);
    (`util.pyrun     ;`k_pyrun        ;4);
    (`util.foreignToq;`foreign_to_q   ;1);
    (`util.callFunc  ;`call_func      ;4);
    (`pyimport       ;`import         ;1);
    (`util.setGlobal ;`set_global     ;2);
    (`util.getGlobal ;`get_global     ;1);
    (`util.setAttr   ;`set_attr       ;3);
    (`getattr        ;`get_attr       ;2)
    )

// @private
// @desc
// Load defined C functions to PyKX for later use
{.Q.dd[`.pykx;x`qname]set util.load x`cname`args}each util.CFunctions;

// @private
// @desc Convert q/Python objects to Pythonic foreigns
util.toPython:{wrap util.pyForeign[$[type[x]in 104 105 112h;wrap[unwrap x]`;x];y;z]}

// @private
// @desc Check if argument is of specified type
util.isch  :{$[(104= type y);$[x~y[::]0;1b;0b];0b]}

// @private
// @desc
// Validation functions to allow checking of type
// intended to be used as function arguments
util.ispy  :util.isch[`..python]

// @private
util.isnp  :util.isch[`..numpy]
util.ispd  :util.isch[`..pandas]
util.ispa  :util.isch[`..pyarrow]
util.isk   :util.isch[`..k]
util.israw :util.isch[`..raw]

// @private
// @desc
// Determine if a supplied object requires conversion
util.isconv:{any(util.ispy;util.isnp;util.ispd;util.ispa;util.isk;util.israw)@\:x}

// @private
// @desc Convert a supplied argument to the specified python object type
util.convertArg:{
  $[util.isw x;x;
    util.isconv x;
    .z.s[(x; 1; 0b)];
    $[not util.isconv x 0;util.toPython . x;
      util.ispy  x 0; [.z.s[(x[0][::][1]; 1; x[2])]];
      util.isnp  x 0; [.z.s[(x[0][::][1]; 2; x[2])]];
      util.ispd  x 0; [.z.s[(x[0][::][1]; 3; x[2])]];
      util.ispa  x 0; [.z.s[(x[0][::][1]; 4; x[2])]];
      util.isk   x 0; [.z.s[(x[0][::][1]; 5; x[2])]];
      util.israw x 0; [.z.s[(x[0][::][1]; x[1]; 1b)]];
      '"Unsupported conversion attempted"
      ]
    ]
  }

// @private
// @desc Convert a supplied argument to the default q -> Python type
util.toDefault:{
  $[util.isw x;(::);
    util.isconv x;(::);
    "py"      ~ util.defaultConv;topy;
    "np"      ~ util.defaultConv;tonp;
    "pd"      ~ util.defaultConv;topd;
    "pa"      ~ util.defaultConv;topa;
    "k"       ~ util.defaultConv;tok;
    "raw"     ~ util.defaultConv;toraw;
    "default" ~ util.defaultConv;todefault;
    (::)
    ]x
  };

// @private
// @desc
// Foreign object manipulation used for data conversions to q/foreign
// attribute manipulation and data setting
util.pykx:{[f; x]
  f:unwrap f;
  $[-11h<>t:type x0:x 0;
    $[t=102h;
      $[any u:x0~/:(*;<;>);
        [c:(wrap;toq;::)where[u]0;$[1=count x;.pykx.c c,;c .[;1_x]@]pyfunc f];
        (:)~x0;[setattr . f,@[;0;{`$_[":"=s 0]s:string x}]1_x;];
        (@)~x0;[
          fn:@[wrap[f];`:__getitem__;{'"Python object has no attribute __getitem__."}];
          $[count 2_x;.[;2_x];]fn x 1
          ];
        (=)~x0;[
          fn:@[wrap[f];`:__setitem__;{'"Python object has no attribute __setitem__."}];
          fn . (x 1;x 2);
          ];
        '`NYI
        ];
      wrap pyfunc[f] . x];
      ":"~first a0:string x0;
      .[{$[1=count x;;.[;1_x]]wrap y getattr/` vs`$1_z};
        (x;f;a0);
        {[f;x;str;err]
          .pykx.pyexec"from pathlib import Path";
          .[wrap[f];
            (.pykx.eval["lambda x:Path(str(x))"]`$1_string x 0),1 _ x;
            {[x;y]'"Python object has no attribute '",
                   x,"' or could not apply Path '",
                   x,"' to Python object raising error: ",y}[str]
            ]
          }[f;x;a0]];
      x0~`.;f;x0~`;toq f;
      wrap pyfunc[f] . x]
  }

// @private
// @desc Functionality for the wrapping of functions and foreign objects
util.wfunc:{[f;x]r:wrap f x 0;$[count x:1_x;.[;x];]r}
util.wf:{[f;x].pykx.util.pykx[f;x]}

// @private
// @desc
// Functionality used for checking if an supplied
// argument is a Python foreign or wrapped object 
util.isw:{
  if[not 105h~type x;:0b];
  $[.pykx.util.wf~$[104 105h~t:type each u:get x;
      :.z.s last u;
      104h~first t;
      first value first u;
      0b];:1b;
    101 105h~t;:(::)~first u;
    100 105h~t;:.pykx.toq~first u;
    0b]}

// @private
// @desc Functionality for management of keywords/keyword dictionaries etc.
util.iskw  :util.isch[`..pykw]
util.isargl:util.isch[`..pyas]
util.iskwd :util.isch[`..pyks]
util.isarg :{any(util.iskw;util.isargl;util.iskwd)@\:x}

// @private
// @desc
// Parse supplied positional, named and keyword being supplied to
// a Python function such that their application to the function is
// correct
util.parseArgs:{
  hasargs:$[(x~enlist[::])&1=count x;0;1];
  kwlist:x where util.iskw each x;
  kwdict:$[1<count[x where util.iskwd each x];
    '"Expected only one key word dictionary to be used in function call";
    $[0=count[x where util.iskwd each x]; ()!(); (x where util.iskwd each x)[0][::]1]
    ];
  kwargs:$[0<count kwlist;
    ({x[::][1]} each kwlist)!({x[::]2} each kwlist);
    ()!()
    ];
  parse_keys: key[kwargs],key kwdict;
  if[not count[parse_keys]=count distinct parse_keys;
    '"Expected only unique key names for keyword arguments in function call"
    ];
  if[any{not -11h=type x}each parse_keys;
    '"Expected Symbol Atom for keyword argument name"
    ];
  // Join will overwrite duplicated keys so we must do the check first
  kwargs: kwargs,kwdict;
  if[not count kwargs;kwargs:()!()];
  kwargs:(key kwargs)!{[kwargs; x] unwrap util.convertArg util.toDefault kwargs[x]}[kwargs;] each key kwargs;
  arglist:{unwrap util.convertArg util.toDefault x} each (x where not util.isarg each x),
    $[0<count x where util.isargl each x;
      $[not 1<count x where util.isargl each x;
        (x where util.isargl each x)[0][::]1;
        '"Expected only one arg list to be using in function call"
        ];
      x where util.isargl each x
      ];
  (hasargs; arglist; kwargs)
  };

// -----------------------
// User Callable Functions
// -----------------------

// @name pykw
// @category api
// @overview
// _Allow users to apply individual keywords to a Python function_
//
// ```q
// `argName pykw argValue
// ```
//
// !!! Warning
// 
//      This function will be set in the root `.q` namespace
//
// **Parameters:**
//
// name       | type     | description
// -----------|----------|------------
// `argName`  | `symbol` | Name of the keyword argument to be applied
// `argValue` | `any`    | Value to be applied as a keyword
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which when used with a wrapped callable Python 
//
// **Example:**
//
// The following example shows the usage of `pykw` with a Python function
//
// ```q
// q)p)import numpy as np
// q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
// q)qfunc:.pykx.get[`func;<] / callable, returning q
// q)qfunc[`d pykw 1;`c pykw 2;`b pykw 3;`a pykw 4] / all keyword args specified
// 4 3 2 1 24
// q)qfunc[1;2;`d pykw 3;`c pykw 4]   / mix of positional and keyword args
// ```
.q.pykw     :{x[y;z]}(`..pykw;;;)  / identify keyword args with `name pykw value

// @name pyarglist
// @category api
// @overview
// _Allow users to apply a list of arguments to a Python function, equivalent to `*args`_
//
// ```q
// pyarglist argList
// ```
//
// !!! Warning
//
// 	This function will be set in the root `.q` namespace
//
// **Parameters:**
//
// name       | type   | description
// -----------|--------|------------
// `argList`  | `list` | List of optional arguments
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which when used with a wrapped callable Python
//
// **Example:**
//
// The following example shows the usage of `pyarglist` with a Python function and
// various configurations of it's use
//
// ```q
// q)p)import numpy as np
// q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
// q)qfunc:.pykx.get[`func;<] / callable, returning q
// q)qfunc[pyarglist 1 1 1 1]          / full positional list specified
// 1 1 1 1 1
// q)qfunc[pyarglist 1 1]              / partial positional list specified
// 1 1 3 4 12
// q)qfunc[1;1;pyarglist 2 2]          / mix of positional args and positional list
// 1 1 2 2 4
// q)qfunc[pyarglist 1 1;`d pykw 5]    / mix of positional list and keyword args
// 1 1 3 5 15
// ```
.q.pyarglist:{x y}(`..pyas;;)      / identify pos arg list (*args in python)

// @name pykwargs
// @category api
// @overview
// _Allow users to apply a dictionary of named arguments to a Python function, equivalent to `*kwargs`_
//
// ```q
// pykwargs argDict
// ```
//
// !!! Warning
// 
//      This function will be set in the root `.q` namespace 
//
// **Parameters:**
//
// name       | type   | description
// -----------|--------|------------
// `argDict`  | `dict` | A dictionary of named keyword arguments mapped to their value
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which when used with a wrapped callable Python
//
// **Example:**
//
// The following example shows the usage of `pykwargs` with a Python function and
// various configurations of it's use
//
// ```q
// q)p)import numpy as np
// q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
// q)qfunc:.pykx.get[`func;<] / callable, returning q
// q)qfunc[pykwargs`d`c`b`a!1 2 3 4]             / full keyword dict specified
// 4 3 2 1 24
// q)qfunc[2;2;pykwargs`d`c!3 3]                 / mix of positional args and keyword dict
// 2 2 3 3 36
// q)qfunc[`d pykw 1;`c pykw 2;pykwargs`a`b!3 4] / mix of keyword args and keyword dict
// 3 4 2 1 24
// ```
.q.pykwargs :{x y}(`..pyks;;)      / identify keyword dict (**kwargs in python)

// @name .pykx.topy
// @category api
// @overview
// _Tag a q object to be indicate conversion to a Python object when called in Python_
//
// ```q
// .pykx.topy[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be defined as a Python object in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a Python type object. |
//
// ```q
// // Denote that a q object once passed to Python should be managed as a Python object
// q).pykx.topy til 10
// enlist[`..python;;][0 1 2 3 4 5 6 7 8 9]
//
// // Pass a q object to Python with default conversions and return type
// q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
// <class 'numpy.ndarray'>
//
// // Pass a q object to Python treating the Python object as a Python Object
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.topy til 10
// <class 'list'>
// ```
topy:{x y}(`..python;;)

// @name .pykx.tonp
// @category api
// @overview
// _Tag a q object to be indicate conversion to a Numpy object when called in Python_
//
// ```q
// .pykx.tonp[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be defined as a Numpy object in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a Numpy type object. |
//
// ```q
// // Denote that a q object once passed to Python should be managed as a Numpy object
// q).pykx.tonp til 10
// enlist[`..numpy;;][0 1 2 3 4 5 6 7 8 9]
//
// // Update the default conversion type to be non numpy
// q).pykx.util.defaultConv:"py"
//
// // Pass a q object to Python with default conversions and return type
// q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
// <class 'list'>
// 
// // Pass a q object to Python treating the Python object as a Numpy Object
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.tonp til 10
// <class 'numpy.ndarray'>
// ```
tonp:{x y}(`..numpy;;)

// @name .pykx.topd
// @category api
// @overview
// _Tag a q object to be indicate conversion to a Pandas object when called in Python_
//
// ```q
// .pykx.topd[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be defined as a Pandas object in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a Pandas type object. |
//
// ```q
// // Denote that a q object once passed to Python should be managed as a Pandas object
// q).pykx.topd til 10
// enlist[`..pandas;;][0 1 2 3 4 5 6 7 8 9]
//
//
// // Pass a q object to Python with default conversions and return type
// q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
// <class 'numpy.ndarray'>
//
// // Pass a q object to Python treating the Python object as a Pandas Object
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.topd til 10
// <class 'pandas.core.series.Series'>
// ```
topd:{x y}(`..pandas;;)

// @name .pykx.topa
// @category api
// @overview
// _Tag a q object to be indicate conversion to a PyArrow object when called in Python_
//
// ```q
// .pykx.topa[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be defined as a PyArrrow object in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a PyArrow type object. |
//
// ```q
// // Denote that a q object once passed to Python should be managed as a PyArrow object
// q).pykx.topa til 10
// enlist[`..pyarrow;;][0 1 2 3 4 5 6 7 8 9]
//
// // Pass a q object to Python with default conversions and return type
// q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
// <class 'numpy.ndarray'>
//
// // Pass a q object to Python treating the Python object as a PyArrow Object
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.topa til 10
// <class 'pyarrow.lib.Int64Array'>
// ```
topa:{x y}(`..pyarrow;;)

// @name .pykx.tok
// @category api
// @overview
// _Tag a q object to be indicate conversion to a Pythonic PyKX object when called in Python_
//
// ```q
// .pykx.tok[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be defined as a PyKX object in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a PyKX type object. |
//
// ```q
// // Denote that a q object once passed to Python should be managed as a PyKX object
// q).pykx.tok til 10
// enlist[`..k;;][0 1 2 3 4 5 6 7 8 9]
//
// // Pass a q object to Python with default conversions and return type
// q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
// <class 'numpy.ndarray'>
//
// // Pass a q object to Python treating the Python object as a PyKX object
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.tok til 10
// <class 'pykx.wrappers.LongVector'>
// ```
tok: {x y}(`..k;;)

// @name .pykx.toraw
// @category api
// @overview
// _Tag a q object to be indicate a raw conversion when called in Python_
//
// ```q
// .pykx.toraw[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be converted in its raw form in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a raw object. |
//
// ```q
// // Denote that a q object once passed to Python should be managed as a Numpy object
// q).pykx.toraw til 10
// enlist[`..raw;;][0 1 2 3 4 5 6 7 8 9]
//
// // Pass a q object to Python with default conversions and return type
// q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
// <class 'numpy.ndarray'>
//
// // Pass a q object to Python treating the Python object as a raw Object
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.toraw til 10
// <class 'list'>
// ```
toraw: {x y}(`..raw;;)

// @name .pykx.todefault
// @category api
// @overview
// _Tag a q object to indicate it should use the PyKX default conversion when called in Python_
//
// ```q
// .pykx.todefault[qObject]
// ```
//
// **Parameters:**
//
// name      | type    | description |
// ----------|---------|-------------|
// `qObject` | `any`   | A q object which is to be converted to a default form in Python. |
//
// **Return:**
//
// type         | description
// -------------|------------
// `projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a default object. |
//
// !!! Note
//     The `todefault` conversion is used to match embedPy conversion logic, in particular it converts q lists to Python lists when dealing with contiguous datatypes rather than to nested single value array types. Additionally it converts q tables to Pandas DataFrames
//
// ```q
// // Denote that a q object once passed to Python should be managed as a default object
// // in this case a q list is converted to numpy 
// q).pykx.todefault til 10
// enlist[`..numpy;;][0 1 2 3 4 5 6 7 8 9]
//
// // Pass a q list to Python treating the Python object as PyKX default
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.todefault (til 10;til 10)
// <class 'list'>
//
// // Pass a q Table to Python by default treating the Python table as a Pandas DataFrame
// q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.todefault ([]til 10;til 10)
// <class 'pandas.core.frame.DataFrame'>
// ```
todefault:{$[0h=type x;toraw x;$[99h~type x;all 98h=type each(key x;value x);0b]|98h=type x;topd x;tonp x]}

// @kind function
// @name .pykx.wrap
// @category api
// @overview
// _Convert a foreign object generated from Python execution to a callable `q` object._
//
// ```q
// .pykx.wrap[pyObject]
// ```
//
// **Parameters:**
//
// name       | type      | description |
// -----------|-----------|-------------|
// `pyObject` | `foreign` | A Python object which is to be converted to a callable q object. |
//
// **Returns:**
//
// type          | description |
// --------------|-------------|
// `composition` | The Python object wrapped such that it can be called using q |
//
// ```q
// // Create a q foreign object in Python
// q)a:.pykx.pyeval"pykx.Foreign([1, 2, 3])"
// q)a
// foreign
// q).pykx.print a
// [1, 2, 3]
//
// // Wrap the foreign object and convert to q
// q)b:.pykx.wrap a
// q)b
// {[f;x].pykx.util.pykx[f;x]}[foreign]enlist
// q)b`
// 1 2 3
// ```
wrap:ce util.wf@

// @kind function
// @name .pykx.unwrap
// @category api
// @overview
// _Convert a wrapped foreign object generated from this interface into a python foreign._
//
// ```q
// .pykx.unwrap[wrapObj]
// ```
//
// **Parameters:**
//
//  name      | type                | description |
// -----------|---------------------|-------------|
//  `wrapObj` | composition/foreign | A (un)wrapped Python foreign object. |
//
// **Returns:**
//
//  type      | description |
// -----------|-------------|
//  `foreign` | The unwrapped representation of the Python foreign object. |
//
// ```q
// // Generate an object which returns a wrapped Python foreign
// q).pykx.set[`test;.pykx.topd ([]2?0p;2?`a`b`c;2?1f;2?0t)]
// q)a:.pykx.get`test
// q)show a
// {[f;x].pykx.util.pykx[f;x]}[foreign]enlist
//
// // Unwrap the wrapped object
// q).pykx.unwrap a
// foreign
// ```
unwrap:{
  c:last get last get first get last@;
  $[util.isw x;t:type each u:get x;:x];
  if[(101 105h~t) and (::)~first u;:c u]; 
  if[(100 105h~t) and .pykx.toq~first u;:c u];
  if[104 105h~t;:(last u)`.];
  x`.}


// @kind function
// @name .pykx.setdefault
// @category api
// @overview
// _Define the default conversion type for KX objects when converting from q to Python_
//
// ```q
// .pykx.setdefault[conversionFormat]
// ```
//
// **Parameters:**
//
// name               | type   | description |
// -------------------|--------|-------------|
// `conversionFormat` | string | The Python data format to which all q objects when passed to Python will be converted. |
//
// **Returns:**
//
// type | description |
// -----|-------------|
// `::` | Returns generic null on successful execution and updates variable `.pykx.util.defaultConv`
//
// ??? "Supported Options"
//
//     The following outline the supported conversion types and the associated values which can be passed to set these values
//
//     Conversion Format                                              | Accepted inputs              |
//     ---------------------------------------------------------------|------------------------------|
//     [Numpy](https://numpy.org/)                                    | `"np", "numpy", "Numpy"`     |
//     [Pandas](https://pandas.pydata.org/docs/user_guide/index.html) | `"pd", "pandas", "Pandas"`   |
//     [Python](https://docs.python.org/3/library/datatypes.html)     | `"py", "python", "Python"`   |
//     [PyArrow](https://arrow.apache.org/docs/python/index.html)     | `"pa", "pyarrow", "PyArrow"` |
//     [K](type_conversions.md)                                       | `"k", "q"`                   |
//     raw                                                            | `"raw"`                      |
//     default                                                        | `"default"`                  |
//
//
// ```q
// // Default value on startup is "default"
// q).pykx.util.defaultConv
// "default"
//
// // Set default value to Pandas
// q).pykx.setdefault["Pandas"]
// q).pykx.util.defaultConv
// "pd"
// ```
setdefault:{
  x:lower x;
  util.defaultConv:$[
    x in ("np";"numpy")        ;"np";
    x in ("py";"python")       ;"py";
    x in (enlist"k" ;enlist"q");"k";
    x in ("pd";"pandas")       ;"pd";
    x in ("pa";"pyarrow")      ;"pa";
    x in enlist["raw"]         ;"raw";
    x in enlist "default"      ;"default";
    '"unknown conversion type: ",x
    ];
  setenv[`PYKX_DEFAULT_CONVERSION;$[-10h=type util.defaultConv;enlist;]util.defaultConv];
  }

// @kind function
// @name .pykx.toq
// @category api
// @overview
// _Convert an (un)wrapped `PyKX` foreign object into an analogous q type._
//
// ```q
// .pykx.toq[pythonObject]
// ```
//
// **Parameters:**
//
// name           | type                   | description |
// ---------------|------------------------|-------------|
// `pythonObject` | foreign/composition    | A foreign Python object or composition containing a Python foreign to be converted to q
//
// **Return:**
//
// type  | description
// ------|------------
// `any` | A q object converted from Python
//
// ```q
// // Convert a wrapped PyKX foreign object to q
// q)show a:.pykx.eval["1+1"]
// {[f;x].pykx.util.pykx[f;x]}[foreign]enlist
// q).pykx.toq a
// 2
//
// // Convert an unwrapped PyKX foreign object to q
// q)show b:a`.
// foreign
// q).pykx.toq b
// 2
// ```
py2q:toq:{$[type[x]in 104 105 112h;util.foreignToq unwrap x;x]}

// @private
// @name .pykx.pyfunc
// @category api
// @overview
// _Convert a provided foreign object to a callable function returning a foreign object result_
pyfunc:{ce .[util.callFunc x],`.pykx.util.parseArgs}

// @kind function
// @name .pykx.pyeval
// @category api
// @overview
// _[Evaluates](https://docs.python.org/3/library/functions.html#eval) a `CharVector` as python code and return the result as a `q` foreign._
//
// ```q
// .pykx.pyeval[pythonCode]
// ```
//
// **Parameters:**
//
// name         | type     | description |
// -------------|----------|-------------|
// `pythonCode` | `string` | A string of Python code to be evaluated returning the result as a q foreign object. |
//
// **Return:**
//
//  type      | description |
// -----------|-------------|
//  `foreign` | The return of the Python string evaluation returned as a q foreign. |
//
// ```q
// // evaluate a Python string
// q).pykx.pyeval"1+1"
// foreign
//
// // Use a function defined in Python taking a single argument
// q).pykx.pyeval["lambda x: x + 1"][5]
// foreign
//
// // Use a function defined in Python taking multiple arguments
// q).pykx.pyeval["lambda x, y: x + y"][4;5]
// foreign
// ```
pyeval:util.pyrun[1b; 0b; 1b]

// @kind function
// @name .pykx.pyexec
// @category api
// @overview
// _[Executes](https://docs.python.org/3/library/functions.html#exec) a `string` as python code in Python memory._
//
// ```q
// .pykx.pyexec[pythonCode]
// ```
//
// **Parameters:**
//
// name         | type      | description |
// -------------|-----------|-------------|
// `pythonCode` | string    | A string of Python code to be executed. |
//
// **Return:**
//
//  type | description |
// ------|-------------|
//  `::` | Returns generic null on successful execution, will return an error if execution of Python code is unsuccessful. |
//
// ```q
// // Execute valid Python code
// q).pykx.pyexec"1+1"
// q).pykx.pyexec"a = 1+1"
//
// // Evaluate the Python code returning the result to q
// q).pykx.qeval"a"
// 2
//
// // Attempt to execute invalid Python code
// q).pykx.pyexec"1+'test'"
// 'TypeError("unsupported operand type(s) for +: 'int' and 'str'")
//   [0]  .pykx.pyexec["1+'test'"]
//        ^
// ```
pyexec:util.pyrun[0b; 1b; 0b]

// @kind function
// @name .pykx.eval
// @category api
// @overview
// _[Evaluates](https://docs.python.org/3/library/functions.html#eval) a `string` as python code and return the result as a wrapped `foreign` type._
//
// ```q
// .pykx.eval[pythonCode]
// ```
//
// **Parameters:**
//
// name         | type      | description |
// -------------|-----------|-------------|
// `pythonCode` | string    | A string of Python code to be executed returning the result as a wrapped foreign object. |
//
// **Return:**
//
// type | description
// -----|------------
// `composition` | A wrapped foreign object which can be converted to q or Python objects
//
// ```q
// // Evaluate the code and return as a wrapped foreign object
// q).pykx.eval"1+1"
// {[f;x].pykx.util.pykx[f;x]}[foreign]enlist
//
// // Evaluate the code and convert to Python foreign
// q).pykx.eval["1+1"]`.
// foreign
//
// // Evaluate the code and convert to a q object
// q).pykx.eval["lambda x: x + 1"][5]`
// 6
// ```
.pykx.eval:ce util.wfunc pyeval

// @kind function
// @name .pykx.qeval
// @category api
// @overview
// _[Evaluates](https://docs.python.org/3/library/functions.html#eval) a `CharVector` in Python returning the result as a q object._
//
// ```q
// .pykx.qeval[pythonCode]
// ```
//
// **Parameters:**
//
// name         | type      | description |
// -------------|-----------|-------------|
// `pythonCode` | string    | A string of Python code to be evaluated returning the result as a q object. |
//
// **Return:**
//
// type  | description |
// ------|-------------|
// `any` | The return of the Python string evaluation returned as a q object. |
//
// ```q
// // evaluate a Python string
// q).pykx.qeval"1+1"
// 2
//
// // Use a function defined in Python taking a single argument
// q).pykx.qeval["lambda x: x + 1"][5]
// 6
//
// // Use a function defined in Python taking multiple arguments
// q).pykx.qeval["lambda x, y: x + y"][4;5]
// 9
// ```
qeval:{toq .pykx.eval x}

// @kind function
// @name .pykx.pyimport
// @category api
// @overview
// _Import a Python library and store as a foreign object._
// 
// ```q
// .pykx.pyimport[libName]
// ```
//
// **Parameters:**
//
// name      | type   | description |
// ----------|--------|-------------|
// `libName` | symbol | The name of the Python library/module to imported for use |
//
// **Return:**
//
// type      | description
// ----------|------------
// `foreign` | Returns a foreign object associated with an imported library on success, otherwise will error if library/module cannot be imported.
//
// ```q
// // Import numpy for use as a q object named numpy
// q)np:.pykx.pyimport`numpy
// q).pykx.print np
// <module 'numpy' from '/usr/local/lib64/python3.9/site-packages/numpy/__init__.py'>
// ```
pyimport;    // Note this function is dynamically loaded from C

// @kind function
// @name .pykx.import
// @category api
// @overview
// _Import a Python library and store as a wrapped foreign object to allow use in q projections/evaluation._
//
// ```q
// .pykx.import[libName]
// ```
//
// **Parameters:**
//
// name      | type   | description |
// ----------|--------|-------------|
// `libName` | symbol | The name of the Python library/module to imported for use |
//
// **Return:**
//
// type          | description
// --------------|------------
// `composition` | Returns a wrapped foreign object associated with an imported library on success, otherwise will error if library/module cannot be imported.
//
// ```q
// // Import numpy for use as a q object named numpy
// q)np:.pykx.import`numpy
// q).pykx.print np
// <module 'numpy' from '/usr/local/lib64/python3.9/site-packages/numpy/__init__.py'>
//
// // Use a function from within the numpy library using attribute retrieval
// q).pykx.print np[`:arange]
// <built-in function arange>
// q)np[`:arange][10]`
// 0 1 2 3 4 5 6 7 8 9
// ```
import:ce util.wfunc pyimport

// @kind function
// @name .pykx.repr
// @category api
// @overview
// _Evaluate the python function `repr()` on an object retrieved from Python memory_
//
// ```q
// .pykx.repr[pythonObject]
// ```
//
// **Parameters:**
//
// name           | type  | description |
// ---------------|-------|-------------|
// `pythonObject` | `any` | A Python object retrieved from the Python memory space, if passed a q object this will retrieved using [`.Q.s1`](https://code.kx.com/q/ref/dotq/#qs1-string-representation). |
//
// **Return:**
//
// type     | description
// ---------|------------
// `string` | The string representation of the Python/q object
//
// ```q
// // Use a wrapped foreign object
// q)a: .pykx.eval"1+1"
// q).pykx.repr a
// ,"2"
//
// // Use a foreign object
// q)a: .pykx.eval"'hello world'"
// q).pykx.repr a`.
// "hello world"
//
// // Use a q object
// q).pykx.repr til 5
// "0 1 2 3 4"
// ```
repr :{$[type[x]in 104 105 112h;util.repr[1b] unwrap x;.Q.s x]}

// @kind function
// @name .pykx.print
// @category api
// @overview
// _Print a python object directly to stdout. This is equivalent to calling `print()` on the object in Python._
//
// ```q
// .pykx.print[pythonObject]
// print[pythonObject]
// ```
//
// **Parameters:**
//
// name           | type              | description |
// ---------------|-------------------|-------------|
// `pythonObject` | (wrapped) foreign | A Python object retrieved from the Python memory space, if passed a q object this will be 'shown' |
//
// **Return:**
//
// type | description
// -----|------------
// `::` | Will print the output to stdout but return null
//
// !!! Note
//
//         For back compatibility with embedPy this function is also supported in the shorthand form `print` which uses the `.q` namespace. To not overwrite `print` in your q session and allow use only of the longhand form `.pykx.print` set the environment variable `UNSET_PYKX_GLOBALS` to any value.
//
// ```q
// // Use a wrapped foreign object
// q)a: .pykx.eval"1+1"
// q).pykx.print a
// 2
//
// // Use a foreign object
// q)a: .pykx.eval"'hello world'"
// q).pykx.print a`.
// hello world
// 
// // Use a q object
// q).pykx.print til 5
// 0 1 2 3 4
// 
// // Print the return of a conversion object
// q).pykx.print .pykx.topd ([]5?1f;5?0b)
//           x     x1
// 0  0.178084  False
// 1  0.301772   True
// 2  0.785033   True
// 3  0.534710  False
// 4  0.711172  False
// ```
print:{
  $[type[x]in 104 105 112h ;
    $[any(util.isw;util.isconv)@\:x;
     .pykx.eval["lambda x:print(x)"]x;
     show x];
    show x
    ];
  }

// @kind function
// @name .pykx.version
// @category api
// @overview
// _Retrieve the version of PyKX presently being used by a q process_
//
// ```q
// .pykx.version[]
// ```
//
// **Return:**
//
// type     | description
// ---------|------------
// `string` | The version number of PyKX installed within the users q session
//
// ```q
// q).pykx.version[]
// "2.0.0"
// ```
version:{pyexec"import pykx as kx";string qeval"kx.__version__"}

// @kind function
// @name .pykx.set
// @category api
// @overview
// _Set a q object to a named and type specified object in Python memory_
//
// ```q
// .pykx.set[objectName;qObject]
// ```
//
// **Parameters:**
//
// name         | type     | description |
// -------------|----------|-------------|
// `objectName` | `symbol` | The name to be associated with the q object being persisted to Python memory |
// `qObject`    | `any`    | The q/Python entity that is to be stored to Python memory
//
// **Return:**
//
// type | description
// -----|------------
// `::` | Returns null on successful execution
//
// ```q
// // Set a q array of guids using default behavior
// q).pykx.set[`test;3?0Ng]
// q)print .pykx.get`test
// [UUID('3d13cc9e-f7f1-c0ee-782c-5346f5f7b90e')
//  UUID('c6868d41-fa85-233b-245f-55160cb8391a')
//  UUID('e1e5fadd-dc8e-54ba-e30b-ab292df03fb0')]
//
// // Set a q table as pandas dataframe
// q).pykx.set[`test;.pykx.topd ([]5?1f;5?1f)]
// q)print .pykx.get`test
//           x        x1
// 0  0.301772  0.392752
// 1  0.785033  0.517091
// 2  0.534710  0.515980
// 3  0.711172  0.406664
// 4  0.411597  0.178084
//
// // Set a q table as pyarrow table
// q).pykx.set[`test;.pykx.topa ([]2?0p;2?`a`b`c;2?1f;2?0t)]
// q)print .pykx.get`test
// pyarrow.Table
// x: timestamp[ns]
// x1: string
// x2: double
// x3: duration[ns]
// ----
// x: [[2002-06-11 11:57:24.452442976,2001-12-28 01:34:14.199305176]]
// x1: [["c","a"]]
// x2: [[0.7043314231559634,0.9441670505329967]]
// x3: [[2068887000000,41876091000000]]
// ```
.pykx.set:{
  if[not -11h=type x;
    '"Expected a SymbolAtom for the attribute to set in .pykx.set"
    ];
  kwlist:import[`keyword;`:kwlist]`;
  if[x in kwlist;
    '"User attempting to overwrite Python keyword: ",string x
    ];
  util.setGlobal[x; util.convertArg[util.toDefault y]`.]}

// @kind function
// @name .pykx.get
// @category api
// @overview
// _Retrieve a named item from the Python memory_
//
// ```q
// .pykx.get[objectName]
// ```
//
// **Parameters:**
//
// name          | type      | description |
// --------------|-----------|-------------|
// `objectName` | symbol    | A named entity to retrieve from Python memory as a wrapped q foreign object. |
//
// **Return:**
//
// type          | description
// --------------|------------
// `composition` | A wrapped foreign object which can be converted to q or Python objects
//
// ```q
// // Set an item in Python memory and retrieve using .pykx.get
// q).pykx.set[`test;til 10]
// q).pykx.get[`test]
// {[f;x].pykx.util.pykx[f;x]}[foreign]enlist
//
// // Convert to q and Python objects
// q).pykx.get[`test]`
// 0 1 2 3 4 5 6 7 8 9
//
// // Retrieve an item defined entirely using Python
// q).pykx.pyexec"import numpy as np"
// q).pykx.pyexec"a = np.array([1, 2, 3])"
// q).pykx.get[`a]`
// 1 2 3
// ```
.pykx.get:ce util.wfunc util.getGlobal

// @kind function
// @name .pykx.setattr
// @category api
// @overview
// _Set an attribute of a Python object, this is equivalent to calling Python's [setattr(f, a, x)](https://docs.python.org/3/library/functions.html#setattr) function_
//
// ```q
// .pykx.setattr[pythonObject;attrName;attrObj]
// ```
//
// **Parameters:**
//
// name           | type                  | description |
// ---------------|-----------------------|-------------|
// `pythonObject` | `foreign/composition` | The Python object on which the defined attribute is to be set |
// `attrName`     | `symbol`              | The name to be associated with the set attribute |
// `attrObject`   | `any`                 | The object which is to be set as an attribute associated with `pythonObject` |
//
// **Returns:**
//
// type | description |
// -----|-------------|
// `::` | Returns generic null on successful execution otherwise returns the error message raised
//
// **Example:**
//
// ```q
// // Define a Python object to which attributes can be set
// q).pykx.pyexec"aclass = type('TestClass', (object,), {'x': pykx.LongAtom(3), 'y': pykx.toq('hello')})";
// q)a:.pykx.get`aclass
// 
// // Retrieve an existing attribute to show defined behavior
// q)a[`:x]`
// 3
// 
// // Retrieve a named attribute that doesn't exist
// q)a[`:r]`
// 
// // Set an attribute 'r' and retrieve the return
// q).pykx.setattr[a; `r; til 4]
// q)a[`:r]`
// 0 1 2 3
// q).pykx.print a[`:r]
// [0 1 2 3]
// 
// // Set an attribute 'k' to be a Pandas type
// q).pykx.setattr[a;`k;.pykx.topd ([]2?1f;2?0Ng;2?`2)]
// q)a[`:k]`
// x         x1                                   x2
// -------------------------------------------------
// 0.4931835 0a3e1784-0125-1b68-5ae7-962d49f2404d mi
// 0.5785203 5aecf7c8-abba-e288-5a58-0fb6656b5e69 ig
// q).pykx.print a[`:k]
//           x                                    x1  x2
// 0  0.493183  0a3e1784-0125-1b68-5ae7-962d49f2404d  mi
// 1  0.578520  5aecf7c8-abba-e288-5a58-0fb6656b5e69  ig
// 
// // Attempt to set an attribute against an object which does not support this behavior
// q)arr:.pykx.eval"[1, 2, 3]"
// q).pykx.setattr[arr;`test;5]
// 'AttributeError("'list' object has no attribute 'test'")
//   [1]  /opt/kx/pykx.q:218: .pykx.util.setattr:
//   cx:count x;
//   util.setAttr[unwrap x 0;x 1;;x 2]
//   ^
//     $[cx>4;
// ```
setattr:{util.setAttr[unwrap x;y;util.convertArg[util.toDefault z]`.]}

// @kind function
// @name .pykx.getattr
// @category api
// @overview
// _Retrieve an attribute or property form a foreign Python object returning another foreign._
//
// ```q
// .pykx.getattr[pythonObject;attrName]
// ```
//
// **Parameters:**
//
// name           | type                  | description
// ---------------|-----------------------|-------------
// `pythonObject` | `foreign/composition` | The Python object from which the defined attribute is to be retrieved.
// `attrName`     | `symbol`              | The name of the attribute to be retrieved.
//
// **Returns:**
//
// type      | description
// ----------|------------
// `foreign` | An unwrapped foreign object containing the retrieved
//
// !!! Note
//
//     Application of this function is equivalent to calling Python's [`getattr(f, 'x')`](https://docs.python.org/3/library/functions.html#getattr) function.
//
//     The wrapped foreign objects provide a shorthand version of calling `.pykx.getattr`. Through the use of the ````:x``` syntax for attribute/property retrieval
//
// **Example:**
//
// ```q
// // Define a class object from which to retrieve Python attributes
// q).pykx.pyexec"aclass = type('TestClass', (object,), {'x': pykx.LongAtom(3), 'y': pykx.toq('hello')})";
//
// // Retrieve the class object from Python as a q foreign
// q)show a:.pykx.get[`aclass]`.
// foreign
//
// // Retrieve an attribute from the Python foreign
// q).pykx.getattr[a;`y]
// foreign
//
// // Print the Python representation of the foreign object
// q)print .pykx.getattr[a;`y]
// hello
//
// // Retrieve the attribute from a Python foreign and convert to q
// q).pykx.wrap[.pykx.getattr[a;`y]]`
// `hello
// ```
getattr;      // Note this function is loaded directly from C


// @kind function
// @name .pykx.pycallable
// @category api
// @overview
// _Convert a Python foreign object to a callable function which returns a Python foreign result_
//
// ```q
// .pykx.pycallable[pyObject]
// ```
//
// **Parameters:**
//
// name         | type      | description                                  
// -------------|-----------|-------------
// `pyObject`   | `foreign` | A Python object representing an underlying callable function
// 
// **Returns:**
//
// type      | description
// ----------|------------
// `foreign` | The return of the Python callable function as a foreign object
//
// **Example:**
//
// ```q
// q)wrappedPy:.pykx.import[`numpy;`:arange]
// q)show setCallable:.pykx.pycallable[wrappedPy][1;3]
// foreign
// q).pykx.print setCallable
// [1 2]
// ```
pycallable:{$[util.isw x;wrap[unwrap[x]](>);util.isf x;wrap[x](>);'"Could not convert provided function to callable with Python return"]}

// @kind function
// @name .pykx.qcallable
// @category api
// @overview
// _Convert a Python foreign object to a callable function which returns a q result_
//
// ```q
// .pykx.qcallable[pyObject]
// ```
//
// **Parameters:**
//
// name         | type      | description
// -------------|-----------|-------------
// `pyObject`   | `foreign` | A Python object representing an underlying callable function
// 
// **Returns:**
//
// type  | description
// ------|------------
// `any` | The return of the Python callable function as an appropriate q object
//
// **Example:**
//
// ```q
// q)wrappedPy:.pykx.import[`numpy;`:arange]
// q)show setCallable:.pykx.pycallable[wrappedPy][1;3]
// foreign
// q).pykx.print setCallable
// [1 2]
// ```
qcallable:{$[util.isw x;wrap[unwrap[x]](<);util.isf x;wrap[x](<);'"Could not convert provided function to callable with q return"]}

// @kind function
// @name .pykx.safeReimport
// @category api
// @overview
// _Isolated execution of a q function which relies on importing PyKX_
//
// ```q
// .pykx.safeReimport[qFunction]
// ```
//
// **Parameters:**
//
// name         | type       | description
// -------------|------------|-------------
// `qFunction`  | `function` | A function which is to be run following unsetting of PyKX environment variables and prior to their reset
//
// **Returns:**
//
// type   | description
// -------|------------
// `any`  | On successful execution this function will return the result of the executed function
//
// **Example:**
//
// ```q
// q)\l pykx.q
// q).pykx.safeReimport[{system"python -c 'import pykx as kx'";til 5}]
// 0 1 2 3 4
// ```
safeReimport:{[x]
  pyexec["pykx_internal_reimporter = pykx.PyKXReimport()"];
  envlist:(`PYKX_DEFAULT_CONVERSION;
    `PYKX_UNDER_Q;
    `SKIP_UNDERQ;
    `PYKX_SKIP_UNDERQ;
    `PYKX_UNDER_PYTHON;
    `UNDER_PYTHON;
    `PYKX_LOADED_UNDER_Q;
    `PYKX_Q_LOADED_MARKER;
    `PYKX_EXECUTABLE;
    `PYKX_DIR);
  envvals:getenv each envlist;

  .pykx.eval["pykx_internal_reimporter.reset()"];
  r: @[{(0b;x y)}x;(::);{(1b;x)}];

  pyexec["del pykx_internal_reimporter"];
  setenv'[envlist;envvals];
  $[r 0;';::] r 1
  }

// @kind function
// @name .pykx.debugInfo
// @category api
// @overview
// _Library and environment information which can be used for environment debugging_
//
// ```q
// .pykx.debugInfo[]
// ```
//
// **Returns:**
//
// type   | description
// -------|------------
// `list` | A list of strings containing information useful for debugging
//
// **Example:**
//
// ```q
// q).pykx.debugInfo[]
// "**** PyKX information ****"
// "pykx.args: ()"
// "pykx.qhome: /usr/local/anaconda3/envs/qenv/q"
// "pykx.qlic: /usr/local/anaconda3/envs/qenv/q"
// "pykc.licensed: True"
// ..
// ```
debugInfo:{
  pykxQHeader:enlist"**** PyKX under q Information ****";
  pykxQInfo  :{string[x 0],": ",x 1}each flip(key;value)@\:.pykx.debug;
  pykxPythonInfo:"\n" vs string .pykx.import[`pykx;`:util.debug_environment][pykwargs enlist[`return_info]!enlist 1b]`;
  pykxPythonInfo,pykxQHeader,pykxQInfo
  }

// @kind function
// @name .pykx.console
// @category api
// @overview
// _Open an interactive python REPL from within a q session similar to launching python from the command line._
//
// ```q
// .pykx.console[]
// ```
//
// **Returns:**
//
// type | description
// -----|------------
// `::` | This function has no explicit return but execution of the function will initialise a Python REPL.
//
// **Example:**
//
// ```q
// Enter PyKX console and evaluate Python code
// q).pykx.console[]
// >>> 1+1
// 2
// >>> list(range(10))
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// >>> quit()
// q)
//
// // Enter PyKX console setting q objects using PyKX
// q).pykx.console[]
// >>> import pykx as kx
// >>> kx.q['table'] = kx.q('([]2?1f;2?0Ng;2?`3)'
// >>> quit()
// q)table
// x         x1                                   x2 
// --------------------------------------------------
// 0.439081  49f2404d-5aec-f7c8-abba-e2885a580fb6 mil
// 0.5759051 656b5e69-d445-417e-bfe7-1994ddb87915 igf
//
// // Enter PyKX console setting Python objects using PyKX
// q).pykx.console[]
// >>> a = list(range(5))
// >>> quit()
// q).pykx.eval["a"]`
// 0 1 2 3 4
// ```
console:{pyexec"from code import InteractiveConsole\n__pykx_console__ = InteractiveConsole(globals())\n__pykx_console__.push('import sys')\n__pykx_console__.push('quit = sys.exit')\n__pykx_console__.push('exit = sys.exit')\ntry:\n    line = __pykx_console__.interact(banner='', exitmsg='')\nexcept SystemExit:\n    pykx._pykx_helpers.clean_errors()"}

// @private
// @desc
// Set the execution function used when loading files with the extension `*.p`
// or when using the following syntax `p)<python code>` within a q session
.p.e:{.pykx.pyexec x}     // If changing this line please ensure you have updated the check used at the beginning of this file to warn users about PyKX being loaded with other Python libraries

// @private
// @desc
// Set default conversion type for K objects.
setdefault {$[""~c:getenv`PYKX_DEFAULT_CONVERSION;"default";c]}[];

// @private
// @desc
// Finalise loading of PyKX functionality setting environment variables
// needed to ensure loading PyKX multiple times does not result in unexpected errors
finalise[];

// @private
// @kind function
// @name .pykx.listExtensions
// @category api
// @overview
// _List all q scripts in the extensions directory which can be loaded_
//
// ```q
// .pykx.listExtensions[]
// ```
//
// **Returns:**
//
// type   | description
// -------|------------
// `list` | A list of strings denoting the available extensions in your version of PyKX
//
// **Example:**
//
// ```q
// q)\l pykx.q
// q).pykx.listExtensions[]
// "dashboards"
// ```
listExtensions:{-2 _/:lst where like[;"*.q"]lst:string key hsym`$pykxDir,"/extensions/"}

// @private
// @kind function
// @name .pykx.loadExtension
// @category api
// @overview
// _Loading of a PyKX extension_
//
// ```q
// .pykx.loadExtension[ext]
// ```
//
// **Parameters:**
//
// name   | type     | description
// -------|----------|-------------
// `ext`  | `string` | The name of the extension which is to be loaded
//
// **Returns:**
//
// type   | description
// -------|------------
// `null` | On successful execution this function will load the extension and return null
//
// **Example:**
//
// ```q
// q)\l pykx.q
// q)`dash in key `.pykx
// 0b
// q).pykx.listExtensions[]
// "dashboards"
// q)`dash in key `.pykx
// 1b
// ```
loadExtension:{[ext]
  if[not 10h=type ext;'"Extension provided must be of type string"];
  if[not ext in listExtensions[];'"Extension provided '",ext,"' not available"];
  .[util.loadfile;
    (pykxDir,"/extensions/";ext,".q");
    {'x," raised when attempting to load extension"}
    ];
  }

// @desc Restore context used at initialization of script
system"d ",string .pykx.util.prevCtx;
