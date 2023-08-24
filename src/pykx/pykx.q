.pykx.i.prevCtx:system"d";
\d .pykx

// TO-DO
//
// Need to add logging for things that are supported in early versions of this new
// version of embedPy functionality that we will be deprecating (.pykx.py2q will migrate
// to .pykx.toq)

// Retrieve any startup flags provided by the user
i.startup:.Q.opt .z.x
i.o:first string .z.o;

if[""~getenv`PYKX_LOADED_UNDER_Q; [
  i.dirCommand:"-c \"import pykx; print(pykx.config.pykx_dir)\" 2>",$[i.o="w";"nul <nul";"/dev/null"];
  if[""~pykxDir::getenv`PYKX_DIR;
    pykxDir::first @[system"python ",;i.dirCommand;{system"python3 ",i.dirCommand}];
    pykxDir::ssr[pykxDir; "\\"; "/"]];
 ]];


// This env var tells PyKX that it should not load embedded q, as q symbols are already
// defined in the process.
setenv[`PYKX_UNDER_Q;"true"];

// Compose a list of functions
k)c:{'[y;x]}/|:
// Compose using enlist for generation of variadic functions
k)ce:{'[y;x]}/enlist,|:


if[not ""~getenv`UNDER_PYTHON; i.load:2:[hsym`$pykxDir,"/pykx";]];
if[""~getenv`PYKX_LOADED_UNDER_Q; [
  i.pyEnvInfo:: ("None"; "None"; "");
  if [count[getenv[`PYKX_Q_LOADED_MARKER]]=0; system "l pykx_init.q_"];
  ]];
.pykx.preinit[];
\d .pykx

// Find the path to where PyKX is installed, likely in site-packages
if[pykxDir~"/pykx";
    '"Failed to find pykx - ensure your Python environment is properly configured and activated"];

// Set default conversion type for K objects to np unless otherwise specified with an env var.
i.defaultConv:$[""~i.x:getenv`PYKX_DEFAULT_CONVERSION;"np";$[i.x in ("py";"np";"pd";"pa";"k"); x; '"Unknown default conversion type"]];

// Update default conversion type for K objects at runtime
setdefault:{x:lower x;.pykx.i.defaultConv:$[x in ("np";"numpy");"np";x in ("py";"python");"py";x in (enlist"k" ;enlist"q"); "k";x in ("pd";"pandas");"pd";x in ("pa";"pyarrow");"pa";'"unknown conversion type: ",x]}

// Convert a Python foreign object to q definition of py2q maintained for backwards compatibility with embedPy
py2q:toq:{$[type[x]in 104 105 112h;i.load[(`foreign_to_q;1)]unwrap x;x]}

// Convert q/Python objects to Pythonic foreigns
i.toPython:{wrap i.load[(`k_to_py_foreign;3)][$[type[x]in 104 105 112h;wrap[unwrap x]`;x];y;z]}

i.isch  :{$[(104= type y);$[x~y[::]0;1b;0b];0b]}
i.ispy  :i.isch[`..python]
i.isnp  :i.isch[`..numpy]
i.ispd  :i.isch[`..pandas]
i.ispa  :i.isch[`..pyarrow]
i.isk   :i.isch[`..k]
i.israw :i.isch[`..raw]
i.isconv:{any(i.ispy;i.isnp;i.ispd;i.ispa;i.isk;i.israw)@\:x}
i.convertArg:{$[i.isconv x; .z.s[(x; 1; 0b)];$[not i.isconv x 0; i.toPython . x;i.ispy  x 0; [.z.s[(x[0][::][1]; 1; x[2])]];i.isnp  x 0; [.z.s[(x[0][::][1]; 2; x[2])]];i.ispd  x 0; [.z.s[(x[0][::][1]; 3; x[2])]];i.ispa  x 0; [.z.s[(x[0][::][1]; 4; x[2])]];i.isk   x 0; [.z.s[(x[0][::][1]; 5; x[2])]];i.israw x 0; [.z.s[(x[0][::][1]; x[1]; 1b)]]]]};
i.toDefault:{$[i.isconv x;(::);"py"~i.defaultConv;topy;"np"~i.defaultConv;tonp;"pd"~i.defaultConv;topd;"pa"~i.defaultConv;topa;"k"~i.defaultConv;tok;(::)]x};

topy:{x y}(`..python;;)       / identify python conversion
tonp:{x y}(`..numpy;;)        / identify numpy conversion
topd:{x y}(`..pandas;;)       / identify pandas conversion
topa:{x y}(`..pyarrow;;)      / identify pyarrow conversion
tok: {x y}(`..k;;)            / identify K conversion
toraw: {x y}(`..raw;;)        / identify raw conversion

// Foreign object wrapping and manipulation
i.pykx:{[f; x]
  f:unwrap f;
  $[-11h<>t:type x0:x 0;
    $[t=102h;
      $[any u:x0~/:(*;<;>);
        [c:(wrap;toq;::)where[u]0;$[1=count x;.pykx.c c,;c .[;1_x]@]pyfunc f];
        (:)~x0;[setattr . f,@[;0;{`$_[":"=s 0]s:string x}]1_x;];
        (@)~x0;[
          if["None"~repr fn:wrap[f][`:__getitem__];'"Python object has no attribute __getitem__."];
          $[count 2_x;.[;2_x];]fn x 1
          ];
        (=)~x0;[
          if["None"~repr fn:wrap[f][`:__setitem__];'"Python object has no attribute __setitem__."];
          fn . (x 1;x 2)
          ];
        '`NYI
        ];
      wrap pyfunc[f] . x];
      ":"~first a0:string x0;
      $[1=count x;;.[;1_x]]wrap f getattr/` vs`$1_a0;
      x0~`.;f;x0~`;toq f;
      wrap pyfunc[f] . x]
  }


// Wrapping helping functionality
i.wf:{[f;x].pykx.i.pykx[f;x]}
i.isw:{$[105=type x;.pykx.i.wf~$[104 105h~t:type each u:get x;:.z.s last u;104h~first t;first value first u;0b];0b]}

// Wrapping and unwrapping functionality
wrap:ce i.wf@
unwrap:{$[i.isw x;$[104 105h~type each u:get x;(last u)`.;x`.];x]}
wfunc:{[f;x]r:wrap f x 0;$[count x:1_x;.[;x];]r}


// Replace check here to use C instead of q and discern if it's actually a foreign
pyfunc:{if[not 112h=type x;'`type];ce .[i.load[(`call_func;4)]x],`.pykx.i.parseArgs}

// Language specific wrapping functionality
wrapq:ce {[f;x]i.pykx[f;x]`}@
wrappy:ce {[f;x]i.pykx[f;x]`.}@


// Python evaluation functionality
i.pyrun:i.load (`k_pyrun;4)
pyevalNoRet:i.pyrun[0b; 0b; 0b]
pyeval:i.pyrun[1b; 0b; 1b]
pyexec:i.pyrun[1b; 1b; 0b]
.pykx.eval:{wrap pyeval x}
qeval:{toq .pykx.eval x}
.p.e:{.pykx.pyexec x}


// Import functionality imports and makes available a module to be introspected and used
import:ce wfunc i.load(`import;1)

// Functionality for management of keywords/keyword dictionaries etc.
i.iskw  :i.isch[`..pykw]
i.isargl:i.isch[`..pyas]
i.iskwd :i.isch[`..pyks]
i.isarg :{any(i.iskw;i.isargl;i.iskwd)@\:x}

.q.pykw     :{x[y;z]}(`..pykw;;;)  / identify keyword args with `name pykw value
.q.pyarglist:{x y}(`..pyas;;)      / identify pos arg list (*args in python)
.q.pykwargs :{x y}(`..pyks;;)      / identify keyword dict (**kwargs in python)

i.parseArgs:{
  hasargs:$[(x~enlist[::])&1=count x;0;1];
  kwlist:x where i.iskw each x;
  kwdict:$[1<count[x where i.iskwd each x];
    '"Expected only one key word dictionary to be used in function call";
    $[0=count[x where i.iskwd each x]; ()!(); (x where i.iskwd each x)[0][::]1]
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
  kwargs:(key kwargs)!({[kwargs; x] unwrap i.convertArg i.toDefault kwargs[x]}[kwargs;] each key kwargs);
  (hasargs; {unwrap i.convertArg i.toDefault x} each (x where not i.isarg each x),$[0<count[x where i.isargl each x]; $[1<count[x where i.isargl each x]; '"Expected only one arg list to be using in function call"; (x where i.isargl each x)[0][::]1]; x where i.isargl each x]; kwargs)
  };


// Retrieve print/display representation of evaluated Python
i.repr: i.load (`repr; 2);
repr :{$[type[x]in 104 105 112h;i.repr[1b] unwrap x;.Q.s x]}
print:{$[type[x]in 104 105 112h;i.repr[0b] unwrap x;show x];}


// Functionality for setting and getting items from Python memory
.pykx.set:{i.load[(`set_global;2)][x; i.convertArg[i.toDefault y]`.]}
setattr:{i.load[(`set_attr;3)][unwrap x;y;i.convertArg[i.toDefault z]`.]}
.pykx.get:ce wfunc i.load[(`get_global;1)];
getattr:i.load (`get_attr;2)


safeReimport:{[x]
  pyexec["pykx_internal_reimporter = pykx.PyKXReimport()"];
  i.pykxunderq: getenv`PYKX_UNDER_Q;
  i.skipunderq: getenv`SKIP_UNDERQ;
  i.underpython: getenv`UNDER_PYTHON;
  i.pykxqloadedmarker: getenv`PYKX_Q_LOADED_MARKER;
  i.pykxloadedunderq: getenv`PYKX_LOADED_UNDER_Q;
  .pykx.eval["pykx_internal_reimporter.reset()"];
  r: x[];
  pyexec["del pykx_internal_reimporter"];
  setenv[`PYKX_UNDER_Q;i.pykxunderq];
  setenv[`SKIP_UNDERQ;i.skipunderq];
  setenv[`UNDER_PYTHON;i.underpython];
  setenv[`PYKX_Q_LOADED_MARKER;i.pykxqloadedmarker];
  setenv[`PYKX_LOADED_UNDER_Q;i.pykxloadedunderq];
  r
  };


console:{pyexec"pykx.console.PyConsole().interact(banner='', exitmsg='')"};


if[(not ""~getenv`UNSET_PYKX_GLOBALS)|not `unsetPyKXGlobals in i.startup;
  {@[`.;x;:;get x]}each `print
  ]

.pykx.finalise[];

system"d ",string .pykx.i.prevCtx;
