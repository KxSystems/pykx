// Reference count and GIL checks for the pykx-under-q C library (KXI-73830).
// Results are printed as `prefix,key,value` for tests/test_underq_refcounts.py.

\l pykx.q

N:500;
report:{-1"refcounts_under_q,",x,",",string y;y};

.pykx.pyexec"import sys";
rc:{.pykx.eval["sys.getrefcount(",x,")"]`};

// Growth per call, with one warm-up call so one-off interning is not counted.
perOp:{[name;expr;op]
  op[];
  before:rc expr;
  do[N;op[]];
  report[name;"j"$(rc[expr]-before)%N]
  };

// Built-in exception types are immortal on Python 3.12+, so a leaked reference
// to one is not observable. Use a heap-allocated class instead.
.pykx.pyexec"class Boom(Exception): pass";
perOp["exception_type";"Boom";
  {@[{.pykx.eval["(_ for _ in ()).throw(Boom())"]`};::;{x;}]}];

.pykx.pyexec"import math";
perOp["module";"math";{.pykx.import`math}];

.pykx.pyexec"gv = object()";
perOp["global_object";"gv";{.pykx.get`gv}];

.pykx.pyexec"import types";
.pykx.pyexec"holder = types.SimpleNamespace()";
perOp["attr_owner";"holder";
  {.pykx.setattr[.pykx.unwrap .pykx.eval["holder"];`a;1]}];

// repr leaked the encoded bytes rather than a reference, so it shows up as memory.
.pykx.pyexec"big = list(range(100000))";
rss:{.pykx.eval["int(open('/proc/self/statm').read().split()[1])"]`};
{[] x:rss[];do[N;.pykx.repr .pykx.unwrap .pykx.eval["big"]];
   report["repr_pages_per_op";"j"$(rss[]-x)%N]}[];

// The non-callable early return skipped PyGILState_Release. A Python thread makes
// no progress while the GIL is held, so its tick count is the probe.
.pykx.pyexec"import threading, time";
.pykx.pyexec"ticks = []";
.pykx.pyexec"threading.Thread(target=lambda: [(ticks.append(1), time.sleep(0.005)) for _ in range(10000)], daemon=True).start()";
.pykx.pyexec"time.sleep(0.2)";
.pykx.pyexec"nc = object()";
report["non_callable_errors";
  "j"$"caught"~6#@[{.pykx.util.callFunc[.pykx.unwrap .pykx.eval["nc"];0;();(enlist(`symbol$());enlist(::))]};
                   ::;{"caught: ",x}]];
ticksBefore:.pykx.eval["len(ticks)"]`;
system"sleep 1";
report["thread_ticks_while_idle";(.pykx.eval["len(ticks)"]`)-ticksBefore];

-1"refcounts_under_q,PASS";
exit 0;
