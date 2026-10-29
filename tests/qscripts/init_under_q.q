// k_init_python must not release its arguments or corrupt memory on load,
// reload, or once other references to the values are dropped (KXI-69194).
// Results are printed as they are computed, so a crash shows how far it got.

if[not `pykx in key `;system"l pykx.q"];

report:{-1"init_under_q,",x,",",string y;y};

// The two counts should agree
rcSupported:report["refcount_supported";@[{-16!x;1b};"probe";0b]];
homeRC:$[rcSupported;-16!.pykx.debug`pythonHome;0N];
execRC:$[rcSupported;-16!.pykx.debug`pythonExec;0N];
report["refcount_home";homeRC];
report["refcount_exec";execRC];
rcOk:report["refcount_ok";$[rcSupported;homeRC=execRC;1b]];

system"l pykx.q";
reloadOk:report["reload_ok";@[{2=.pykx.eval["1+1"]`};::;0b]];

// Compare as symbols so the check holds no reference to the char vectors
expLib :`$.pykx.debug`pythonLib;
expHome:`$.pykx.debug`pythonHome;
expBase:`$.pykx.debug`pythonBase;
.pykx.util.pyEnvInfo:();
.pykx.util.pyPath:();
holdersOk:report["holders_ok";all(expLib ~`$.pykx.debug`pythonLib;
                                  expHome~`$.pykx.debug`pythonHome;
                                  expBase~`$.pykx.debug`pythonBase)];

pyOk:report["python_ok";@[{4=.pykx.eval["2+2"]`};::;0b]];

ok:all(rcOk;reloadOk;holdersOk;pyOk);
-1$[ok;"init_under_q,PASS";"init_under_q,FAIL"];
exit $[ok;0;1];
