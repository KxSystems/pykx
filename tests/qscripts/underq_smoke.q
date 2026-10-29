// Smoke check that pykx.q loads and runs inside a real q process (KXI-46558).
//
// The calls below between them reach every q allocator pykxq.dll uses, so a symbol bound to the
// wrong interpreter shows up here: krr on load, then knk, kp, ks, r0/r1, kb, and kj/ktn.
// Results print as they are computed, so a crash partway through says how far it got.

report:{-1"underq_smoke,",x,",",string y;y};

loaded:report["load";@[{system"l pykx.q";1b};::;{-1"underq_smoke,load_error,",x;0b}]];

// Guard on `loaded` so the later checks do not report an absent .pykx namespace as their own
// failure.
check:{[loaded;name;f;fail]report[name;$[loaded;@[f;::;{[fail;e]e;fail}fail];0b]]}[loaded];

evalOk :check["eval"  ;{2=.pykx.eval["1+1"]`}                     ;0b];
reprOk :check["repr"  ;{10h=type .pykx.repr .pykx.eval"1+1"}      ;0b];
printOk:check["print" ;{.pykx.print .pykx.eval"1+1";1b}           ;0b];
// A Python exception must come back as a q error, so the handler is the passing case here.
errOk  :check["error" ;{.pykx.eval["1/0"]`;0b}                    ;1b];
listOk :check["list"  ;{1 2 3~.pykx.eval["[1,2,3]"]`}             ;0b];
isfOk  :check["isf"   ;{1b~.pykx.util.isf .pykx.eval["1"]`.}      ;0b];
modpowOk:check["modpow";{(24=.pykx.modpow[2;10;1000])and 24 49~.pykx.modpow[2 3;10 10;1000 1000]};0b];

ok:all(loaded;evalOk;reprOk;printOk;errOk;listOk;isfOk;modpowOk);
-1$[ok;"underq_smoke,PASS";"underq_smoke,FAIL"];
exit $[ok;0;1];
