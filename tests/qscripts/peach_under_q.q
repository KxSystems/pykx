if[not `pykx in key `;system"l pykx.q"];

n:2000;

sq:.pykx.eval["lambda x: x*x"];
sres:{[i] sq[i]`} peach til n;
sok:(n=count sres) and sres~{x*x} til n;

vq:.pykx.eval["lambda x: range(x + 1)"];
vres:{[i] vq[i]`} peach til n;
vok:(n=count vres) and vres~{til x+1} each til n;

ok:sok and vok;
-1"peach_under_q,slaves,",string system"s";
-1"peach_under_q,scalar_ok,",string sok;
-1"peach_under_q,vector_ok,",string vok;
-1$[ok;"peach_under_q,PASS";"peach_under_q,FAIL"];
exit $[ok;0;1];
