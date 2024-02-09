if[not `pykx in key `;system"l pykx.q"];
-1"stdouterrtest,1,q stdout";
-2"stdouterrtest,2,q stderr";
.pykx.pyexec"import sys"
.pykx.eval"print('stdouterrtest,3,eval print')";
.pykx.eval"print('stdouterrtest,4,eval print file=sys.stdout', file=sys.stdout)";
.pykx.eval"print('stdouterrtest,5,eval print file=sys.stderr', file=sys.stderr)";
.pykx.pyexec"print('stdouterrtest,6,pyexec print')";
.pykx.pyexec"print('stdouterrtest,7,pyexec print file=sys.stdout', file=sys.stdout)";
.pykx.pyexec"print('stdouterrtest,8,pyexec print file=sys.stderr', file=sys.stderr)";
-1 "stdouterrtest,9,",.pykx.repr .pykx.eval"'.pykx.repr'";
.pykx.print .pykx.eval"'stdouterrtest,10,.pykx.print'";
.pykx.print .pykx.tonp .pykx.eval"'stdouterrtest,11,.pykx.print .pykx.tonp .pykx.eval'";
.pykx.eval["lambda x:print(x)"] `$"stdouterrtest,12,.pykx.eval[\"lambda x:print(x)\"]";
.pykx.eval["(lambda x:print(x))('stdouterrtest,13,inside a lambda')"] ;
.pykx.eval["print('stdouterrtest,14,not in a lambda')"] ;
.pykx.eval"print('stdouterrtest,15,eval print')";
.pykx.print .pykx.tonp .pykx.eval"'stdouterrtest,16,.pykx.print .pykx.tonp .pykx.eval'";
.pykx.eval["lambda x:print(x)"][`$"stdouterrtest,17,.pykx.eval[\"lambda x:print(x)\"]"]`;
.pykx.pyexec"def func(x):\n  print(x)";  
func:.pykx.get`func;
func `$"stdouterrtest,18,python function";
func[`$"stdouterrtest,19,python function`"]`;
.pykx.pyexec["(lambda x:print(x))('stdouterrtest,20,inside a lambda')"] ;
.pykx.pyexec["print('stdouterrtest,21,not in a lambda')"] ;
.pykx.pyexec"print('stdouterrtest,22,eval print')";
exit 0