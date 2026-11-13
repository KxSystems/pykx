// Calling pykxq.so before k_pykx_init has run must raise q errors, not SIGSEGV (KXI-73830)

opts:.Q.opt .z.x;
lib:2:[hsym`$(first opts`pykxdir),"/pykxq";];
lib[(`k_init_python;3)] . first each opts`libpython`pyhome`pyexec;

report:{-1"uninit_pyrun,",x,",",y;y};

run:{.[lib[(`k_pyrun;4)];(0b;1b;0b;x);{"QERR: ",x}]};

report["import_error";run"raise ImportError('simulated GLIBC/libffi failure')"];
report["success_after";$[(::)~run"x = 1";"ok";"unexpected"]];
report["second_error";run"raise ValueError('second error')"];
report["foreign_to_q";.[lib[(`foreign_to_q;3)];(::;0b;0b);{"QERR: ",x}]];

-1"uninit_pyrun,PASS";
exit 0;
