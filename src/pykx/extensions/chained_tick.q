\d .u

d:.z.D

init:{w::t!(count t::tables`.)#()}

del:{w[x]_:w[x;;0]?y};.z.pc:{del[;x]each t};

sel:{$[`~y;x;select from x where sym in y]}

pub:{[t;x]{[t;x;w]if[count x:sel[x]w 1;(neg first w)(`upd;t;x)]}[t;x]each w t}

add:{$[(count w x)>i:w[x;;0]?.z.w;.[`.u.w;(x;i;1);union;y];w[x],:enlist(.z.w;y)];(x;$[99=type v:value x;sel[v]y;@[0#v;`sym;`g#]])}

sub:{if[x~`;:sub[;y]each t];if[not x in t;'x];del[x].z.w;add[x;y]}

\d .

if[system"t";
  .z.ts:{.u.pub'[.u.t;value each .u.t];@[`.;.u.t;@[;`sym;`g#]0#]};
  upd:{[t;x] t insert x;}
  ]

if[not system"t";
  system"t 1000";
  upd:{[t;x] .u.pub[t;x];}
  ]

/ get the ticker plant port, default is 5010
.u.x:.z.x,(count .z.x)_enlist":5010"

/ init schema 
.u.rep:{(.[;();:;].)each x;}

.tick.init:{[ports]
  conn_port:enlist[`tickerplant]!enlist":5010";
  conn_ports,:string ports;
  .u.init .u.rep(.u.m:hopen`$":",conn_ports`tickerplant)".u.sub[`;`]"
  }
