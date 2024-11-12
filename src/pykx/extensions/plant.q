\d .u

d:.z.D

init:{w::t!(count t::tables`.)#()}

del:{w[x]_:w[x;;0]?y};.z.pc:{del[;x]each t};

sel:{$[`~y;x;select from x where sym in y]}

pub:{[t;x]{[t;x;w]if[count x:sel[x]w 1;(neg first w)(`upd;t;x)]}[t;x]each w t}

add:{$[(count w x)>i:w[x;;0]?.z.w;.[`.u.w;(x;i;1);union;y];w[x],:enlist(.z.w;y)];(x;$[99=type v:value x;sel[v]y;@[0#v;`sym;`g#]])}

sub:{if[x~`;:sub[;y]each t];if[not x in t;'x];del[x].z.w;add[x;y]}

end:{(neg union/[w[;;0]])@\:(`.u.end;x)}

ld:{
  L::`$(-10_string L),string x;
  if[a~key a:hsym L;
    if[.tick.hardReset;@[hdel;a;{x}]]
    ];
  if[not type key L;
    .[L;();:;()]
    ];
  i::j::-11!(-2;L);
  if[0<=type i;
    -2 (string L)," is a corrupt log. Truncate to length ",
    (string last i),
    " and restart";exit 1
    ];
  hopen L
  };

tick:{
  init[];
  @[;`sym;`g#]each t;
  d::.z.D;
  if[l::count y;
    y:$[0h=type y;raze;]y;
    L::`$":",y,"/",x,10#".";
    l::ld d
    ]
  };

endofday:{end d;d+:1;if[l;hclose l;l::0(`.u.ld;d)]};
ts:{if[d<x;if[d<x-1;system"t 0";'"more than one day?"];endofday[]]};

if[system"t";
 .z.ts:{pub'[t;value each t];@[`.;t;@[;`sym;`g#]0#];i::j;ts .z.D};
 upd:{[t;x]
 if[not -16=type first first x;if[d<"d"$a:.z.P;.z.ts[]];a:"n"$a;x:$[0>type first x;a,x;(enlist(count first x)#a),x]];
 t insert x;if[l;l enlist (`upd;t;x);j+:1];}];

if[not system"t";
 system"t 100";
 .z.ts:{ts .z.D};
 upd:{[t;x]ts"d"$a:.z.P;
 if[not -16=type first first x;a:"n"$a;x:$[0>type first x;a,x;(enlist(count first x)#a),x]];
 f:key flip value t;pub[t;$[0>type first x;enlist f!x;flip f!x]];if[l;l enlist (`upd;t;x);i+:1];}];

\d .

.tick.tabs:()

.tick.init:{[config]
  show config;
  tick_config:enlist[`log_prefix]!enlist "log";
  tick_config,:config;
  .u.tick[tick_config`log_prefix;.tick.logdir];
  }

.tick.set_tables:{[tabname;schema]
  .tick.tabs,:enlist[tabname];
  tabname set schema
  }
