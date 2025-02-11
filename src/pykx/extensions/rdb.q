/ end of day: save, clear, hdb reload
.u.end:{
  t:tables`.;
  t@:where `g=attr each t@\:`sym;
  .Q.hdpf[`$":",.u.x 1;`:.;x;`sym];
  @[;`sym;`g#] each t;
  };

/ init schema and sync up from log file;cd to hdb(so client save can run)
.u.rep:{
  (.[;();:;].)each y;
  if[null first z;:()];
  -11!z;
  system "cd ",x
  };

upd:$[.tick.vanilla;
  insert;
  {
    pre:.tick.RTPPreProc[x;y];
    if[pre~(::);:()];
    ret:x insert pre;
    .tick.RTPPostProc[x;y];
    ret
  }
  ];

\d .tick

loaded:"RDB"
subscriptions:`

// Default analytic operates as a pass through
RTPPreProc:{[x;y]y}
RTPPostProc:{[x;y]:()}

init:{[config]
  rdb_config:`tickerplant`hdb`database!(":5010";":5012";"db");
  rdb_config,:string config;
  .u.x:rdb_config`tickerplant`hdb;
  .u.rep[rdb_config`database] . hopen[`$":",.u.x 0]("{(.u.sub[;`]each x;`.u `i`L)}";.tick.subscriptions)
  }

tabs:()

set_tables:{[tabname;schema]
  tabs,:enlist[tabname];
  tabname set schema
  }
