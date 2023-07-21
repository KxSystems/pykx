if[not 5140=system"p";
   -1"For the purposes of this example -p must be set to 5140, setting port accordingly\n";
   system"p 5140"]

if[not system"t";system"t 1000"]

system"l tick/tick/",(src:first .z.x,enlist"sym"),".q"

\l tick/tick/u.q

\d .u

ld:{
  if[not type key L::`$(-10_string L),string x;.[L;();:;()]];
  i::j::-11!(-2;L);
  if[0<=type i;-2 (string L)," is a corrupt log. Truncate to length ",(string last i)," and restart";exit 1];
  hopen L
  };

tick:{
  init[];
  if[not min(`time`sym~2#key flip value@)each t;
    '`timesym
    ];
  @[;`sym;`g#]each t;
  d::.z.D;
  if[l::count y;
    L::`$":",y,"/",x,10#".";
    l::ld d
    ]
  }

endofday:{
  end d;
  d+:1;
  if[l;
    hclose l;
    l::0(`.u.ld;d)
    ]
  }

ts:{
  if[d<x;
    if[d<x-1;
      system"t 0";
      '"more than one day?"
      ];
    endofday[]
    ]
  }

if[system"t";
  .z.ts:{timer+:system["t"]%1000;
    pub'[t;value each t];
    @[`.;t;@[;`sym;`g#]0#];
    i::j;
    ts .z.D
    };
  upd:{[t;x]
    if[not -16=type first first x;
      if[d<"d"$a:.z.P;
        .z.ts[]
        ];
      a:"n"$a;
      x:$[0>type first x;a,x;(enlist(count first x)#a),x]
      ];
    t insert x;
    if[l;l enlist (`upd;t;x);j+:1];
    }
  ];

\d .
.u.tick[src;.z.x 1];

