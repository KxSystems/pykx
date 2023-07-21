.test.handle:-1i;
.z.pw:{[u;p]
    if[u ~ `test;.test.handle:.z.w];
    :1b;
    }
.z.ac:{
    if[(`$"x-retry-id") in key x[1];
        $[(id:x[1][`$"x-retry-id"]) in key .test.ids;
            .test.ids[id]:.test.ids[id] - 1;
            .test.ids[id]:"I"$x[1][`$"x-retry"]
            ];
        if[.test.ids[id] <> 0;
            :(2;.h.hn["503";`txt;"try again"]);
            ]
        ];
    :(2;.h.hn["200";`txt;x[1]`Authorization])
    }
.z.pp:{.h.hy[`txt] "hello"}
.z.pc:{
    if[x ~ .test.handle;exit 0];
    };
.test.ids:enlist[""]!enlist 0;
