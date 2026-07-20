\d .

.pykx.i.kxic.loadfailed:()!();
.comkxic.Kf:{[v]r:,/'$2\:'ri:"I"$raw:"."\:v:v@&v in .Q.n,".";if[max "b"$ri>1023;'version];vi:sum/ 2/:'.:'"0"^(r[0],20#"0";-20$r[1],10#"0";-10$r 2),\:"b";"***III"$`K`k`Kl`major`minor`patch!enlist[v],enlist["D"$"2022.03.10"],vi,raw};
.[`.comkxic;();,;.comkxic.Kf "3.0.0"];
.[`.comkxic.libs;();,;`nelf`kurl`objstor`qlog`restserver`bq`sql`l64!("6a93b2b";"7fa6ae8";"32363d0";"a49dc65";"00a5b89";"14d547d";"99f3988";"3fa87f1")];
if[(.z.o~`l64)&$[0=#.z.x;in[_getenv`$"PYKX_QCE";("true";,"1")];"--qce"in .z.x];
 {[]
   lf:{@[{."\\l ",$[""~x:getenv`QHOME;getenv[`HOME],"/q";x],"/",($x),"_"};x;{.pykx.i.kxic.loadfailed,:enlist[x]!enlist y}[x]]};
   d:`kurl.q`objstor.q`qlog.q`s.k!$(`insights.lib.kurl`krl;`insights.lib.objstore`obj;`insights.lib.qlog`lg;`insights.lib.sql`sql);
   if[@[{2<#: .s};(::);{0b}];d:d _ `s.k];
   if[~0=#.z.x;
      if[~:in[_getenv`$"PYKX_QCE";("true";,"1")]|("--qce"in .z.x);
         if[~:"--kurl"in .z.x;d:d _ `kurl.q];
         if[~:"--objstor"in .z.x;d:d _ `objstor.q];
         if[~:"--qlog"in .z.x;d:d _ `qlog.q];
         if[~:"--sql"in .z.x;d:d _ `s.k]
      ];
    ];
   if[(`s.k in !d) & |/d[`s.k] in " "\:.z.l 4;@[.s.init;(::);{.pykx.i.kxic.loadfailed,:enlist[x]!enlist y}[`s.k]];d:d _ `s.k];
   if[~0=#d;lf' (!d) @ & |/'(.:d) in " "\:.z.l 4;]; / only load enabled features
 }[]
 ];
