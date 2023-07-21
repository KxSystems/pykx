d:first each .Q.opt .z.x;

if[not `db in key d; -1 "Usage: q setuphdb.q -db hdb"; exit 1];

system "mkdir -p ",d[`db];
hdb:hsym `$first system raze "readlink -f ",d[`db];

mktrades:{[tickers; sz]
  dt:2015.01.01+sz?10;
  tm:sz?24:00:00.000;
  sym:sz?tickers;
  qty:10*1+sz?1000;
  px:90.0+(sz?2001)%100;
  t:([] dt; tm; sym; qty; px);
  t:`dt`tm xasc t;
  t:update px:6*px from t where sym=`goog;
  t:update px:2*px from t where sym=`ibm;
  t:update val:qty*px from t;
  t};
trades:mktrades[`aapl`goog`ibm;1000000];

{[x;y;z] hsym[`$(string[x],"/",string[y],"/trades/")] set .Q.en[x] delete dt from ?[z;enlist (in;`dt;y);0b;()]}[hdb;;trades] each asc exec distinct dt from trades;

exit 0;

/sample execution
/$ q setuphdb.q -db "hdb"

/Sample trade code taken from Q for Mortals 3 by Jeffry A. Borror
/https://code.kx.com/q4m3/1_Q_Shock_and_Awe/#117-example-trades-table
