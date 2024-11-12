config:.Q.def[`port`points!5030 1] .Q.opt .z.x
h:neg hopen config`port
points:config`points

// Define a .u.upd function just in case it needs to call itself if the above handle open fails
.u.upd: {[x;y]};

// Makes the IPC handle call to ticker plant or its ownself 
/ calls the .u.upd function on the tickerplant to publish the Trade/Quotes
/ A protected evaluation is used to ensure that when the ticker goes down, there will no longer be an error message  
.z.ts: {
 symlist:`AAPL`JPM`GOOG`BRK`WPO`IBM;
 exlist:`NYSE`LON`CHI`HK;
 sz:10*points?til 100;px:20+points?100f;
 ask:points?100f;asksz:10*points?til 100;
 bd:ask-points?ask;bdsz:asksz-points?asksz;
 trade_vals:(points?symlist;points?exlist;sz;px);
 quote_vals:(points?symlist;points?exlist;bd;ask;bdsz;asksz);
 upd_vals[`trade;trade_vals];	           // update a trade every run
 if[0=(1?3)0;upd_vals[`quote;quote_vals]]; // if statement just for show to reduce number of quotes
 }

upd_vals:{h(".u.upd";x;y)}

// Set feedhandler to publish messages at a 1 second interval is timer is not set at startup
if[not system"t";
  -1"\nTimer was not set, messages are now being set to send at 1 second intervals\n";
  system"t 1000"
  ];
