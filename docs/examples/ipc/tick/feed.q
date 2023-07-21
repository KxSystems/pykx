.u.x: .z.x, count[.z.x]_ enlist ":5140";

h:neg hopen`:localhost:5140

// Define a .u.upd function just in case it needs to call itself if the above handle open fails
.u.upd: {[x;y]};

// Makes the IPC handle call to ticker plant or its ownself 
/ calls the .u.upd function on the tickerplant to publish the Trade/Quotes
/ A protected evaluation is used to ensure that when the ticker goes down, there will no longer be an error message  
.z.ts: {
 num:rand 1+til 4;
 symlist:num?`AAPL`JPM`GOOG`BRK`WPO`IBM;
 exlist:num?`NYSE`LON`CHI`HK;
 sz:num?(10*(1?til 100))0;
 px:num?(20+1?100f)0;
 trade_vals:(symlist;exlist;sz;px);
 upd_vals[`trade;trade_vals];			// update a trade every run
 }

upd_vals:{h(".u.upd";x;y)}

// Set feedhandler to publish messages at a 1 second interval is timer is not set at startup
if[not system"t";-1"\nTimer was not set, messages are now being set to send at 1 second intervals\n";system"t 1000"];
