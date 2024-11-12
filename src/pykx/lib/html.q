\d .pykx

// @private
// @desc
// Utility for generating the JSON data to be used for the rendering
// of a HTML version of a PyKX in-memory and splayed table
util.html.memsplay:{[c;t]
  n:count t;
  cls:{x!x}$[c[1]<ct:count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
  .j.j
    .pykx.util.html.extendcols[c[1];ct;]
    .pykx.util.html.extendrows[c[0];n;]
    .pykx.util.html.stringify
    .pykx.util.html.addindex[-1+n;]
    .pykx.util.html.filteridxs[cls;c 0;n;t]
  }

// @private
// @desc
// Filter in-memory/splayed data to retrieve the rows of the table
// which are required for display in HTML
util.html.filteridxs:{[cls;maxrows;n;tab]
  headcols:?[tab;enlist(<;`i;maxrows);0b;cls];
  tailcol:$[n>maxrows;?[tab;enlist(=;`i;(last;`i));0b;cls];()];
  headcols,tailcol
  }

// @private
// @desc
// Utility to allow the table to be artificially extended with an additional column
// ... if the console width is smaller than the maximum number of columns
util.html.extendcols:{[maxcols;origtabcols;tab]
  $[maxcols<origtabcols;
    {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
        (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};
    ::]tab
  }

// @private
// @desc
// Utility to allow the table to be artificially extended with an row
// ... if the console length is smaller than the maximum number of rows in the table
util.html.extendrows:{[rows;origtabrows;tab]
  $[rows<origtabrows;{(-2 _ x),(enlist {"..."} each flip 0#x),(-1 sublist x)};]tab
  } 

// @private
// @desc
// Addition of an artificial index column to the table to be used for the presentation of
// an appropriate index in the HTML representation
util.html.addindex:{[idxs;tab]
  i:til c:count tab;
  if[c;i[c-1]:idxs];
  flip[enlist[`pykxTableIndex]!enlist(i)],'tab
  }

// @private
// @desc
// Convert all cells to be displayed to a string representation which makes consumption of the
// data easier for a user, for example converting symbols/enums to strings to remove leading `
util.html.stringify:{[tab]
  flip{
    {$[11h~type x;
       string x;
       enlist[11h]~distinct type each x;
       sv[" "]each string x;
       enlist[20h]~distinct type each x;
       sv[" "]each string @[{value each x};x;x];
       .Q.s1 each x]
       }$[20h~type x;@[value;x;x];x]
    }each flip tab
  }

// @private
// @desc
// Add a <N> row x <M> columns printed statement following the HTML table
util.html.rowcols:{
  $[x[0]<n:count y;
    z,"\n<p>",{reverse "," sv 3 cut reverse string x}[n]," rows × ",
      {reverse "," sv 3 cut reverse string x}[count cols y]," columns</p>";
    z]
  }

// @private
// @desc
// Detect any invalid columns within a table
util.html.detectbadcols:{
  typ:.Q.qp x;
  fn:$[typ;{flip ct!count[ct:cols x]#()};0#];
  c:cols x;
  dup:where 1<count each group c;
  invalid:(c where not c=cols .Q.id fn x)except dup;
  `dup`invalid!(dup;invalid)
  }
