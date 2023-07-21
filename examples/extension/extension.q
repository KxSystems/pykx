// Set the namespace to be same as the script name
\d .extension

test_data:([]100?`a`b`c;100?1f;100?10;100?0b)
test_function:{[data]
  analytic_keys :`max_x1`avg_x2`med_x3;
  analytic_calcs:(
    (max;`x1);
    (avg;`x2);
    (med;`x3));
  ?[data;
    ();
    k!k:enlist `x;
    analytic_keys!analytic_calcs
    ]
  }
