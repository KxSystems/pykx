.tick.init:{[config]
  if[99h<>type config;
    '"Supplied configuration must be a dictionary"
    ];
  if[not `database in key config;
    '"Supplied configuration must contain a 'database' location"
    ];
  @[{system"l ",x;-1"Successfully loaded database: ",x;};string config`database;{-1"Database not loaded"}]; 
  }
