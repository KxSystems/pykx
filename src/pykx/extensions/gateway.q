@[get;`.gw.ports;{.gw.ports:()!()}]

\d .tick

ports:([name:`$()]details:();connection:`int$())

init:{[config]
  if[0~count .gw.ports;
    '"'connections' not set in gateway configuration"
    ];
  if[99h<>type config;
    '"Supplied configuration must be a dictionary"
    ];
  addPort[.gw.ports];
  if[0=count config;:()];
  }

getPort:{[name]
  port:.tick.ports[name;`connection];
  if[()~port;
    '"Unable to retrieve specified port"
    ];
  port
  }

callPort:{[name;arguments]
  port:.tick.getPort[name];
  .[port;
    arguments;
    {[x;y;z]x . enlist y}[port;arguments]
    ]
  }

addPort:{[portInfo]
  {.[{`.tick.ports upsert (x;y;hopen hsym y)};
      (x;y);
      {-2"Failed to establish connection to port: ",x;}];
    }'[key portInfo;value portInfo]
  }

.pykx.pyexec"import pykx as kx\n",
            "class gateway:\n",
            "    def call_port(name, *args):\n",
            "        return kx.q('.tick.callPort', name, args)\n"
