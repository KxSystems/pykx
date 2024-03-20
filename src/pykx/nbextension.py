import pykx as kx
from IPython.display import display

def q(instructions, code): # noqa
    ld = kx.SymbolAtom('.Q.pykxld')
    host = 'localhost'
    port = None
    username = ''
    password = ''
    timeout = 0.0
    large_messages = True
    tls = False
    unix = False
    no_ctx = False
    displayRet = False
    debug = False
    reconnection_attempts = -1
    if len(instructions)>0:

        instructions = instructions.split(' ')
        while True:
            if len(instructions) == 0:
                break
            elif instructions[0] in ('--host', '-h'):
                host = instructions[1]
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] in ('--port', '-p'):
                port = int(instructions[1])
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] in ('--user', '-u'):
                username = instructions[1]
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] == '--pass':
                password = instructions[1]
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] in ('--timeout', '-t'):
                timeout = float(instructions[1])
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] == '--nolarge':
                large_messages = False
                instructions.pop(0)
                continue
            elif instructions[0] == '--tls':
                tls = True
                instructions.pop(0)
                continue
            elif instructions[0] == '--unix':
                unix = True
                instructions.pop(0)
                continue
            elif instructions[0] == '--noctx':
                no_ctx = True
                instructions.pop(0)
                continue
            elif instructions[0] == '--display':
                displayRet = True
                instructions.pop(0)
                continue
            elif instructions[0] == '--debug':
                debug = True
                instructions.pop(0)
                continue
            elif instructions[0] == '--reconnection_attempts':
                reconnection_attempts = float(instructions[1])
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] == '':
                instructions.pop(0)
                continue
            else:
                raise kx.QError(
                    f'Received unknown argument "{instructions[0]}" in %%q magic command'
                )

    if port is not None:
        _q = kx.SyncQConnection(
            host,
            port,
            username=username,
            password=password,
            timeout=timeout,
            large_messages=large_messages,
            tls=tls,
            unix=unix,
            no_ctx=no_ctx,
            reconnection_attempts=reconnection_attempts
        )
        try:
            _q(ld, skip_debug=True)
        except kx.QError as err:
            if '.Q.pykxld' in str(err):
                # .Q.pykxld is not defined on the server so we pass it as inline code
                with open(kx.config.pykx_lib_dir/'q.k', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'pykxld:' in line:
                            ld = _q("k)"+line[7:-1], skip_debug=True)
                            break
            else:
                raise err
    else:
        _q = kx.q
    code = [kx.CharVector(x) for x in code.split('\n')][:-1]
    ret = _q(
        '''{[ld;code;file]
             res:1_ {x,enlist `err`res`trc!$[any x`err;
             (1b;(::);(::));
             .Q.trp[{(0b;(@) . ("q";x);(::))};y;{(1b;x;.Q.sbt y)}]]} over
            enlist[enlist `err`res`trc!(0b;(::);(::))],enlist[file],/:value(ld;code);
            select from res where not (::)~/:res}
       ''',
        ld,
        code,
        b'jupyter_cell.q', skip_debug=True
    )
    if not kx.licensed:
        ret = ret.py()
        for i in range(len(ret['res'])):
            if ret['err'][i]:
                if debug or kx.config.pykx_qdebug:
                    print(ret['trc'][i].decode())
                raise kx.QError(ret['res'][i].decode())
            else:
                display(ret['res'][i]) if displayRet else print(ret['res'][i])
    else:
        for i in range(len(ret)):
            r = _q('@', ret, i)
            if r['err']:
                if debug or kx.config.pykx_qdebug:
                    print(r['trc'])
                raise kx.QError(r['res'].py().decode())
            else:
                display(r['res']) if displayRet else print(r['res'])
    if issubclass(type(_q), kx.QConnection):
        _q.close()


def load_ipython_extension(ipython):
    ipython.register_magic_function(q, 'cell')
