import pykx as kx
from IPython.display import display

from pathlib import Path

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
    save = False
    execute = True
    path = ''
    code_str= code
    locked = False
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
            elif instructions[0] == '--save':
                save = True
                path = instructions[1]
                if ((path[-2:] != '.q') and (path[-3:] != '.q_')):
                    raise NameError("File must be of type '.q' or '.q_'")
                if (path[-3:] == '.q_'):
                    locked = True
                instructions.pop(0)
                instructions.pop(0)
                continue
            elif instructions[0] == '--execute':
                if instructions[1] not in ['True', 'False']:
                    raise NameError("Execute must be 'True' or 'False'")
                execute = instructions[1] == 'True'
                instructions.pop(0)
                instructions.pop(0)
            elif instructions[0] == '':
                instructions.pop(0)
                continue
            else:
                raise kx.QError(
                    f'Received unknown argument "{instructions[0]}" in %%q magic command'
                )

    if execute or save:
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
                    with open(kx.config.pykx_dir/'pykx.q', 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if 'pykxld:' in line:
                                ld = _q("k)"+line[12:], skip_debug=True)
                                break
                else:
                    raise err
        else:
            _q = kx.q
        code = [kx.CharVector(x) for x in code.split('\n')][:-1]
        try:
            if execute:
                ret = _q(
                    '''{[ld;code;file]
                        res:1_ {x,enlist `err`res`trc!$[any x`err;
                        (1b;(::);(::));
                        .Q.trp[{(0b;(@) . ("q";x);(::))};y;{(1b;x;.Q.sbt y)}]]} over
                        enlist[enlist `err`res`trc!(0b;(::);(::))],
                        enlist[file],/:value(ld;code);
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
            if save:
                write_to_q_file(_q, locked, path, code_str)
                serv = " on q server" if issubclass(type(_q), kx.QConnection) else ""
                ex = " without cell logic being run. To run the cell remove '--execute False'."
                ex = '.' if execute else ex
                print(f"Cell contents saved to '{path}'{serv}{ex}")
        except Exception as e:
            if save:
                print(f"Cell contents not saved to '{path}' due to error during execution/saving.")
            raise e
        finally:
            if issubclass(type(_q), kx.QConnection):
                _q.close()


def write_to_q_file(_q, locked, path, code):
    if locked:
        output_file = Path(path[:-1])
        _q('0:', output_file, [kx.CharVector(code)])
        _q('\\_ ' + path[:-1])
        _q('hdel', output_file)
    else:
        output_file = Path(path)
        if issubclass(type(_q), kx.QConnection):
            _q('0:', output_file, [code[:-1].encode()])
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file.write_text(code)


def load_ipython_extension(ipython):
    ipython.register_magic_function(q, 'cell')

    def q_complete(self, event):
        return list(kx.q.reserved_words)
    ipython.set_hook('complete_command', q_complete, re_key='.*')
