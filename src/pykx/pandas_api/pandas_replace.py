from . import api_return


def _init(_q):
    global q
    q = _q


class PandasReplace:

    @api_return
    def replace(self, to_replace, value):
        return q('''
                {[t;s;r]
                gt:$[-11h~type t;get;(::)] t;
                cs:cols $[99h~type gt;value;(::)]gt;
                map:([] c:cs; cT:type each value ?[t;();();cs!cs]);
                map:update s:count[map]#enlist s,sT:type s,r:count[map]#enlist r,rT:type r from map;
                map:select from map where (cT=0) or neg[sT]=cT;
                map:update sOp:?[(sT>=0) or cT=0;count[map]#(~/:);count[map]#(=)] from map;
                map:update rI:{[t;c;s;sOp] where sOp[s;t c]}[0!gt]'[c;s;sOp] from map;
                map:delete from map where 0=count each rI;
                map:update atF:?[(0=cT) or neg[cT]=rT;count[map]#(@[;;:;]);count[map]#({1_ @[(::),x;1+y;:;z]})] from map;
                map:update r:(count each rI)#'enlist each r from map;
                ![t;();0b;map[`c]!exec {[atF;c;rI;r](atF[;rI;r];c)}'[atF;c;rI;r] from map]
                }
                 ''', self, to_replace, value) # noqa: E501
