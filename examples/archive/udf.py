from textwrap import dedent

import pykx as kx


def setup():
    kx.q(dedent('''
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
        '''))


def demo():
    print('Initial trades table:', kx.q('trades'), sep='\n')

    @kx.Function
    def convert_currency(source_currency, target_currency, price):
        from forex_python.converter import CurrencyRates
        rates = CurrencyRates().get_rates(str(source_currency))
        return price.np() * rates[str(target_currency)]

    # Rename px -> pxUSD, val -> valUSD
    kx.q('trades:`dt`tm`sym`qty`pxUSD`valUSD xcol trades')

    kx.q('{[convert_currency] trades:update '
         'pxCAD:convert_currency[`USD;`CAD;pxUSD],'
         'pxEUR:convert_currency[`USD;`EUR;pxUSD],'
         'pxGBP:convert_currency[`USD;`GBP;pxUSD]'
         ' from trades}', convert_currency)

    kx.q('trades:update valCAD:pxCAD*qty, valEUR:pxEUR*qty, valGBP:pxGBP*qty from trades')
    print('\n\nAfter currency update:', kx.q('trades'), sep='\n')

    # The `@kx.Function` decorator is optional, as Python objects (including functions) are
    # automatically converted to K objects when passed into q.
    def sort(x):
        return sorted(
            x.py(),
            key=lambda x: x[-3:] if x.startswith('px') or x.startswith('val') else x,
            reverse=True
        )

    # Rearrange the column order using our sort function defined above
    kx.q('{[f] trades:raze[f cols trades] xcols trades}', sort)
    print('\n\nAfter sorting columns:', kx.q('trades'), sep='\n')


if __name__ == '__main__':
    setup()
    demo()
