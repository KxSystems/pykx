import argparse
import os
os.environ['PYKX_BETA_FEATURES'] = "true"

import pykx as kx

parser=argparse.ArgumentParser()

parser.add_argument(
    "--datapoints",
    help="Number of datapoints to be used when populating "
         "each day of the database. Default = 10000",
    default=10000,
    type=int)

parser.add_argument(
    "--date",
    help="The day prior to today's date which will be used "
         "as the first date in the Database. Default = 1 -> Yesterday",
    default=1,
    type=int)

parser.add_argument(
    "--days",
    help="The number of days prior to '--date' which will be generated. Default = 1",
    default=1,
    type=int)

parser.add_argument(
    "--name",
    help="The name to be given to the database. Default = 'db'",
    default='db',
    type=str)

# Define Schemas
trade = kx.schema.builder({
    'time': kx.TimespanAtom, 'sym': kx.SymbolAtom,
    'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    'px': kx.FloatAtom})

quote = kx.schema.builder({
    'time': kx.TimespanAtom, 'sym': kx.SymbolAtom,
    'exchange': kx.SymbolAtom, 'bid': kx.FloatAtom,
    'ask': kx.FloatAtom, 'bidsz': kx.LongAtom,
    'asksz': kx.LongAtom})

variables = vars(parser.parse_args())
datapoints = variables['datapoints']
date = variables['date']
days = variables['days']
db_name = variables['name']

symlist = ['AAPL', 'JPM', 'GOOG', 'BRK', 'WPO', 'IBM']
exlist = ['NYSE', 'LON', 'CHI', 'HK']

num_trades = datapoints
num_quotes = round(datapoints/3)

if days <= 0:
    raise ValueError('--days supplied as a value <=0')
if date < 0:
    raise ValueError('--date supplied as a value < 0')

db = kx.DB(path=db_name)

while days>0:
    # Generate random trade data
    trade_data = [
        kx.q.asc(kx.random.random(num_trades, kx.q('1D00:00:00.000'))),
        kx.random.random(num_trades, symlist),
        kx.random.random(num_trades, exlist),
        10*kx.random.random(num_trades, 100),
        20+kx.random.random(num_trades, 100.0)]

    # Generate random quote data
    ask = kx.random.random(num_quotes, 100.0)
    asksz = 10*kx.random.random(num_quotes, 100)
    bd = ask - kx.random.random(num_quotes, ask)
    bdsz = asksz - kx.random.random(num_quotes, asksz)
    quote_data = [
        kx.q.asc(kx.random.random(num_quotes, kx.q('1D'))),
        kx.random.random(num_quotes, symlist),
        kx.random.random(num_quotes, exlist),
        bd,
        ask,
        bdsz,
        asksz]

    # Generate trade and quote database partitions
    db.create(trade.insert(trade_data, inplace=False), 'trade', kx.DateAtom('today') - days)
    db.create(quote.insert(quote_data, inplace=False), 'quote', kx.DateAtom('today') - days)

    # Decrement the number of days that need to be supplied
    days -= 1
