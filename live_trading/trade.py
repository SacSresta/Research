from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
import alpaca_trade_api as tradeapi
API_KEY = 'PKY7ODAXYMWTX4JK7DVR'
API_SECRET = 'yvQclRacekxCgwL0KX70SFkdbBAsVRvrBQnYaYGY'
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
async def trade_callback(t):
    print('trade', t)


async def quote_callback(q):
    print('quote', q)


# Initiate Class Instance
stream = Stream(API_KEY,
                API_SECRET,
                base_url=URL('https://paper-api.alpaca.markets'),
                data_feed='iex')  # <- replace to 'sip' if you have PRO subscription

# subscribing to event
stream.subscribe_trades(trade_callback, 'AAPL')
stream.subscribe_quotes(quote_callback, 'IBM')

stream.run()