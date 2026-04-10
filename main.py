import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import faithful_backtest_eth as fth_eth


CONFIG = [
           {'name':'BULL',
           'start_date':'2025-04-10',
           'end_date':'2025-10-15',
           'assets': 10_000
           },
          {'name':'BEAR',
           'start_date':'2025-10-15',
           'end_date':None,
           'assets' : 10_000 # In USDC
           },
           {'name':'BEAR',
           'start_date':'2025-10-15',
           'end_date':None,
           'assets' : 100_000
           },
           {'name':'BULL',
           'start_date':'2025-04-10',
           'end_date':'2025-10-15',
           'assets': 100_000
           },
           {'name':'BEAR',
           'start_date':'2025-10-15',
           'end_date':None,
           'assets' : 1_000_000
           },
           {'name':'BULL',
           'start_date':'2025-04-10',
           'end_date':'2025-10-15',
           'assets': 1_000_000
           }
          ]

for setup in CONFIG:
    fth_eth.main(setup['name'], setup['assets'] ,setup['start_date'], setup['end_date'])

