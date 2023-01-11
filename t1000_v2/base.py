"""
t1000_v2 base module.

This is the principal module of the t1000_v2 project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""

CONFIG_SPEC_CONSTANTS = {
    "candlestick_width": {  # constants
        "day": 1,
        "hour": 0.04,
        "minute": 0.0006
    },
}

class Market:
    def __init__(self) -> None:
        pass

class Wallet:
    def __init__(self, net_worth: int, balance: int) -> None:
        self.net_worth = net_worth
        self.balance = balance

class ExchangeEnvirnment(Market, Wallet):
    def __init__(self, net_worth: int, balance: int) -> None:
        super().__init__(net_worth, balance)
        pass

class Brain:
    def __init__(self, learning_rate, algorithm='PPO', num_workers = 2) -> None:
        self.algorithm = algorithm
        self.num_workers = num_workers
        self.learning_schedule = learning_rate

class Renderer:
    def __init__(self) -> None:
        pass

class TradingFloor:
    """A class containing all the logic and instruments to simulate a trading operation.

    documentation example: https://www.programiz.com/python-programming/docstrings

    ...

    Attributes
    ----------

    assets : list
        a list of assets available to trade.
    checkpoint_path : str
        a path to a saved model checkpoint, useful if you want to resume training from a already trained model (default '').
    algorithm : 

    Methods
    -------

    create_trial_name()
        Return the formatted trial name as a string
    
    """
    def __init__(self, assets: list, checkpoint_path = '', algorithm = 'PPO', currency = 'USD', granularity = 'hour', datapoints = 150, initial_balance = 300, exchange_commission = 0.00075, exchange = 'binance') -> None:
        self.assets = assets
        self.algorithm = algorithm
        self.currency = currency
        self.granularity = granularity
        self.datapoints = datapoints
        self.initial_balance = initial_balance
        self.exchange_commission = exchange_commission
        self.exchange = exchange
        self.dataframe = {}
        self.config_spec = {}

        if checkpoint_path:
            self.get_instruments_from_checkpoint(checkpoint_path)

        self.check_variables_integrity()
        self.populate_dataframes()

    def create_trial_name(self, *args) -> str:
        return '{}_{}_{}_{}'.format('-'.join(self.assets), self.currency, self.granularity, self.datapoints)

    def get_datasets(self, asset: str) -> dict:
        pass

    def get_instruments_from_checkpoint(self, checkpoint_path: str) -> dict:
        pass

    def check_variables_integrity(self) -> None:
        if type(self.assets) != list or len(self.assets) == 0:
            raise ValueError("Incorrect 'assets' value")
        if type(self.currency) != str:
            raise ValueError("Incorrect 'currency' value")
        if type(self.granularity) != str:
            raise ValueError("Incorrect 'granularity' value")
        if type(self.datapoints) != int or 1 > self.datapoints > 2000:
            raise ValueError("Incorrect 'datapoints' value")
        
    def populate_dataframes(self) -> None:
        for asset in self.assets:
            T-1000 = 'a'
            self.dataframe[asset] = {}
            self.dataframe[asset]['train'], self.dataframe[asset]['test'] = self.get_datasets(asset)

