"""
t1000_v2 base module.

This is the principal module of the t1000_v2 project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""
import gym
import numpy as np
from gym import spaces

CONFIG_SPEC_CONSTANTS = {
    "candlestick_width": {  # constants
        "day": 1,
        "hour": 0.04,
        "minute": 0.0006
    },
}


class Renderer:
    def __init__(self) -> None:
        pass


class Market:
    exchange_commission = 0.00075

    def __init__(self) -> None:
        self.df_features = {}

    def populate_dataframes(self):
        pass

    @staticmethod
    def get_data_from_api():
        pass


class Wallet:
    def __init__(self, net_worth: int, balance: int) -> None:
        self.net_worth = net_worth
        self.balance = balance
        self.cost = 0
        self.sales = 0
        self.shares_bought_per_asset = {}
        self.shares_sold_per_asset = {}
        self.shares_held_per_asset = {}


class ExchangeEnvironment(gym.Env, Wallet, Market):
    def __init__(self, net_worth: int, balance: int, assets: list) -> None:
        Wallet.__init__(self, net_worth, balance)
        Market.__init__(self)

        # 4,3 = (balance, cost, sales, net_worth) + (shares bought, shares sold, shares held foreach asset)
        observation_length = len(self.df_features[assets[0]].columns) + 4 + 3

        # action space = buy and sell for each asset, pĺus hold position
        action_space = 1 + len(assets) * 2

        # obs space = (num assets, indicator + (balance, cost, sales, net_worth) + (shares bought, shares sold, shares held foreach asset))
        observation_space = (len(assets),
                             observation_length)

        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([action_space, 1]),
            dtype=np.float16)

        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=observation_space,
            dtype=np.float16)

    def get_HODL_strategy(self) -> None:
        pass

    def compute_reward(self) -> None:
        pass

    def step(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass


class Brain:
    def __init__(self, learning_rate, algorithm='PPO', num_workers=2) -> None:
        self.algorithm = algorithm
        self.num_workers = num_workers
        self.learning_schedule = learning_rate

    @staticmethod
    def rollout():
        pass

    def train(self) -> None:
        pass

    def test(self) -> None:
        pass

    def trial_name_string(self) -> str:
        return 'trial-' + self.algorithm

    def get_instruments_from_checkpoint(self):
        pass


class TradingFloor(ExchangeEnvironment, Brain, Renderer):
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

    def __init__(self, assets: list, checkpoint_path='', algorithm='PPO', currency='USD', granularity='hour', datapoints=150, initial_balance=300, exchange_commission=0.00075, exchange='binance') -> None:
        ExchangeEnvironment.__init__(
            self, initial_balance, initial_balance, assets)
        Brain.__init__(self, 1e-4)
        Renderer.__init__(self)
        self.assets = assets
        self.algorithm = algorithm
        self.currency = currency
        self.granularity = granularity
        self.datapoints = datapoints
        self.config_spec = {}

    def generate_config_spec(self):
        pass

    def add_dataframes_to_config_spec(self):
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
