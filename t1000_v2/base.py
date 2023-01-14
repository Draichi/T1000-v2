"""
t1000_v2 base module.

This is the principal module of the t1000_v2 project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""
import os

import gym
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
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
    """An abstraction of a crypto market

    Attributes
    ----------
        exchange (str): the exchange name
        assets (list[str]): a list containing assets to trade"""
    load_dotenv()
    CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')
    exchange_commission = 0.00075

    def __init__(self, exchange: str, assets: list[str], granularity: str, currency: str, datapoints: int) -> None:
        self.df_features = {}
        self.raw_dataframe = {}
        self.train_dataframe = {}
        self.test_dataframe = {}
        self.exchange = exchange
        self.granularity = granularity
        self.currency = currency
        self.datapoints = datapoints
        self.get_dataframes(assets)

    def get_dataframes(self, assets: list[str]):

        if not self.CRYPTOCOMPARE_API_KEY:
            raise ImportError('CRYPTOCOMPARE_API_KEY not found on .env')

        for asset in assets:
            # fetch api
            self.raw_dataframe[asset] = self.fetch_api(asset)
            # add indicators
            # split into train and test dfs
            # save to csv files
            # create key inside self.df with asset's name
            self.train_dataframe[asset] = {}
            self.test_dataframe[asset] = {}
            # read this csv files to avoid ray indexing issue

        print(self.raw_dataframe)

    def fetch_api(self, asset: str) -> pd.DataFrame:
        """Fetch the CryptoCompare API and return historical prices for a given asset

            Parameters:
                asset (str): The asset name. For a full asset list check: https://min-api.cryptocompare.com/documentation?key=Other&cat=allCoinsWithContentEndpoint

            Returns:
                raw_dataframe (pandas.Dataframe): The API 'Data' key response converted to a pandas Dataframe
        """

        print('> Fetching {} dataset'.format(asset))

        path_to_raw_dataframe = 'data/raw_dataframe_{}.csv'.format(
            asset)

        if os.path.exists(path_to_raw_dataframe):
            raw_dataframe = pd.read_csv(path_to_raw_dataframe)

            return raw_dataframe

        else:
            headers = {'User-Agent': 'Mozilla/5.0',
                       'authorization': 'Apikey {}'.format(self.CRYPTOCOMPARE_API_KEY)}
            url = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit={}&e={}'.format(
                self.granularity, asset, self.currency, self.datapoints, self.exchange)
            response = requests.get(url, headers=headers)
            json_response = response.json()
            status = json_response['Response']

            if status == "Error":
                print('Error fetching {} dataframe'.format(asset))
                raise AssertionError(json_response['Message'])

            result = json_response['Data']
            raw_dataframe = pd.DataFrame(result)

            to_datetime_arg = raw_dataframe['time']
            raw_dataframe.drop(['time', 'conversionType',
                               'conversionSymbol'], axis=1, inplace=True)

            raw_dataframe['Date'] = pd.to_datetime(
                arg=to_datetime_arg, utc=True, unit='s')

            return raw_dataframe


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
    def __init__(self, net_worth: int, balance: int, assets: list, currency: str, exchange: str, granularity: str, datapoints: int) -> None:
        Wallet.__init__(self, net_worth, balance)
        Market.__init__(self, exchange=exchange, assets=assets, currency=currency,
                        datapoints=datapoints, granularity=granularity)

        # 4,3 = (balance, cost, sales, net_worth) + (shares bought, shares sold, shares held foreach asset)
        # observation_length = len(self.df_features[assets[0]].columns) + 4 + 3

        # action space = buy and sell for each asset, pĺus hold position
        # action_space = 1 + len(assets) * 2

        # obs space = (num assets, indicator + (balance, cost, sales, net_worth) + (shares bought, shares sold, shares held foreach asset))
        # observation_space = (len(assets),
        #                      observation_length)

        # self.action_space = spaces.Box(
        #     low=np.array([0, 0]),
        #     high=np.array([action_space, 1]),
        #     dtype=np.float16)

        # self.observation_space = spaces.Box(
        #     low=-np.finfo(np.float32).max,
        #     high=np.finfo(np.float32).max,
        #     shape=observation_space,
        #     dtype=np.float16)

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

    def __init__(self, assets: list[str], checkpoint_path='', algorithm='PPO', currency='USD', granularity='hour', datapoints=150, initial_balance=300, exchange_commission=0.00075, exchange='CCCAGG') -> None:
        ExchangeEnvironment.__init__(
            self, net_worth=initial_balance, assets=assets, currency=currency, exchange=exchange, granularity=granularity, datapoints=datapoints, balance=initial_balance)
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


T1000 = TradingFloor(['BTC'])
