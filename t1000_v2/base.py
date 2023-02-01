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
from ray import data as ray_data
from ta import add_all_ta_features

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
        - exchange (str): the exchange name.
        - assets (list[str]): a list containing assets to trade. Example: `["BTC", "XMR"]`
        - granularity (str): how to fetch the data. Options: `hour`, `day`."""
    load_dotenv()
    CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')
    exchange_commission = 0.00075

    def __init__(self, exchange: str, assets: list[str], granularity: str, currency: str, datapoints: int) -> None:
        self.df_features = {}
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.train_dataframe = {}
        self.test_dataframe = {}
        self.first_prices: dict[str, float] = {}
        self.exchange = exchange
        self.granularity = granularity
        self.currency = currency
        self.datapoints = datapoints

        self.__get_dataframes(assets)

    def __get_dataframes(self, assets: list[str]) -> None:
        """Get the dataframe for each asset

            Parameters:
                assets (list[str]): The list of assets to get the dataframe. Example: `["BTC", "ETH"]`

            Returns:
                None"""

        if not self.CRYPTOCOMPARE_API_KEY:
            raise ImportError('CRYPTOCOMPARE_API_KEY not found on .env')

        for asset in assets:
            print('> Fetching {} dataframe'.format(asset))

            self.dataframes[asset] = self.__fetch_api(asset)

            self.dataframes[asset] = self.__add_indicators(asset)

            self.train_dataframe[asset], self.test_dataframe[asset] = self.__split_dataframes(
                asset)

            self.df_features[asset] = self.__populate_df_features(
                asset, 'train')

            print('> Caching {} dataframe'.format(asset))

            self.__save_complete_dataframe_to_csv(asset)

    def __fetch_api(self, asset: str) -> pd.DataFrame:
        """Fetch the CryptoCompare API and return historical prices for a given asset

            Parameters:
                asset (str): The asset name. For a full asset list check: https://min-api.cryptocompare.com/documentation?key=Other&cat=allCoinsWithContentEndpoint

            Returns:
                raw_dataframe (pandas.Dataframe): The API 'Data' key response converted to a pandas Dataframe
        """

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

            pandas_dataframe = pd.DataFrame(result)
            to_datetime_arg = pandas_dataframe['time']
            pandas_dataframe.drop(['time', 'conversionType',
                                   'conversionSymbol'], axis=1, inplace=True)

            pandas_dataframe['Date'] = pd.to_datetime(
                arg=to_datetime_arg, utc=True, unit='s')

            raw_dataframe = ray_data.from_pandas(pandas_dataframe)

            return pandas_dataframe

    def __add_indicators(self, asset: str) -> pd.DataFrame:
        """Get the `self.raw_dataframe` dataframe and adds the market indicators for the given timeserie.

            Returns:
                raw_dataframe (pandas.DataFrame): A new dataframe based on `self.raw_dataframe` but with the indicators on it"""
        dataframe_with_indicators = {}
        dataframe_with_indicators[asset] = add_all_ta_features(
            self.dataframes[asset], open="open", high="high", low="low", close="close", volume="volumeto")
        return dataframe_with_indicators[asset]

    def __split_dataframes(self, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a dataframe for a selected asset into train_dataframe and test_dataframe

            Parameters:
                asset (str): asset name

            Returns:
                train_dataframe (ray.data.Dataset): A dataset containing the data to train
                test_dataframe (ray.data.Dataset): A dataset containing the data to test"""
        ray_dataframe = ray_data.from_pandas(self.dataframes[asset])
        train, test = ray_dataframe.train_test_split(test_size=0.25)
        return train.to_pandas(), test.to_pandas()

    def __save_complete_dataframe_to_csv(self, asset: str) -> None:
        """Save the dataframe with prices and indicators to a csv file to speed up future runnings

            Parameters:
                asset (str): The asset name

            Returns:
                None"""
        path_to_dataframe = 't1000_v2/data/complete_{}_dataframe.csv'.format(
            asset)
        self.dataframes[asset].to_csv(path_to_dataframe)

    def __populate_df_features(self, asset: str, mode: str) -> None:
        if mode == 'train':
            return self.train_dataframe[asset].loc[:,
                                                   self.train_dataframe[asset].columns != 'Date']

        elif mode == 'test':
            return self.test_dataframe[asset].loc[:,
                                                  self.test_dataframe[asset].columns != 'Date']


class Wallet:
    def __init__(self, net_worth: float, balance: float, assets: list[str], exchange_commission: float) -> None:
        self.exchange_commission = exchange_commission
        self._net_worth = net_worth
        self.initial_balance = balance
        self.balance = balance
        self.cost: float = 0
        self.sales: float = 0
        self.shares_bought_per_asset: dict[str, float] = {}
        self.shares_sold_per_asset: dict[str, float] = {}
        self.shares_held_per_asset: dict[str, float] = {}
        self.initial_bought = {}
        self.trades: dict[str, list] = {}
        self.assets: list[str] = assets

    @property
    def net_worth(self) -> float:
        """Returs the foo var"""
        return self._net_worth

    @net_worth.setter
    def net_worth(self, value: float):
        self._net_worth = value

    def __reset_net_worth(self):
        self.net_worth = self.initial_balance

    def __reset_all_shares(self):
        for asset in self.assets:
            self.shares_bought_per_asset[asset] = 0
            self.shares_held_per_asset[asset] = 0
            self.shares_sold_per_asset[asset] = 0

    def reset_shares_bought_and_sold(self):
        for asset in self.assets:
            self.shares_bought_per_asset[asset] = 0
            self.shares_sold_per_asset[asset] = 0

    def reset_balance(self):
        self.balance = self.initial_balance
        self.reset_cost_and_sales()
        self.__reset_net_worth()
        self.__reset_all_shares()

    def reset_trades(self):
        for asset in self.assets:
            self.trades[asset] = []

    def reset_cost_and_sales(self):
        self.cost = 0
        self.sales = 0

    def __amount_can_be_spent(self, amount: float) -> bool:
        """Calculate if has balance to spend"""
        if self.balance >= self.balance * amount * (1 + self.exchange_commission):
            return True
        else:
            return False

    def open_trade_ticket(self, action_type: float, action_strength: float):
        """Check if it's possible to buy or sell"""
        is_bought = False
        is_sold = False
        is_possible_to_buy = self.__amount_can_be_spent(amount=action_strength)
        for index, asset in enumerate(self.assets * 2):
            if action_type < index / 2 + 1 and is_possible_to_buy and not is_bought:
                is_bought = True
            elif action_type < index + 1 and not is_sold:
                is_sold = True


class ExchangeEnvironment(Wallet, Market, gym.Env):
    def __init__(self, net_worth: int, balance: int, assets: list, currency: str,
                 exchange: str, granularity: str, datapoints: int, exchange_commission: float) -> None:

        Wallet.__init__(self, net_worth=net_worth,
                        balance=balance, assets=assets, exchange_commission=exchange_commission)

        Market.__init__(self, exchange=exchange, assets=assets, currency=currency,
                        datapoints=datapoints, granularity=granularity)

        self.current_step: int = 0
        self.current_price: float = 0
        # 4,3 = (balance, cost, sales, net_worth) + (shares bought, shares sold, shares held foreach asset)
        self.observation_length = len(
            self.df_features[assets[0]].columns) + 4 + 3

        # action space = buy and sell for each asset, pĺus hold position
        action_space = 1 + len(assets) * 2

        # obs space = (num assets, indicator + (balance, cost, sales, net_worth) + (shares bought, shares sold, shares held foreach asset))
        observation_space = (len(assets),
                             self.observation_length)

        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([action_space, 1]),
            dtype=np.float16)

        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=observation_space,
            dtype=np.float16)

    def __compute_initial_bought(self):
        """spread the initial account balance through all assets"""
        for asset in self.assets:
            self.initial_bought[asset] = 1/len(self.assets) * \
                self.initial_balance / self.first_prices[asset]

    def __get_next_observation(self):
        empty_observation_array = np.empty((0, self.observation_length), int)
        for asset in self.assets:
            current_market_state = np.array(
                self.df_features[asset].values[self.current_step])
            current_state = np.array([np.append(current_market_state, [
                self.balance,
                self.cost,
                self.sales,
                self.net_worth,
                self.shares_bought_per_asset[asset],
                self.shares_sold_per_asset[asset],
                self.shares_held_per_asset[asset]
            ])])
            next_observation = np.append(
                empty_observation_array, current_state, axis=0)

            print('> next observation:', next_observation)

            return next_observation

    def __take_action(self, action: list[float]):
        """Take an action within the environment"""
        action_type = action[0]
        action_strength = action[1]
        # TODO
        # - compute_current_price()

        """bounds of action_space doesn't seem to work, so this line is necessary to not overflow actions"""
        if 0 < action_strength <= 1 and action_type > 0:
            # TODO
            # - reset_shares_bought_n_sold()
            # - reset_cost_n_sales()
            # - buy_or_sell(action_type=action_type, amount=amount)
            # - compute_trade()
            self.reset_shares_bought_and_sold()
            self.reset_cost_and_sales()

    def get_HODL_strategy(self) -> None:
        pass

    def compute_reward(self) -> None:
        pass

    def step(self, action: list[float]):
        """Execute one time step within the environment"""
        # TODO
        self.__take_action(action)
        self.current_step += 1

    def reset(self):
        """Reset the ExchangeEnvironment to it's initial state"""
        self.current_step = 0
        self.reset_balance()
        self.reset_trades()
        # TODO
        # - get_first_prices

        return self.__get_next_observation()

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
            self, net_worth=initial_balance, assets=assets, currency=currency, exchange=exchange, granularity=granularity, datapoints=datapoints, balance=initial_balance, exchange_commission=exchange_commission)
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
T1000.reset()
