import pandas as pd 
import numpy as np
import random
from datetime import datetime,timedelta
import pickle
import os
from  functools import lru_cache

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from datetime import datetime
from alpaca.data.timeframe import TimeFrame

import plotly.graph_objects as go
from typing import List, Optional
import ta

from ta import add_all_ta_features

class DataFetcher:
    def __init__(
            self,
            api_key : str,
            secret_key : str,
            ):

        self.alpaca_client = StockHistoricalDataClient(
            api_key = api_key,
            secret_key = secret_key,
        )

    # @lru_cache(maxsize = 1)
    def fetch_data(
            self,
            save_location : str,
            features :List,
            symbol_list : Optional[tuple[str]] = None,
            end_date : datetime= datetime.now().date(),
            years_back : int= 2,
            save_ : bool = True,
            ):
        
        self.features = features
        print(f"Fetching data")
        #fetching SnP 500 symbols if needed
        if symbol_list is None:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            df = table[0]
            symbol_list = df['Symbol'].tolist()
            print(f"---fetched symbols from SnP 500 index")

        self.symbol_list = symbol_list
        self.start_date = end_date - timedelta(weeks = 52 * years_back)

        #fetching data from Alpaca API
        self.request_params_ = StockBarsRequest(
            symbol_or_symbols = symbol_list,
            timeframe = TimeFrame.Day,
            start = self.start_date,
            end = end_date,
            adjustment = 'all',
        )

        print(f"---fetching historical daily data from Alpaca | from : {self.start_date} to: {end_date}")
        self.raw_data_= self.alpaca_client.get_stock_bars(
            request_params  = self.request_params_,
        ).df.reset_index()

        print(f"---Data fetching completed")
        print()

        #Null values analysis and handling
        print(f"Null values analysis")
        mask = (self.raw_data_.symbol.value_counts() == self.raw_data_.symbol.value_counts().max())

        if len(mask[~mask].index) == 0:
            print("---No null values were found")
        else:
            print(f"---found null values at : {mask[~mask].index}")
        
        self.raw_data_ = self.raw_data_[self.raw_data_.symbol.isin(mask[mask].index)]
        self.raw_data_.timestamp = self.raw_data_.timestamp.apply(lambda x: x.date())
        
        print(f"---number of unique symbols in raw data : {self.raw_data_.symbol.nunique()}")
        print()
        #creating all 3 types of datasets
        self.__create_datasets()

        
        #Saving datasets if required
        if save_:
            print(f"Saving data")
            self.__save_data_(save_location)
            
        return self
    
    def __create_seq_(self, data, window_size):

        seq_dict = {}
        seq_dict['features'] = np.array([data[self.features].values[i:i+window_size] for i in range (len(data)-window_size+1)])
        seq_dict['returns'] = np.array([data.return_.values[i+window_size-1] for i in range (len(data)-window_size+1)])

        return seq_dict
    

    def __create_dataset_seq_(self, window_size :int = 3):
        #user input is in months , here it is converted into days since data in daily historical prices (20 business days in a month)
        seq_len = window_size * 20
        
        dataset_df_ = self.dataset_df_.sort_values(by = 'timestamp', ascending= True)
        data_sqs_dict = dataset_df_.groupby(['symbol']).apply(lambda x : self.__create_seq_(data = x, window_size = seq_len))

        return data_sqs_dict.to_dict()
    

    def __create_datasets(self, time_horizon : int = 20, window_size : int = 3):
        self.time_horizon = time_horizon
        dataset_df_ = self.raw_data_.copy()
        dataset_df_['target_'] = dataset_df_.groupby('symbol')['close'].shift(-time_horizon)
        dataset_df_ = dataset_df_.dropna()
        dataset_df_['return_'] = dataset_df_.apply(lambda x: ((x['target_']/x['close']) - 1), axis = 1)
        
        dataset_df_ = dataset_df_.groupby('symbol', as_index=False).apply(lambda x : add_all_ta_features(x, open="open", high="high", low="low", close="close", volume="volume", fillna=True)).reset_index(drop = True)

        self.dataset_df_ = dataset_df_
        self.dataset_seq = self.__create_dataset_seq_(window_size)
        return self

    def get_dataset_(self):
        return self.dataset_df_

    def get_dataset_seq_(self):
        return self.dataset_seq
    
    def daily_hisotrical_df_(self):
        return self.raw_data_
    
    def __save_file(self, data, save_location, fname, format):
        if format == 'csv':
            save_location = os.path.join(save_location, f'{fname}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.csv') 
            data.to_csv(save_location, index = False)
            print(f"---{fname} saved @ {save_location}")

        if format == 'pkl':
            save_location = os.path.join(save_location, f'{fname}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.pkl')
            with open(save_location, 'wb') as f:
                pickle.dump(data, f)
            print(f"---{fname} saved @ {save_location}")

    def __save_data_(self, save_location):
        save_location = os.path.join(save_location, datetime.now().strftime("%Y_%m_%d"))

        if not os.path.exists(save_location):
            os.makedirs(save_location)

        self.__save_file(self.raw_data_, save_location, "dataset_raw", 'csv')
        self.__save_file(self.dataset_df_, save_location, "dataset_processed", 'csv')
        self.__save_file(self.dataset_seq, save_location, "dataset_sqs", 'pkl')
        self.__save_file(self.features, save_location, "feature_set", 'pkl')

    def visualize_sample(
            self,
            symbol_ : Optional[str] = None
            ):
        
        if symbol_ is None:
            symbol_ = random.choice(self.raw_data_.symbol.unique().tolist())

        visu_df_ = self.raw_data_.loc[self.raw_data_.symbol == symbol_].copy()

        fig = go.Figure()
        x_vals = visu_df_.timestamp
        ohlc_inst = go.Candlestick(x=x_vals, open=visu_df_.open, high=visu_df_.high, low=visu_df_.low, close=visu_df_.close, name= symbol_)
        fig.add_trace( ohlc_inst)

        head_msg = symbol_
        fig.update_layout(title=head_msg, height=600, width=1200,
                    yaxis=dict(gridcolor='lightgray'),showlegend=True,  plot_bgcolor='white',xaxis_rangeslider_visible=True )

        fig.show()