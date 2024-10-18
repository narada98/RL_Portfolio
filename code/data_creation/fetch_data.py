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



class DataFetcher:
    def __init__(
            self,
            api_key : str,
            secret_key : str
            ):

        self.alpaca_client = StockHistoricalDataClient(
            api_key = api_key,
            secret_key = secret_key,
        )

    # @lru_cache(maxsize = 1)
    def fetch_data(
            self,
            symbol_list : Optional[tuple[str]] = None,
            end_date : datetime= datetime.now().date(),
            years_back : int= 2,
            ):
        
        if symbol_list is None:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            df = table[0]
            symbol_list = df['Symbol'].tolist()

        self.symbol_list = symbol_list
        self.start_date = end_date - timedelta(weeks = 52 * years_back)

        self.request_params_ = StockBarsRequest(
            symbol_or_symbols = symbol_list,
            timeframe = TimeFrame.Day,
            start = self.start_date,
            end = end_date,
            adjustment = 'all',
        )

        self.raw_data_= self.alpaca_client.get_stock_bars(
            request_params  = self.request_params_,
        ).df.reset_index()

        mask = (self.raw_data_.symbol.value_counts() == self.raw_data_.symbol.value_counts().max())

        if len(mask[~mask].index) == 0:
            print("No null values were found")
        else:
            print(f"found null values at : {mask[~mask].index}")
        
        self.raw_data_ = self.raw_data_[self.raw_data_.symbol.isin(mask[mask].index)]

        self.raw_data_.timestamp = self.raw_data_.timestamp.apply(lambda x: x.date())

        print(f"number of unique symbols in raw data : {self.raw_data_.symbol.nunique()}")
        return self
    
    def create_seq(self, close_arr, return_arr, size_, seq_len):
        return(np.array([np.concatenate((close_arr[i:i+seq_len] , np.array([return_arr[i+seq_len-1]]))) for i in range(size_-(seq_len-1))]))
    
    def create_dataset_seq_(self,window_size :int = 3):
        
        #user input is in months , here it is converted into days since data in daily historical prices (20 business days in a month)
        seq_len = window_size * 20
        
        dataset_df_ = self.dataset_df_.sort_values(by = 'timestamp', ascending= True)
        groups = dataset_df_.groupby('symbol')[['symbol', 'close','return_']].apply(lambda x: self.create_seq(x['close'].to_numpy(), x['return_'].to_numpy(), len(x), seq_len))
    
        return groups.to_dict()
    
    def create_datasets(self, time_horizon : int = 20, window_size : int = 3):
        self.time_horizon = time_horizon
        dataset_df_ = self.raw_data_.copy()
        dataset_df_['target_'] = dataset_df_.groupby('symbol')['close'].shift(-time_horizon)
        dataset_df_ = dataset_df_.dropna()
        dataset_df_['return_'] = dataset_df_.apply(lambda x: ((x['target_']/x['close']) - 1), axis = 1)
        
        self.dataset_df_ = dataset_df_
        self.dataset_seq = self.create_dataset_seq_(window_size)
        
        return self

    def get_dataset_(self):
        return self.dataset_df_

    def get_dataset_seq_(self):
        return self.dataset_seq
    
    def daily_hisotrical_df_(self):
        
        return self.raw_data_
       
    def save_df_(self, save_location):
        save_location = os.path.join(save_location, datetime.now().strftime("%Y_%m_%d"))

        if not os.path.exists(save_location):
            os.makedirs(save_location)
            
        save_location = os.path.join(save_location, f'historical_raw_price_df_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.csv')
        self.raw_data_.to_csv(save_location, index = False)

    def save_dataset_(self, save_location):
        save_location = os.path.join(save_location, datetime.now().strftime("%Y_%m_%d"))

        if not os.path.exists(save_location):
            os.makedirs(save_location)
            
        save_location = os.path.join(save_location, f'dataset_{self.time_horizon}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.csv')
        self.dataset_df_.to_csv(save_location, index = False)

    def save_seq_(self, save_location):
        save_location = os.path.join(save_location, datetime.now().strftime("%Y_%m_%d"))

        if not os.path.exists(save_location):
            os.makedirs(save_location)
         
        save_location = os.path.join(save_location, f'historical_price_seq_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.pkl')
        with open(save_location, 'wb') as f:
            pickle.dump(self.dataset_seq, f)

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