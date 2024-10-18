import pandas as pd 
import random
from datetime import datetime,timedelta

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

    def fetch_data(
            self,
            symbol_list : Optional[List[str]] = None,
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

        return self
    
    def daily_hisotrical_df_(self):
        
        return self.raw_data_
    
    def daily_hisotrical_seq_(self,window_size :int = 6):
        
        #user input is in months , here it is converted into days since data in daily historical prices (20 business days in a month)
        seq_len = window_size * 20

        raw_data_ = self.raw_data_.sort_values(by = 'timestamp', ascending= True)
        groups = raw_data_.groupby('symbol')[['symbol', 'close']].apply(lambda x: x['close'].values.tolist())
        seq_dict_ = {key : [groups[key][i : i+ seq_len] for i in range(len(groups[key])) if len(groups[key][i : i+ seq_len]) == seq_len] for key in groups.keys()}

        return seq_dict_
    
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