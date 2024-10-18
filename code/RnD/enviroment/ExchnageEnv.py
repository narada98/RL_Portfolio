import numpy as np 
import pickle
import random
import torch

''' 
to do : consider transaction cost when calculating return
'''


class MarketEnvironment:
    
    def __init__(
            self,
            data_path,
            holding_period,
            train_test_split,
            symbol_universe,
            device
            ):
        
        self.device = device
        self.holding_period = 20 * holding_period #Days
        
        with open(data_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        
        self.data = (np.array([self.data_dict[symbol] for symbol in symbol_universe])).transpose(1,0,2)
        self.split_ = int(self.data.shape[0] * train_test_split)
        self.t_ = 0

    def reset(self, mode, transaction_cost = 1e-7):

        if mode == 'train':
            self.t_ = random.randint(0, self.split_)
            self.end_t_ = self.split_

        elif mode == 'test':
            self.t_ = self.split_
            self.end_t_ = len(self.data)-1

        self.current_allocations = np.zeros((self.data.shape[1]))
        self.transaction_cost = transaction_cost

        # return self
    
    def get_return(self, allocations):
        
        # t_ : window idx
        # current_price (at t_) * return * allcoations
        return (self.data[self.t_, : , -2] * self.data[self.t_, : , -1] * allocations.detach().cpu().numpy()).sum()

    def get_random_state(self):

        random_t = random.randrange(len(self.data)-1)
        return torch.from_numpy(self.data[random_t , : , :-1]).to(self.device).to(torch.float32)

    def get_state(self):
        
        if self.is_end():
            return None
        
        return torch.from_numpy(self.data[self.t_ , : , :-1]).to(self.device).to(torch.float32)

    def is_end(self):
        return self.t_ > self.end_t_

    def step(self, allocations):

        return_ = self.get_return(allocations)
        self.t_ = self.t_ + self.holding_period
        state_ = self.get_state()
        is_end = self.is_end()

        return state_, return_, is_end, self.transaction_cost
       