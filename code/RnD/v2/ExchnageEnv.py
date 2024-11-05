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
            episode_duration,
            train_test_split,
            symbol_universe,
            feature_set,
            device
            ):
        
        self.device = device
        self.holding_period = 20 * holding_period #Days
        self.feature_set = feature_set
        
        with open(data_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        
        self.features = (np.array([self.data_dict[symbol]['features'] for symbol in symbol_universe])).transpose(1,2,0,3)
        self.returns = (np.array([self.data_dict[symbol]['returns'] for symbol in symbol_universe])).transpose(1,0)
        #shape from (symbols, time_steps, seq_len, features) -> (time_steps ,seq_len, symbols, features)
        # self.split_ = int(self.features.shape[0] * train_test_split)
        
        # self.t_ = 0
        self.episode_duration = episode_duration * self.holding_period
        self.split_ = int(self.features.shape[0]- 1 - self.episode_duration)

    def reset(self, mode, transaction_cost = 1e-7):

        if mode == 'train':
            
            # self.t_ = 0
            # self.end_t_ = self.split_
            
            #limiting an episode to 1 year
            self.t_ = random.randint(0, self.split_ - self.episode_duration - 1)
            self.end_t_ = self.t_ + self.episode_duration

        elif mode == 'test':
            # self.t_ = self.split_
            # self.end_t_ = self.features.shape[0]-1

            self.t_ = random.randint(self.split_, self.features.shape[0]- 1 - self.episode_duration)
            self.end_t_ = self.t_ + self.episode_duration

        self.current_allocations = np.zeros((self.features.shape[-2]))
        self.transaction_cost = transaction_cost

        # return self
    
    def get_return(self, allocations):
        
        # t_ : window idx
        # current_price (at t_) * return * allcoations
        close_price = self.features[self.t_, :, : , self.feature_set.index('close')]
        return_price = self.returns[self.t_, :]
        return (close_price * return_price * allocations.detach().cpu().numpy()).sum()
    
        # return (torch.tensor(self.data[self.t_, : , -2]) * torch.tensor(self.data[self.t_, : , -1]) * allocations).sum()
    
    def get_random_state(self):

        random_t = random.randrange(len(self.features)-1)
        return torch.from_numpy(self.features[random_t , : , :, :]).to(self.device).to(torch.float32)

    def get_state(self):
        
        if self.is_end():
            return None
        
        return torch.from_numpy(self.features[self.t_ , : , :, :]).to(self.device).to(torch.float32)

    def is_end(self):
        return self.t_ > self.end_t_

    def step(self, allocations):

        return_ = self.get_return(allocations)
        self.t_ = self.t_ + self.holding_period
        state_ = self.get_state()
        is_end = self.is_end()

        return state_, return_, is_end, self.transaction_cost
       