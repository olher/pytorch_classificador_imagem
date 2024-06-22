#%%
from torch import cuda as c

class cuda():
    
    def __init__(self):
    
        self.is_available = c.is_available()
        self.device = 'cuda' if c.is_available() else 'cpu'
        self.device_name = c.get_device_name(c.current_device()) if c.is_available() else 'CPU'
