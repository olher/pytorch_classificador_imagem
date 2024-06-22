from torch import nn

class MLP_Basicao(nn.Module):

    def __init__(self, x_len, y_len):

        super().__init__()

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
                      nn.Linear(x_len, 8),
                      nn.ReLU(),
                      nn.Linear(8, y_len),
                      nn.Softmax()
                      )
        
    def forward(self, x):
        
        output = self.flatten(x)
        output = self.layers(output)
        return output
    

class ModeloConvolucional(nn.Module):

    def __init__(self, x_len, y_len):

        super().__init__()

        self.convlayer = nn.Sequential(
                         nn.Conv2d(3, 16, kernel_size=(3,3), padding=1, stride=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),

                         nn.Conv2d(16, 32, kernel_size=(3,3), padding=1, stride=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2)
                         )

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
                      nn.Linear(x_len, 8),
                      nn.ReLU(),
                      nn.Linear(8, y_len),
                      nn.Softmax()
                      )
        
    def forward(self, x):
        
        output = self.convlayer(x)
        output = self.flatten(output)
        output = self.layers(output)
        return output