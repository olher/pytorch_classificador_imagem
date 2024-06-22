from torch import save as s


class save():

    def __init__(self, model, optimizer):
        
        s(model.state_dict(), 'model.pth')
        s(optimizer.state_dict(), 'optimizer.pth')