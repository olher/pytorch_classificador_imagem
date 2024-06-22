from device import cuda
from checkpoint import save
from utils import results
import torch
import pandas as pd

class rate():
    
    def __init__(self, model, opt, loss):
        
        print(cuda().device_name)
        self.device = cuda().device
        self.model = model
        self.opt = opt
        self.loss = loss


    def train(self, epochs, df_train, save_model=True):

        self.model.train()
        self.train_results = {'epochs':[], 'loss':[]}

        for epoch in range(epochs+1):
            
            train_loss = 0.0
            corrects = 0
            values = 0
            
            for x, y in df_train:
            
                self.opt.zero_grad()
                x = x.to(self.device)
                
                target = torch.zeros(size=[10])
                target[y] = 1
                target = target.to(self.device)

                output = self.model.forward(x).to(self.device)
                lossfunc = self.loss(output, y)
                lossfunc.backward()
                self.opt.step()
                train_loss += lossfunc.item()
                corrects += int((y == output.argmax(dim=1)).float().sum().item())
                values += len(output)

            self.train_results['epochs'].append(epoch)
            self.train_results['loss'].append(train_loss/len(df_train))
                
            if epoch % 10 == 0:

                if save_model: save(model=self.model, optimizer=self.opt)
                
                load = f'{epoch/epochs*100:.2f}'
                loss = f'{train_loss/len(df_train):.2f}'
                acc = f'{corrects/values*100:.2f}'

                results(values={'Epoch':[epoch, 15]
                               ,'Load%':[load, 15]
                               ,'Loss':[loss, 15]
                               ,'Acc':[acc, 15]}
                        ,title='TRAIN RESULTS'
                        ,row=epoch)
        
        self.train_results = pd.DataFrame(self.train_results)
        
        return self
                

    def test(self, df_test):
        
        self.model.eval()
        test_loss = 0.0
        
        predicted_test = {'x':[], 'y_true':[], 'y_pred':[]}

        with torch.no_grad():
        
            corrects = 0
            values = 0
        
            for x, y in df_test:
                
                x = x.to(self.device)
                target = torch.zeros(size=[10])
                target[y] = 1
                target = target.to(self.device)

                output = self.model.forward(x).to(self.device)
                test_loss += self.loss(output, y).item()
                
                # SAVE PREDICT MODEL
                for n in range(0, len(y)):
                    predicted_test['x'].append(x.tolist()[n])
                    predicted_test['y_true'].append(y.tolist()[n])
                    predicted_test['y_pred'].append(output.argmax(dim=1).tolist()[n])              

                # ACCURACY 
                values += len(output)
                corrects += int((y == output.argmax(dim=1)).float().sum().item())
        
        results(values={'Epoch':[0, 15]
                       ,'Load%':[100, 15]
                       ,'Loss':[round(test_loss/len(df_test),4), 15]
                       ,'Acc':[round(corrects/values,4)*100, 15]}
                               
                       ,title='TEST RESULTS'
                       ,row=0)
        
        self.predicted_test = predicted_test
        
    

    def predict(self, df_predict):


        self.model.eval()
        
        predicted = {'x':[], 'pred':[]}

        with torch.no_grad():
        
            corrects = 0
            values = 0
        
            for x, _ in df_predict:

                x = x.to(self.device)
                output = self.model.forward(x).to(self.device)
                
                # SAVE PREDICT MODEL
                predicted['x'].append(x.tolist())
                predicted['pred'].append((output == torch.max(output)).float().tolist())              

                # ACCURACY 
                values += len(output)
                corrects += int(torch.sum((torch.sum((((output == torch.max(output)).float() == y).float()), dim=1) == y.shape[1]).float(), dim=0).tolist())
        
                
        self.predicted = predicted