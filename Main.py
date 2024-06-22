#%%
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from torch import nn, optim
from device import cuda
from data import data
from models import ModeloConvolucional
from rate import rate
from utils import plots
from torch import load
import torch
from random import randint



#/////////////////////#
#        DEVICE       #
#/////////////////////#

cuda = cuda()
device = cuda.device



#/////////////////////#
#         DATA        #
#/////////////////////#

data = data()
train = data.train_dataloader
test = data.test_dataloader



#/////////////////////#
#        MODEL        #
#/////////////////////#

model = ModeloConvolucional(x_len=2048, y_len=len(data.cat)).to(device)
optimizer = optim.SGD(model.parameters(), lr=.001)
loss = nn.CrossEntropyLoss()



#/////////////////////#
#      LOAD MODEL     #
#/////////////////////#

load_model = True

if load_model:

    try:
        model.load_state_dict(load('model.pth'))
        optimizer.load_state_dict(load('optimizer.pth'))
        print('loaded model')

    except:
        print('error load model')



#/////////////////////#
#        TRAIN        #
#/////////////////////#

rate = rate(model=model, opt=optimizer, loss=loss)
# rate.train(epochs=20, df_train=train, save_model=True)



#/////////////////////#
#  TRAIN - PLOT LOSS  #
#/////////////////////#

# plots().train_loss(rate.train_results)



#/////////////////////#
#         TEST        #
#/////////////////////#

rate.test(df_test=test)



#/////////////////////#
# TEST - PLOT PREDICT #
#/////////////////////#

predicted = rate.predicted_test
df_predicted = pd.DataFrame(predicted)
plots().conf_matrix(y_true=df_predicted['y_true'], y_pred=df_predicted['y_pred'])



#/////////////////////#
#  TEST - PLOT IMAGE  #
#/////////////////////#

x = torch.tensor(predicted['x'])
y_true = [data.cat[n] for n in predicted['y_true']]
y_pred = [data.cat[n] for n in predicted['y_pred']]



x[:, 0, :, :] = x[:, 0, :, :] * data.std[0] + data.mean[0]
x[:, 1, :, :] = x[:, 1, :, :] * data.std[1] + data.mean[1]
x[:, 2, :, :] = x[:, 2, :, :] * data.std[2] + data.mean[2]

for n in range(3):
    v = randint(0, len(x))
    plots().image(img=x[v], y=f'predict: {y_pred[v]} | true: {y_true[v]}')




#/////////////////////#
#       PREDICT       #
#/////////////////////#

#%%
data = data()
tensor_predict = data.predict_data().tensor.unsqueeze(0).to(device)


y_index = model.forward(tensor_predict).argmax()
y_predict = data.cat[y_index]

tensor_predict[:, 0, :, :] = tensor_predict[:, 0, :, :] * data.std[0] + data.mean[0]
tensor_predict[:, 1, :, :] = tensor_predict[:, 1, :, :] * data.std[1] + data.mean[1]
tensor_predict[:, 2, :, :] = tensor_predict[:, 2, :, :] * data.std[2] + data.mean[2]

plots().image(img=tensor_predict, y=y_predict)