#%% 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class results():
    
    def __init__(self, values, title, row):
        
        if row == 0:
            
            tam = sum(v[1] for v in list(values.values()))
            
            print('-'*(tam+1))
            print('|', end='')
            print(f'{title:^{tam-1}}', end='')
            print('|')
            print('-'*(tam+1))

        for label, (result, size) in values.items():
            size -= (len(label)+2)
            print(f'| {label} {result:>{size-2}} ', end='')

        print('|')


class plots():

    def train_loss(self, df):
        sns.lineplot(df, x='epochs', y='loss')
        plt.title('APRENDIZAGEM POR ÉPOCA')
        plt.xlabel('ÉPOCA')
        plt.ylabel('LOSS')
        plt.show()


    def conf_matrix(self, y_true, y_pred):

        matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

        sns.heatmap(matrix, cmap='binary', annot=True, fmt='g')
        plt.title('MATRIZ DE CONFUSÃO')
        plt.xlabel('PREDICT')
        plt.ylabel('TRUE')
        plt.show()


    def image(self, img, y):
        
        plt.imshow(img.squeeze().permute(1,2,0))
        plt.title(y)
        plt.axis('off')
        plt.show()

        return self