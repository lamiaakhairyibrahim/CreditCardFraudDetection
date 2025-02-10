import seaborn as sns
import matplotlib.pyplot as mp
import pandas as pd


class Visualization:
    def __init__(self , x , y):
        self.x = x
        self.y = y 

    def visual(self):
        data = pd.concat([self.x , self.y] , axis = 1)
        #plotting the heatmap for correlation
        mp.figure(figsize=(20,20))
        ax = sns.heatmap(data.corr(), annot=True)
        mp.show()
            