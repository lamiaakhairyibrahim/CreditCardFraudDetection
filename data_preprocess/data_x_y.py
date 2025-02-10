import pandas as pd
from src.data_preprocess.preprocess_data import Preprocess
from src.data_preprocess.visualization import Visualization
from src.model.model import Model

class data:
    def __init__(self , path_of_data):
        self.path_of_data  = path_of_data
        self.data_featur_label()

    def data_featur_label(self):
        data = pd.read_csv( self.path_of_data)
        process = Preprocess(data=data)
        x , y = process.process()
        visual = Visualization(x , y)
        visual.visual()
        model = Model(x=x , y=y)
        eval_conv_matrix_forest, conv_matrix_forest , model_forest = model.train_Evaluation_RandomForest()
        print("conf_matrix for randam forest : \n", conv_matrix_forest)
        print("conf_matrix for randam forest : \n", eval_conv_matrix_forest)
        print(f"the model saved in {model_forest}")
       
        eval_conv_matrix_logistic, conv_matrix_logistic , model_logistic = model.train_Evaluation_LogisticRegression()
        print("conf_matrix for  logistic : \n",  conv_matrix_logistic)
        print("conf_matrix for  logistic: \n", eval_conv_matrix_logistic)
        print(f"the model saved in {model_logistic}")