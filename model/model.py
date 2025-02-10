from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix
import joblib

class Model:
    def __init__(self , x , y):
        self.x = x
        self.y = y
        print('the number of catigorical classes in label not palanced -> class impalance')
        print(self.y.value_counts())
        print("we must to handle the problem of class imbalance using (Class Weights) , Proper Evaluation Metrics")
        print("i use the RandomForestClassifier and LogisticRegression")
        print("loading ....")
    
    def train_Evaluation_RandomForest(self):
        x_train , x_eval , y_train  , y_eval = train_test_split(self.x , self.y , test_size= 0.2 , random_state= 0 )
        model_forest = RandomForestClassifier(class_weight="balanced" , random_state = 0 , n_estimators= 50 )
        model_forest.fit(x_train , y_train )
        y_pred_forest = model_forest.predict(x_eval )
        conv_matrix_forest = confusion_matrix(y_eval , y_pred_forest )
        eval_conv_matrix_forest = classification_report(y_eval , y_pred_forest )
        # save model 
        self.save_model(model_forest , "model_forest.pkl")
        return eval_conv_matrix_forest , conv_matrix_forest , model_forest
    def train_Evaluation_LogisticRegression(self):
        x_train , x_eval , y_train  , y_eval = train_test_split(self.x , self.y , test_size= 0.2 , random_state= 0 )
        model_logestic = LogisticRegression(class_weight="balanced")
        model_logestic.fit(x_train,y_train )
        y_pred_logestic = model_logestic.predict(x_eval)
        conv_matrix_logist = confusion_matrix(y_eval , y_pred_logestic)
        eval_conv_matrix_logist = classification_report(y_eval , y_pred_logestic )
        self.save_model(model_logestic , "model_logestic.pkl")
        return  eval_conv_matrix_logist , conv_matrix_logist , model_logestic
    
    def save_model(self , model , filename ):
        joblib.dump(model , filename)
        print(f"model saved as {filename}")

    @staticmethod
    def load_model(filename):
        return joblib.load(filename)
 



        

