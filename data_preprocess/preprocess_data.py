import pandas as pd
from haversine import haversine, Unit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class Preprocess:
    def __init__(self , data):
        self.data = data

    
    def process(self):
        print("Training set for Credit Card Transactions")
        print("The dataset contains : ")
        data_info = """
                    nocolum         colum 
                       0          index - Unique Identifier for each row
                       1          trans_date_trans_time - Transaction DateTime
                       2          cc_num - Credit Card Number of Customer
                       3          merchant - Merchant Name
                       4          category - Category of Merchant
                       5          amt - Amount of Transaction
                       6          first - First Name of Credit Card Holder
                       7          last - Last Name of Credit Card Holder
                       8          gender - Gender of Credit Card Holder
                       9          street - Street Address of Credit Card Holder
                       10         city - City of Credit Card Holder
                       11         state - State of Credit Card Holder
                       12         zip - Zip of Credit Card Holder
                       13         lat - Latitude Location of Credit Card Holder
                       14         long - Longitude Location of Credit Card Holder
                       15         city_pop - Credit Card Holder's City Population
                       16         job - Job of Credit Card Holder
                       17         dob - Date of Birth of Credit Card Holder
                       18         trans_num - Transaction Number
                       17         unix_time - UNIX Time of transaction
                       18         merch_lat - Latitude Location of Merchant
                       19         merch_long - Longitude Location of Merchant
                       20         is_fraud - Fraud Flag <--- Target Class"""
        print(data_info)
        print("shape of the data : " , self.data.shape)
        print("data info : " , self.data.info())
        print("sampel from the data : \n" , self.data.head())
        print("if data contain missing value or not ")
        print(self.data.isnull().sum())
        print("this column [trans_date_trans_time] contain data and time of the process ,we can split this column to year , month , day , hour , minute")
        self.data["datetime"] = pd.to_datetime(self.data["trans_date_trans_time"] , format= '%Y-%m-%d %H:%M:%S')
        self.data = self.data.drop(["trans_date_trans_time"] , axis= 1)
        self.data["year"]   = self.data["datetime"].dt.year
        self.data["month"]  = self.data["datetime"].dt.month
        self.data["day"]    = self.data["datetime"].dt.day_of_week
        self.data["Hour"]   = self.data["datetime"].dt.hour
        self.data["minute"] = self.data["datetime"].dt.minute
        print("you can see sample of data after editing : ")
        print(self.data.head())
        print("this column ['dob'] contain Date of Birth of Credit Card Holder , we can calculate the age ")
        self.data["dob"] = pd.to_datetime(self.data["dob"] , format= '%Y-%m-%d')
        self.data["age"] = (self.data["datetime"] - self.data["dob"]).dt.days // 365
        print("you can see sample from age column ")
        print(self.data["age"].head())
        print("we can calculate the time between the process and the previous process for the same customer")
        print("to calculate this time must sorted data by columns : 'cc_num' , 'datetime' ")
        print("this time by mitutes")
        self.data = self.data.sort_values(by= ['cc_num' , 'datetime'])
        self.data['time_diff'] = ((self.data.groupby('cc_num')['datetime'].diff()).dt.total_seconds())/60
        print("the shape of data after calculate time :")
        print(self.data[['cc_num' , 'datetime' ,'time_diff']])
        print(self.data.info())
        print(self.data.head())
        print("After sorting and calculating the [tim_diff] we can see two new problems")
        print("the problem is \n 1 - Missing value in column [time_diff], due to all first processes for the customer always being[nan].\n 2 - unsorted index")
        print("the number of missing value must equal the number of categorical in [cc_num]")
        print("number of categorical data in [cc_num] : " , self.data["cc_num"].nunique())
        print("the number of missing value in [diff_time] : " , self.data["time_diff"].isna().sum())
        print("let's solve this problem , by fill missing value by (0)")
        self.data["time_diff"] = self.data['time_diff'].fillna(0)
        print("the number of missing value after fill missing data by (0) : " , self.data["time_diff"].isna().sum())
        print("this problem is Done ^_^")
        print("let's solve the problem of unsorted index ")
        self.data = self.data.reset_index(drop=True)
        print("you can see the data after solving the two problems : ")
        print(self.data.head())
        print("you cane see the [gender] column contain two catigorical data : " , self.data["gender"].nunique())
        print("we can solve this problem by 'one-hot-encoding' , and thin concat the our data and the data after one hot encoding ")
        data_gender = pd.get_dummies(self.data["gender"]).astype("int").reset_index(drop=True)
        self.data = pd.concat([self.data, data_gender],axis=1)
        print("you can see the data info after all this changes ")
        print(self.data.info())
        print("when we see on categorical columns , let's calculates the categorical , to decide if we can convert it to numerical or not")
        print("the number catigorical in [merchant] : " , self.data['merchant'].nunique())
        print("the number catigorical in [category] : " , self.data['category'].nunique())
        print("the number catigorical in [street] : " , self.data['street'].nunique())
        print("the number catigorical in [city] : " , self.data['city'].nunique())
        print("the number catigorical in [state] : " , self.data['state'].nunique())
        print("the number catigorical in [job] : " , self.data['job'].nunique())
        print("we can convert by label encoding ")
        le = LabelEncoder()
        self.data['merchant'] = le.fit_transform(self.data['merchant'])
        self.data["category"] = le.fit_transform(self.data["category"])
        self.data['street'] = le.fit_transform(self.data['street'])
        self.data['city'] = le.fit_transform(self.data['city'])
        self.data['state'] = le.fit_transform(self.data['state'])
        self.data["job"] = le.fit_transform(self.data["job"])
        print("data info : ")
        print(self.data.info())
        print("the distance between catumer and merchant must good for us")
        print("let's calculat the distance ")
        print("loading .....")
        self.data['distance'] = self.data.apply(lambda row:haversine((row['lat'], row['long']), 
                                                (row['merch_lat'], row['merch_long']) , unit=Unit.MILES),axis=1)
        print("you can see the distance column : ")
        print(self.data['distance'])
        print("remove the columns that do not benefit for us")
        self.data = self.data.drop(["datetime" , 'merch_long' ,
                                     'long' , 'lat',  'merch_lat' , 
                                     "trans_num" ,"zip" , 
                                     'Unnamed: 0' , 'first' , 'last' , 'gender' , 'dob' , "unix_time"],axis = 1)
        print("the data after remove unusful columns")
        print(self.data.info())
        print("now split the data to feature and labels ")
        x_data = self.data.drop(['is_fraud'] , axis = 1)
        y = self.data["is_fraud"]
        print("featuer data is : ")
        print(x_data.head())
        print("the label data: ")
        print(y.head())
        print("must normalize the featuer to be fear when the model fit ")
        scaler = StandardScaler()
        # Fit and transform the data
        standardized_data = scaler.fit_transform(x_data)
        x  = pd.DataFrame(standardized_data  ,  columns = x_data.columns )
        print("Standardized Data (Z-score Normalization):")
        print(x.head())
        print(x.info())
        print(y)
        return ( x , y )




    