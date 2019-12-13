#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def balancedDataSetCreater(df_numeric_label):
    '''
    input:
    1. df_numeric_label : (unbalanced) data set with numeric label , type: DataFrame
    
    output:
    1. bal_df : balanced data set, type: DataFrame
    '''
    #convert index and name of columns of DataFrame
    
    df_numeric_label = pd.DataFrame(np.array(df_numeric_label))
    
    df_class1 = pd.DataFrame({'f1' : [] ,'f2' : [], 'f3' : [] ,'f4' : [],'f5' : [] ,'f6' : [],'f7' : [] ,'f8' : [],'f9' : [] ,'f10' : []
                              ,'f11' : [] ,'f12' : [],'f13' : [] ,'f14' : [],'f15' : [] ,'f16' : [],'f17' : [] ,'f18' : [],'f19' : [] ,'f20' : []
                              ,'f21' : [] ,'f22' : [],'f23' : [] ,'f24' : [],'f25' : [] ,'f26' : [],'f27' : [] ,'f28' : [],'f29' : [] ,'f30' : []
                              ,'f31' : [] ,'f32' : [],'f33' : [] ,'f34' : [],'f35' : [] ,'f36' : [],'f37' : [] ,'f38' : [],'f39' : [] ,'f40' : []
                              ,'f41' : [] ,'f42' : [],'f43' : [] ,'f44' : [],'f45' : [] ,'f46' : [],'f47' : [] ,'f48' : [],'f49' : [] ,'f50' : []    
                              ,'f51' : [] ,'f52' : [],'f53' : [] ,'f54' : [],'f55' : [] ,'f56' : [],'f57' : [] ,'f58' : [],'f59' : [] ,'f60' : []
                              ,'f61' : [] ,'f62' : [],'f63' : [] ,'f64' : [],'f65' : [] ,'f66' : [],'f67' : [] ,'f68' : [],'f69' : [] ,'f70' : []
                              ,'f71' : [] ,'f72' : [],'f73' : [] ,'f74' : [],'f75' : [] ,'f76' : [],'f77' : [] ,'f78' : [],'f79' : [] ,'f80' : []
                              ,'f81' : [] ,'f82' : [],'f83' : [] ,'f84' : [],'f85' : [] ,'f86' : [],'f87' : [] ,'f88' : [],'f89' : [] ,'f90' : []    
                              ,'f91' : [] ,'f92' : [],'f93' : [] ,'f94' : [],'f95' : [] ,'f96' : [],'f97' : []
                              ,'label' : []})
    df_class2 = pd.DataFrame({'f1' : [] ,'f2' : [], 'f3' : [] ,'f4' : [],'f5' : [] ,'f6' : [],'f7' : [] ,'f8' : [],'f9' : [] ,'f10' : []
                              ,'f11' : [] ,'f12' : [],'f13' : [] ,'f14' : [],'f15' : [] ,'f16' : [],'f17' : [] ,'f18' : [],'f19' : [] ,'f20' : []
                              ,'f21' : [] ,'f22' : [],'f23' : [] ,'f24' : [],'f25' : [] ,'f26' : [],'f27' : [] ,'f28' : [],'f29' : [] ,'f30' : []
                              ,'f31' : [] ,'f32' : [],'f33' : [] ,'f34' : [],'f35' : [] ,'f36' : [],'f37' : [] ,'f38' : [],'f39' : [] ,'f40' : []
                              ,'f41' : [] ,'f42' : [],'f43' : [] ,'f44' : [],'f45' : [] ,'f46' : [],'f47' : [] ,'f48' : [],'f49' : [] ,'f50' : []    
                              ,'f51' : [] ,'f52' : [],'f53' : [] ,'f54' : [],'f55' : [] ,'f56' : [],'f57' : [] ,'f58' : [],'f59' : [] ,'f60' : []
                              ,'f61' : [] ,'f62' : [],'f63' : [] ,'f64' : [],'f65' : [] ,'f66' : [],'f67' : [] ,'f68' : [],'f69' : [] ,'f70' : []
                              ,'f71' : [] ,'f72' : [],'f73' : [] ,'f74' : [],'f75' : [] ,'f76' : [],'f77' : [] ,'f78' : [],'f79' : [] ,'f80' : []
                              ,'f81' : [] ,'f82' : [],'f83' : [] ,'f84' : [],'f85' : [] ,'f86' : [],'f87' : [] ,'f88' : [],'f89' : [] ,'f90' : []    
                              ,'f91' : [] ,'f92' : [],'f93' : [] ,'f94' : [],'f95' : [] ,'f96' : [],'f97' : []
                              ,'label' : []})
    #conversion column names to integers
    df_class1 = pd.DataFrame(np.array(df_class1))
    df_class2 = pd.DataFrame(np.array(df_class2))
    
    #96 means 'label'
    for index in range(0,len(df_numeric_label)):
        if(df_numeric_label.loc[index, 97] == 0):
            #class1
            df_class1 = df_class1.append(df_numeric_label.loc[index])
        elif(df_numeric_label.loc[index, 97] == 1):
            #class2
            df_class2 = df_class2.append(df_numeric_label.loc[index]) 
    
    #convert index in order
    df_class1 = pd.DataFrame(np.array(df_class1))
    df_class2 = pd.DataFrame(np.array(df_class2))    
    
    ##Seperate X and y
    #97 means 'label'
    #Seperate feature data points 
    X_class1 = df_class1.drop( 97 ,axis = 1) 
    X_class2 = df_class2.drop( 97 ,axis = 1) 
    #Seperate class labels
    y_class1=pd.DataFrame(df_class1.loc[:,97])
    y_class2=pd.DataFrame(df_class2.loc[:,97])

    #Down sample data points of class1,2,3 to the same number of data points of class4
    X_class2_down , temp_X_test, y_class2_down , temp_y_test = train_test_split(X_class2, y_class2, train_size = len(df_class1))

    #combine feature data points and class label
    df_class1 = pd.concat([X_class1, y_class1] , axis = 1, sort = False)
    df_class2_down = pd.concat([X_class2_down, y_class2_down] , axis = 1, sort = False)

    #comebine each data point from class1 , 2, 3, 4
    df_data_set_bal = pd.concat([df_class1, df_class2_down])

    return df_data_set_bal



####################################Main Training Part####################33
df = pd.read_csv('Autism.csv')

#assign pre space for data frame that has numeric labels
df_numeric_label = df

#Convert categorical label names to numeric labels
for index in range(0,len(df_numeric_label)):
    if(df_numeric_label.loc[index,'Column21'] == 'YES'):
        df_numeric_label.loc[index,'Column21'] = 0 #class1
    elif(df_numeric_label.loc[index,'Column21'] == 'NO'):
        df_numeric_label.loc[index,'Column21'] = 1 #class2
        
#replace all ? value to NaN in DataFrame
df[df=='?'] = np.nan

#Drop all data samples that contain missing values(i.e., NaN value in any feature)
droped_df = pd.DataFrame(np.array(df.dropna()))

#Build numeric features only DataFrame
temp_numeric_df1 = droped_df.truncate(before=0, after=10 , axis = "columns")
temp_numeric_df2 = droped_df.truncate(before=17, after=17, axis = "columns")

num_feature_only_df = pd.concat([temp_numeric_df1, temp_numeric_df2] , axis = 1, sort = False)

#Build categorical features only DataFrame
temp_numeric_df3 = droped_df.truncate(before=11, after=16 , axis = "columns")
temp_numeric_df4 = droped_df.truncate(before=18, after=19, axis = "columns")

catego_feature_only_df = pd.concat([temp_numeric_df3, temp_numeric_df4] , axis = 1, sort = False)

#Convert categorical data to numeric data using one hot encoder method
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(catego_feature_only_df)
encoded_df = pd.DataFrame(encoder.transform(catego_feature_only_df).toarray())

#combine two DataFrames
numeric_encoded_X = pd.concat([num_feature_only_df, encoded_df] , axis = 1, sort = False)
#get label data
y=pd.DataFrame(droped_df.loc[:,20])#20 means 'label' column

#combine two DataFrames one more time to make a whole dataset(i.e.,(x,y) pair)
encoded_numeric_df = pd.concat([numeric_encoded_X, y] , axis = 1, sort = False)

#split training data set and test data set 
#80% of D for D'', 20% of D for D_test with rule of thumb
X_tr_80 , X_te_20, y_tr_80, y_te_20 = train_test_split(numeric_encoded_X, y, train_size=8/10)
X_tr_80_arr = np.array(X_tr_80)
y_tr_80_arr = np.array(y_tr_80)

#Split D_test to D_val_te and D_fin_te
#50% of D_test for each data set
X_val_te_50 , X_fin_te_50 , y_val_te_50 , y_fin_te_50 = train_test_split(X_te_20, y_te_20 , train_size = 1/2)
X_val_te_50_arr = np.array(X_val_te_50) 
X_fin_te_50_arr = np.array(X_fin_te_50) 
y_val_te_50_arr = np.array(y_val_te_50) 
y_fin_te_50_arr = np.array(y_fin_te_50)


#Thanks to code from https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
#How to convert object type to int type
X_tr_80_arr = X_tr_80_arr.astype('int')
y_tr_80_arr = y_tr_80_arr.astype('int')
X_val_te_50_arr = X_val_te_50_arr.astype('int')
y_val_te_50_arr = y_val_te_50_arr.astype('int')
X_fin_te_50_arr = X_fin_te_50_arr.astype('int')
y_fin_te_50_arr = y_fin_te_50_arr.astype('int')

#Create balanced Data
D_dot_dot = pd.DataFrame(np.array(pd.concat([X_tr_80, y_tr_80] , axis = 1, sort = False)))
D_dot_dot.columns = ['f1' ,'f2' , 'f3' ,'f4' ,'f5' ,'f6' ,'f7' ,'f8' ,'f9' ,'f10' 
                              ,'f11' ,'f12' ,'f13' ,'f14' ,'f15' ,'f16' ,'f17' ,'f18' ,'f19' ,'f20' 
                              ,'f21' ,'f22' ,'f23' ,'f24' ,'f25' ,'f26' ,'f27' ,'f28' ,'f29' ,'f30' 
                              ,'f31' ,'f32' ,'f33' ,'f34' ,'f35' ,'f36' ,'f37' ,'f38' ,'f39' ,'f40' 
                              ,'f41' ,'f42' ,'f43' ,'f44' ,'f45' ,'f46' ,'f47' ,'f48' ,'f49' ,'f50'     
                              ,'f51' ,'f52' ,'f53' ,'f54' ,'f55' ,'f56' ,'f57' ,'f58' ,'f59' ,'f60'
                              ,'f61' ,'f62' ,'f63' ,'f64' ,'f65' ,'f66' ,'f67' ,'f68' ,'f69' ,'f70' 
                              ,'f71' ,'f72' ,'f73' ,'f74' ,'f75' ,'f76' ,'f77' ,'f78' ,'f79' ,'f80' 
                              ,'f81' ,'f82' ,'f83' ,'f84' ,'f85' ,'f86' ,'f87' ,'f88' ,'f89' ,'f90'   
                              ,'f91' ,'f92' ,'f93' ,'f94' ,'f95' ,'f96' ,'f97' 
                              ,'label']


bal_df = balancedDataSetCreater(D_dot_dot)
X_bal = bal_df.truncate(before=0, after=96 , axis = "columns")
y_bal = pd.DataFrame(bal_df.loc[:,97])#97 means 'label' column
#use whole data set as a balanced data set 
X_bal_arr = np.array(X_bal)
y_bal_arr = np.array(y_bal)

#convert object type to int type
X_bal_arr = X_bal_arr.astype('int')
y_bal_arr = y_bal_arr.astype('int')

#save data files
#1. whole data set
encoded_numeric_df.to_csv(r'data_autism\data_autism.csv')

#2. D''
D_dot_dot.to_csv(r'data_autism\data_train_autism.csv')

#.3. D''_bal
bal_df.to_csv(r'data_autism\data_train_bal_autism.csv')

#4. D_val_test
D_val_test = pd.DataFrame(np.array(pd.concat([pd.DataFrame(X_val_te_50_arr), pd.DataFrame(y_val_te_50_arr)] , axis = 1, sort = False)))
D_val_test.to_csv(r'data_autism\data_val_test_autism.csv')               

#5. D_fin_test
D_fin_test = pd.DataFrame(np.array(pd.concat([pd.DataFrame(X_fin_te_50_arr), pd.DataFrame(y_fin_te_50_arr)] , axis = 1, sort = False)))
D_fin_test.to_csv(r'data_autism\data_fin_test_autism.csv')               


# In[ ]:





# In[ ]:




