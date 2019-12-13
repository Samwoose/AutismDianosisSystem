#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import random
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


#Load Preprocessed Datasets
#D''
D_dot_dot = pd.read_csv(r'data_autism\data_autism.csv')
X_tr_80 = D_dot_dot.drop('Unnamed: 0' , axis = 1)
X_tr_80 = X_tr_80.drop('20.1', axis = 1) #20.1 means label
y_tr_80 = pd.DataFrame(D_dot_dot.loc[:,'20.1'])
X_tr_80_arr = np.array(X_tr_80)
y_tr_80_arr = np.array(y_tr_80)
#convert object type to int type
X_tr_80_arr = X_tr_80_arr.astype('int')
y_tr_80_arr = y_tr_80_arr.astype('int')

#D''_bal
D_dot_dot_bal = pd.read_csv(r'data_autism\data_train_bal_autism.csv')
X_tr_bal = D_dot_dot_bal.drop('Unnamed: 0' , axis = 1)
X_tr_bal = X_tr_bal.drop('97', axis = 1) #97 means label
y_tr_bal = pd.DataFrame( D_dot_dot_bal.loc[:,'97'])
X_bal_arr = np.array(X_tr_bal)
y_bal_arr = np.array(y_tr_bal)
#convert object type to int type
X_bal_arr = X_bal_arr.astype('int')
y_bal_arr = y_bal_arr.astype('int')

#D_val_test
D_val_test = pd.read_csv(r'data_autism\data_val_test_autism.csv')
X_val_te_50 = D_val_test.drop('Unnamed: 0' , axis = 1)
X_val_te_50 = X_val_te_50.drop('97', axis = 1) #97 means label
y_val_te_50 = pd.DataFrame( D_val_test.loc[:,'97'])
X_val_te_50_arr = np.array(X_val_te_50)
y_val_te_50_arr = np.array(y_val_te_50)
#convert object type to int type
X_val_te_50_arr = X_val_te_50_arr.astype('int')
y_val_te_50_arr = y_val_te_50_arr.astype('int')

#D_fin_test
D_fin_test = pd.read_csv(r'data_autism\data_fin_test_autism.csv')
X_fin_te_50 = D_fin_test.drop('Unnamed: 0' , axis = 1)
X_fin_te_50 = X_fin_te_50.drop('97', axis = 1) #97 means label
y_fin_te_50 = pd.DataFrame( D_fin_test.loc[:,'97'])
X_fin_te_50_arr = np.array(X_fin_te_50)
y_fin_te_50_arr = np.array(y_fin_te_50)
#convert object type to int type
X_fin_te_50_arr = X_fin_te_50_arr.astype('int')
y_fin_te_50_arr = y_fin_te_50_arr.astype('int')



##load trained models
#Random Forest models
loaded_model_RF_Unbal_autism = pickle.load(open('fin_RF_clf_unbal_model_autism.sav','rb'))
loaded_model_RF_bal_autism = pickle.load(open('fin_RF_clf_bal_model_autism.sav','rb'))

#L2 models
loaded_model_L2_Unbal_autism = pickle.load(open('fin_LR_cla_L2_unbal_model_autism.sav','rb'))
loaded_model_L2_bal_autism = pickle.load(open('fin_LR_cla_L2_bal_model_autism.sav','rb'))

#L1 models
loaded_model_L1_Unbal_autism = pickle.load(open('fin_LR_cla_L1_unbal_model_autism.sav','rb'))
loaded_model_L1_bal_autism = pickle.load(open('fin_LR_cla_L1_bal_model_autism.sav','rb'))



##Chosen Random Forest Models
print('********************** Random Forest Models************************ \n')
#E_train
predicted_label_on_train_RF = loaded_model_RF_Unbal_autism.predict(X_tr_80_arr)
accuracy_on_train_RF = accuracy_score(y_tr_80_arr, predicted_label_on_train_RF)
print('Accuracy on train data set with RF is : ',accuracy_on_train_RF )

#E_val_test(model1)
predicted_label_on_val_test_RF = loaded_model_RF_Unbal_autism.predict(X_val_te_50_arr)
accuracy_on_val_test_RF = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_RF)
print('Accuracy on validation test data set with RF is : ',accuracy_on_val_test_RF )

print('Selected parameters : \n' , loaded_model_RF_Unbal_autism.get_params()  , '\n')

#E_train_bal
predicted_label_on_train_bal_RF = loaded_model_RF_bal_autism.predict(X_bal_arr)
accuracy_on_train_bal_RF = accuracy_score(y_bal_arr, predicted_label_on_train_bal_RF)
print('Accuracy on balanced train data set with RF is : ',accuracy_on_train_bal_RF )

#E_val_test(model2)
predicted_label_on_val_test_bal_RF = loaded_model_RF_bal_autism.predict(X_val_te_50_arr)
accuracy_on_val_test_bal_RF = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_bal_RF)
print('Accuracy on validation test data set with RF trained by balanced D'' is : ',accuracy_on_val_test_bal_RF )

print('Selected parameters : \n' , loaded_model_RF_bal_autism.get_params()  , '\n')


##Chosen L2 models
print('********************** L2 Models************************ \n')
#E_train
predicted_label_on_train_L2 = loaded_model_L2_Unbal_autism.predict(X_tr_80_arr)
accuracy_on_train_L2 = accuracy_score(y_tr_80_arr, predicted_label_on_train_L2)
print('Accuracy on train data set with L2 is : ',accuracy_on_train_L2 )

#E_val_test(model3)
predicted_label_on_val_test_L2 = loaded_model_L2_Unbal_autism.predict(X_val_te_50_arr)
accuracy_on_val_test_L2 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_L2)
print('Accuracy on validation test data set with L2 is : ',accuracy_on_val_test_L2 )

print('Selected parameters : \n' , loaded_model_L2_Unbal_autism.get_params()  , '\n')

#E_train_bal
predicted_label_on_train_bal_L2 = loaded_model_L2_bal_autism.predict(X_bal_arr)
accuracy_on_train_bal_L2 = accuracy_score(y_bal_arr, predicted_label_on_train_bal_L2)
print('Accuracy on balanced train data set with L2 is : ',accuracy_on_train_bal_L2 )

#E_val_test(model4)
predicted_label_on_val_test_bal_L2 = loaded_model_L2_bal_autism.predict(X_val_te_50_arr)
accuracy_on_val_test_bal_L2 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_bal_L2)
print('Accuracy on validation test data set with L2 trained by balanced D'' is : ',accuracy_on_val_test_bal_L2 )

print('Selected parameters : \n' , loaded_model_L2_bal_autism.get_params()  , '\n')


##Chosen L1 models
print('********************** L1 Models************************ \n')
#E_train
predicted_label_on_train_L1 = loaded_model_L1_Unbal_autism.predict(X_tr_80_arr)
accuracy_on_train_L1 = accuracy_score(y_tr_80_arr, predicted_label_on_train_L1)
print('Accuracy on train data set with L1 is : ',accuracy_on_train_L1 )

#E_val_test(model5)
predicted_label_on_val_test_L1 = loaded_model_L1_Unbal_autism.predict(X_val_te_50_arr)
accuracy_on_val_test_L1 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_L1)
print('Accuracy on validation test data set with L1 is : ',accuracy_on_val_test_L1 )

print('Selected parameters : \n' , loaded_model_L1_Unbal_autism.get_params()  , '\n')

#E_train_bal
predicted_label_on_train_bal_L1 = loaded_model_L1_bal_autism.predict(X_bal_arr)
accuracy_on_train_bal_L1 = accuracy_score(y_bal_arr, predicted_label_on_train_bal_L1)
print('Accuracy on balanced train data set with L1 is : ',accuracy_on_train_bal_L1 )

#E_val_test(model6)
predicted_label_on_val_test_bal_L1 = loaded_model_L1_bal_autism.predict(X_val_te_50_arr)
accuracy_on_val_test_bal_L1 = accuracy_score(y_val_te_50_arr, predicted_label_on_val_test_bal_L1)
print('Accuracy on validation test data set with L1 trained by balanced D'' is : ',accuracy_on_val_test_bal_L1 )

print('Selected parameters : \n' , loaded_model_L1_bal_autism.get_params()  , '\n')


#Final model selection and its final performance measures on D_fin_test
print('********************** Final Model************************ \n')
#Random Forest trained by balanced data set is chosen as final model
#Calculate E_fin_test
#model2
predicted_label_on_fin_test_bal_RF = loaded_model_RF_bal_autism.predict(X_fin_te_50_arr)
accuracy_on_fin_test_bal_RF = accuracy_score(y_fin_te_50_arr, predicted_label_on_fin_test_bal_RF)
print('Accuracy on final test data set with RF trained by balanced D'' is : ',accuracy_on_fin_test_bal_RF )

#Calculate Confusion Matrix
print('Confusion matrix: \n', confusion_matrix(y_fin_te_50_arr, predicted_label_on_fin_test_bal_RF))

#Calculate F-1 Score
print('F1 score: ', f1_score(y_fin_te_50_arr, predicted_label_on_fin_test_bal_RF, average='macro'))

#Final model's parameters
print(print('Selected parameters : \n' , loaded_model_RF_bal_autism.get_params()  , '\n'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




