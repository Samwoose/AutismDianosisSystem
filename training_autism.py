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



#########################Collection of functions##############################
def averageCalculator(accuracy_list):
    average = sum(accuracy_list) / len(accuracy_list)
    return average

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





def logisticRegressionExecuter(X_tr, y_tr, X_te, y_te, regularizer):
    '''
    Input:
    1.X_tr & y_tr : training data set(i.e., features and labels respectively)  , type: data frame or numpy array
    2.X_te & y_te : test data set(i.e., features and labels respectively) , type: data frame or numpy array
    3.regularizer : name of regularizer used in this algorithm. e.g., L1 : Lasso regression, L2: Logistic regression , type:String
    
    Output:
    1.final_c : chosen amount of penalty(regularizer) = 1/lambda  , type: float
    2.reularizer : used regularizer , type: string
    3.ave_accuracy_with_chosen_C : average accuracy with the chosen C. It is found by cross validation , type: float  
    4.fin_accuracy_on_train : final accuracy on the train data set , type: float
    5.fin_accuracy_on_test : final accuracy on the test data set , type: float
    6.fin_LR_cla : chosen model, type: object
    
    '''
    
    #preassignment session
    c_values_50points = np.logspace(-2 , 3 , num = 50)
    # lambda is big(i.e. 10^2) => C = 0.01 inverse proportionally
    # lambda is small(i.e. 10^-3) => C = 1000 inverse proportionally
    
    number_of_fold = 5
    kf = KFold(n_splits = number_of_fold, shuffle = True)

    #storage for average accuracy for each lambda
    stor_ave_acc = []
    
    #Cross Validation
    for i in range(0, len(c_values_50points)):
        temp_c = c_values_50points[i]
        
        #create a storage for 5 accuracies for same lambda and different train and test set from the whole train set
        temp_acc_list = []
        
        print('Cross Validation Processing : current value of C is ', temp_c )
        
        #calculate accuracy 5 times with the same lambda
        for k in range(0,5):
            
            for train_index , test_index in kf.split(X_tr):
                X_val_tr , X_val_test = X_tr[train_index] , X_tr[test_index]
                y_val_tr , y_val_test = y_tr[train_index] , y_tr[test_index]
                
                X_val_tr_arr = np.array(X_val_tr)
                X_val_test_arr = np.array(X_val_test)
                
                y_val_tr_arr = np.array(y_val_tr)
                y_val_test_arr = np.array(y_val_test)
                
                if(regularizer == 'l1'):
                    #Logistic Regression for each lambda and each time with different train and test set from the whole train set
                    LR_cla_val = LogisticRegression(penalty = regularizer, C = temp_c, solver = 'saga' , max_iter = 1000).fit(X_val_tr_arr, np.ravel(y_val_tr_arr , order = 'C') )
                
                elif(regularizer == 'l2'):
                    #Logistic Regression for each lambda and each time with different train and test set from the whole train set
                    LR_cla_val = LogisticRegression(penalty = regularizer, C = temp_c, solver = 'sag' , max_iter = 1000).fit(X_val_tr_arr, np.ravel(y_val_tr_arr , order='C') )
                
                predicted_label_on_test_data = LR_cla_val.predict(X_val_test_arr)
                
                #calculate accuracy    
                accuracy_on_x_test = accuracy_score(y_val_test_arr, predicted_label_on_test_data)
                
                #save accuracy for each lambda 5 times total
                temp_acc_list.append(accuracy_on_x_test)
                
        #save average of 5 accuracies for each lambda
        average_acc = averageCalculator(temp_acc_list)
        stor_ave_acc.append(average_acc)
    
    #find location of C value
    c_location = stor_ave_acc.index(max(stor_ave_acc))
    final_c = c_values_50points[c_location]
    
    print('chosen C is ' , final_c , '\n')
    print('Average Acurracy with the chosen C is ' , max(stor_ave_acc) , '\n')
    
    #Thanks to code from https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected to resolve data conversion warning 
    #get final logistic regression model using the final c value we found by cross validation
    
    if(regularizer == 'l1'):
        fin_LR_cla = LogisticRegression(penalty = regularizer, C = final_c, solver = 'saga' , max_iter = 1000).fit(X_tr, np.ravel(y_tr , order = 'C') )
        
    elif(regularizer == 'l2'):
        fin_LR_cla = LogisticRegression(penalty = regularizer, C = final_c, solver = 'sag', max_iter = 1000).fit(X_tr, np.ravel(y_tr , order = 'C'))
        
    fin_predicted_label_on_train_data = fin_LR_cla.predict(X_tr)
    
    #calculate accuracy on whole train data set
    fin_accuracy_on_train = accuracy_score(y_tr, fin_predicted_label_on_train_data)
    print('Accuracy on whole train data with chosen C value is : ', fin_accuracy_on_train)
    
    #calculate accuracy on whole test data set
    fin_predicted_label_on_test_data = fin_LR_cla.predict(X_te)
    fin_accuracy_on_test = accuracy_score(y_te, fin_predicted_label_on_test_data)
    print('Accuracy on whole test data with chosen C value is : ', fin_accuracy_on_test)
    
    
    
    return final_c, max(stor_ave_acc) , fin_accuracy_on_train , fin_accuracy_on_test , regularizer , fin_LR_cla




def randomForestExecuter(X_tr, y_tr, X_te, y_te):
    '''
    Input:
    1.X_tr & y_tr : training data set(i.e., features and labels respectively)  , type: data frame or numpy array
    2.X_te & y_te : test data set(i.e., features and labels respectively) , type: data frame or numpy array
    
    Output:
    1.final_num_estimators : chosen number of estimators  , type: integer
    2.ave_accuracy_with_chosen_num_estimators : average accuracy with the number of the estimators. It is found by cross validation , type: float  
    3.fin_accuracy_on_train_RF : final accuracy on the train data set , type: float
    4.fin_accuracy_on_test_RF : final accuracy on the test data set , type: float
    5.fin_RF_clf : Chosen model, type: obejct
    
    '''
    
    '''
    RF = Random Forest
    '''
    
    
    #create possible number of estimators that will be used in cross validation to find best number of estimators
    n_estimators_values_list = []
    for i_RF in range(1,101):
        n_estimators_values_list.append(i_RF*10)
    n_estimators_values_arr = np.array(n_estimators_values_list) #range of number of estimators : from 10 to 1000, step size 10 
    
    num_of_fold_RF = 5
    kf_RF = KFold(n_splits = num_of_fold_RF, shuffle = True)
    
    #storage for average accuracy for each number of estimator
    stor_ave_acc_RF = []
    
    #Cross-Validation
    for j_RF in range(0,len(n_estimators_values_arr)):
        temp_n_estimator = n_estimators_values_arr[j_RF]
        
        #create a storage for 5 accuracies for same number of estimators and different train and test set from the whole train set
        temp_acc_list_RF = []
        
        print('Cross Validation Processing : current number of estimators is ', temp_n_estimator )
        
        for k_RF in range(0,5):
            
            for train_index_RF , test_index_RF in kf_RF.split(X_tr):
                X_val_tr_RF , X_val_test_RF = X_tr[train_index_RF] , X_tr[test_index_RF]
                y_var_tr_RF , y_val_test_RF = y_tr[train_index_RF] , y_tr[test_index_RF]
                
                X_val_tr_RF_arr = np.array(X_val_tr_RF)
                X_val_test_RF_arr = np.array(X_val_test_RF)
                
                y_val_tr_RF_arr = np.array(y_var_tr_RF)
                y_val_test_RF_arr = np.array(y_val_test_RF)
                
                #create D_bag with bag size 1 
                X_train_bag  = X_val_tr_RF_arr
                y_train_bag  = y_val_tr_RF_arr
                
                #Random Forest Process
                temp_RF_clf = RandomForestClassifier(n_estimators = temp_n_estimator, bootstrap=True)
                #training and predict on test validation set
                temp_RF_clf.fit(X_train_bag, np.ravel(y_train_bag,order='C'))
                predicted_label_on_test_val_data_RF = temp_RF_clf.predict(X_val_test_RF_arr)
                
                #evaluate accuracy on test data set
                accuracy_on_test_val_data_RF = accuracy_score(y_val_test_RF_arr, predicted_label_on_test_val_data_RF)
                
                #save accuracy for each number of estimators 5 times total
                temp_acc_list_RF.append(accuracy_on_test_val_data_RF)
                
        #save average of 5 accuracies for each number of estimators
        average_acc_RF = averageCalculator(temp_acc_list_RF)
        stor_ave_acc_RF.append(average_acc_RF)
    
    #find location of number of estimator value
    num_estimator_location = stor_ave_acc_RF.index(max(stor_ave_acc_RF))
    final_num_of_estimators = n_estimators_values_arr[num_estimator_location]
    print('Chosen number of estimators is ' , final_num_of_estimators , '\n' )
    print('Average Acurracy with the chosen number of estimators is ' , max(stor_ave_acc_RF) , '\n')
    
    #Final training
    fin_RF_clf = RandomForestClassifier(n_estimators = final_num_of_estimators, bootstrap = True)
    fin_RF_clf.fit(X_tr,np.ravel(y_tr,order='C'))
    fin_predicted_label_on_train_data_RF = fin_RF_clf.predict(X_tr)
    
    #calculate accuracy on whole train data set
    fin_accuracy_on_train_RF = accuracy_score(y_tr, fin_predicted_label_on_train_data_RF)
    print('Accuracy on whole train data with chosen number of estimator is : ', fin_accuracy_on_train_RF)
    
    #calculate accuracy on whole test data set
    fin_predicted_label_on_test_data_RF = fin_RF_clf.predict(X_te)
    fin_accuracy_on_test_RF = accuracy_score(y_te, fin_predicted_label_on_test_data_RF)
    print('Accuracy on whole test data with chosen number of estimator is : ' , fin_accuracy_on_test_RF)
    
    
    
    return final_num_of_estimators, max(stor_ave_acc_RF) , fin_accuracy_on_train_RF , fin_accuracy_on_test_RF, fin_RF_clf
      


####################################Main Training Part####################
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


#training Session 
####################Logistic Regression(L2 norm, Function version) on Unbalanced Data ###################(less than 14 mins)
c_L2, ave_acc_L2, fin_acc_tr_L2, fin_acc_test_L2, name_reg_L2, fin_LR_cla_L2_unbal = logisticRegressionExecuter(X_tr_80_arr, y_tr_80_arr, X_val_te_50_arr, y_val_te_50_arr, 'l2')

#save the model to disk
filename_L2_Unbal = 'fin_LR_cla_L2_unbal_model_autism.sav'
pickle.dump(fin_LR_cla_L2_unbal, open(filename_L2_Unbal, 'wb'))


####################Lasso Regression(L1 norm, Function version) on Unbalanced Data ###################(less than 22 mins)
c_L1, ave_acc_L1, fin_acc_tr_L1, fin_acc_test_L1, name_reg_L1, fin_LR_cla_L1_unbal = logisticRegressionExecuter(X_tr_80_arr, y_tr_80_arr, X_val_te_50_arr, y_val_te_50_arr, 'l1')

#save the model to disk
filename_L1_Unbal = 'fin_LR_cla_L1_unbal_model_autism.sav'
pickle.dump(fin_LR_cla_L1_unbal, open(filename_L1_Unbal, 'wb'))

####################Random Forest(function version ) on Unbalanced Data ###################(Less than  30 mins)
fin_num_of_estimators, ave_acc_RF , fin_acc_on_tr_RF , fin_acc_on_test_RF, fin_RF_clf_unbal = randomForestExecuter(X_tr_80_arr, y_tr_80_arr, X_val_te_50_arr, y_val_te_50_arr)

#save the model to disk
filename_RF_Unbal = 'fin_RF_clf_unbal_model_autism.sav'
pickle.dump(fin_RF_clf_unbal, open(filename_RF_Unbal, 'wb'))

####################Logistic Regression(L2 norm, Function version) on balanced Data ###################(less than  5 mins)
c_L2_bal, ave_acc_L2_bal, fin_acc_tr_L2_bal, fin_acc_test_L2_bal, name_reg_L2_bal, fin_LR_cla_L2_bal = logisticRegressionExecuter(X_bal_arr, y_bal_arr, X_val_te_50_arr, y_val_te_50_arr, 'l2')

#save the model to disk
filename_L2_bal = 'fin_LR_cla_L2_bal_model_autism.sav'
pickle.dump(fin_LR_cla_L2_bal, open(filename_L2_bal, 'wb')) 

####################Lasso Regression(L1 norm, Function version) on balanced Data & evaluate accuracy on 10 percent test set ###################(less than 13 mins)
c_L1_bal, ave_acc_L1_bal, fin_acc_tr_L1_bal, fin_acc_test_L1_bal, name_reg_L1_bal, fin_LR_cla_L1_bal = logisticRegressionExecuter(X_bal_arr, y_bal_arr, X_val_te_50_arr, y_val_te_50_arr, 'l1')

#save the model to disk
filename_L1_bal = 'fin_LR_cla_L1_bal_model_autism.sav'
pickle.dump(fin_LR_cla_L1_bal, open(filename_L1_bal, 'wb')) 

####################Random Forest(function version ) on balanced Data & evaluate accuracy on 10 percent test set ###################(Less than 20  mins)
fin_num_of_estimators_bal, ave_acc_RF_bal , fin_acc_on_tr_RF_bal , fin_acc_on_test_RF_bal, fin_RF_clf_bal = randomForestExecuter(X_bal_arr, y_bal_arr, X_val_te_50_arr, y_val_te_50_arr)

#save the model to disk
filename_RF_bal = 'fin_RF_clf_bal_model_autism.sav'
pickle.dump(fin_RF_clf_bal, open(filename_RF_bal, 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




