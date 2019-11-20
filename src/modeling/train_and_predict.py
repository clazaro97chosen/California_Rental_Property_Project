import numpy as np
import pandas as pd
# to make this  stable across runs
np.random.seed(22)

from sklearn.cluster import OPTICS
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

def winning_pipeline(mydata,mytestdata,myfinalmodel,feature_selection_done = True,myfeatures =None,numerical_attributes = None):
    """
    If feature _selection has not been performed:
        Function performs Cross Validation (with scaling within folds) on the data passed through. 
        Scales the data with RobustScaler() and Imputes the data with IterativeImputer(). Additionally adds clusters for the cities latitude and longitude
    Else:
        Performs Cross-Validation given the estimator on a subset of the features of mydata which were passed through to myfeatures
        
    Arguments
        @myestimator: sklearn estimator
        @mydata: training data with  missing values and is not scaled)
        @myfolds: number of folds for cross validation 
        @feature_selection_done: Boolean flag indicating if feature_selection has been done to the data in `mydata`
        @myfeatures: list of informative features from features
        @checknoise: Whether scoring for Cross-Validation should be Explained Variance 
    
    """
    # part 1 create location feature for training data using optics clustering
    optics_df = mydata[['Latitude','Longitude']].copy()
    clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
    clust.fit(optics_df)
    #
    optics_df['clust_label'] = clust.labels_
    #
    location_max = np.max(optics_df.clust_label.unique())
    #optics labels noisy samples as -1 need to replace for successful onehotencoding
    optics_df['clust_label'].replace([-1],location_max+1,inplace=True)
    #one hot encoding and combining to mydata
    enc = OneHotEncoder(categories='auto')
    
    optics_df_1hot = enc.fit_transform(optics_df[['clust_label']])

    location_labels = ['cluster' + str(l) for l in optics_df.clust_label.unique()]
    
    optics_df_1hot = pd.DataFrame(optics_df_1hot.todense(),index = optics_df.index,columns= location_labels )
    #part1done cluster columns added 
    
    
    mydata = pd.concat([mydata,optics_df_1hot],axis=1)
    
    
    
    
    #part 2 drop unneccessary columns in our case
    
    mydata_labels = mydata['med_rental_rate'].copy()
    
    mydata = mydata.drop('med_rental_rate',axis =1)
    
    if feature_selection_done:
        mydata = mydata.loc[:,myfeatures].copy()
    else: 
        mydata = mydata.drop(['city','Latitude','Longitude','change_hunits','studio_1000_1499', 'studio_1500_more',
       'studio_750_999', 'onebed_1000_1499', 'onebed_1500_more',
       'onebed_750_999', 'twobed_1000_1499', 'twobed_1500_more',
       'twobed_750_999', 'threebed_1000_1499', 'threebed_1500_more',
       'threebed_750_999'],axis=1)
   
   
    
    
    imputer = IterativeImputer(max_iter = 10 ,random_state =22,min_value=0)
    imputed_dat = imputer.fit_transform(mydata)
    
    
    #scale only numerical attrbs which are everything but the columns which were appended earlier
    imputed_dat = pd.DataFrame(imputed_dat,columns=mydata.columns)
    
    ct = ColumnTransformer(
            [('scale1',RobustScaler(),numerical_attributes)],
            remainder = 'passthrough')
        
    X_train_prepped = ct.fit_transform(imputed_dat)
    
    #to pickle 
    processed_training_data = X_train_prepped.copy()
    
    #nowfor the test data
    
    
    # part 1 create location feature for test data using optics clustering
    optics_df = mytestdata[['Latitude','Longitude']].copy()
    clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
    clust.fit(optics_df)
    #
    optics_df['clust_label'] = clust.labels_
    #
    location_max = np.max(optics_df.clust_label.unique())
    #optics labels noisy samples as -1 need to replace for successful onehotencoding
    optics_df['clust_label'].replace([-1],location_max+1,inplace=True)
    #one hot encoding and combining to mydata
    enc = OneHotEncoder(categories='auto')
    
    optics_df_1hot = enc.fit_transform(optics_df[['clust_label']])

    location_labels = ['cluster' + str(l) for l in optics_df.clust_label.unique()]
    
    optics_df_1hot = pd.DataFrame(optics_df_1hot.todense(),index = optics_df.index,columns= location_labels )
    #part1done cluster columns added 
    
    
    mytestdata = pd.concat([mytestdata,optics_df_1hot],axis=1)
    
    
    
    
    #part 2 drop unneccessary columns in our case
    
    mytest_data_labels = mytestdata['med_rental_rate'].copy()
    mytestdata = mytestdata.drop('med_rental_rate',axis =1)
    
    if feature_selection_done:
        mytestdata = mytestdata.loc[:,myfeatures].copy()
    else: 
        mydata = mydata.drop(['city','Latitude','Longitude','change_hunits','studio_1000_1499', 'studio_1500_more',
       'studio_750_999', 'onebed_1000_1499', 'onebed_1500_more',
       'onebed_750_999', 'twobed_1000_1499', 'twobed_1500_more',
       'twobed_750_999', 'threebed_1000_1499', 'threebed_1500_more',
       'threebed_750_999'],axis=1)
        
    #prepare testdata them
    imputed_testdata = imputer.transform(mytestdata)
    imputed_testdata = pd.DataFrame(imputed_testdata,columns=mytestdata.columns)
    mytestdata_prepared = ct.transform(imputed_testdata)

    #to pickle 
    processed_test_data = mytestdata_prepared.copy()
    #make final predictions
    myfinalmodel.fit(X_train_prepped,mydata_labels)
    final_predictions = myfinalmodel.predict(mytestdata_prepared)
    
    final_mse = mean_squared_error(mytest_data_labels,final_predictions)
    final_rmse = np.sqrt(final_mse)
    final_expvar = explained_variance_score(mytest_data_labels,final_predictions)
    return {'final_rmse':final_rmse,'final_predictions':final_predictions,'final_expvar':final_expvar,'myfinalmodel':myfinalmodel,
    'processed_training_data':processed_training_data,'processed_test_data':processed_test_data}

    