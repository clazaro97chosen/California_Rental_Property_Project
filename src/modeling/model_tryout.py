
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



def model_pipeline(myestimator,mydata,myfolds,feature_selection_done = False,myfeatures = None,checknoise = False):
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
    # part 1 create location feature for data using optics clustering
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
   
    #part2done

    #part3perform cross validation while scaling and imputing on the folds
    
    skfolds = KFold(n_splits = myfolds,random_state=22,shuffle=True)
    results = []

    mydata = (np.array(mydata))
    mydata_labels = (np.array(mydata_labels))
    for train_index, test_index in skfolds.split(mydata,mydata_labels):
        clone_est = clone(myestimator)
        X_train_folds = mydata[train_index]
        y_train_folds = mydata_labels[train_index]
        X_test_fold = mydata[test_index]
        y_test_fold = mydata_labels[test_index]
        
        #impute
        imputer = IterativeImputer(max_iter = 10 ,random_state =22,min_value=0)
        X_train_folds = imputer.fit_transform(X_train_folds)
        #scale only numerical attrbs which are everything but the columns which were appended earlier
        num_attrbs = mydata.shape[1]-len(location_labels)
        ct_columns = list(range(num_attrbs))
        
        
        
        ct = ColumnTransformer(
            [('scale1',RobustScaler(),ct_columns)],
            remainder = 'passthrough')
        
        X_train_folds = ct.fit_transform(X_train_folds)
        
        
        clone_est.fit(X_train_folds,y_train_folds)
        #transform do not fit X_test_fold in order to predict
        X_test_fold = imputer.transform(X_test_fold)
        X_test_fold = ct.transform(X_test_fold)
        y_pred = clone_est.predict(X_test_fold)
        if checknoise:
            fold_expvar = explained_variance_score(y_test_fold,y_pred)
            results.append(fold_expvar)
        else:
            fold_mse = mean_squared_error(y_test_fold,y_pred)
            results.append(fold_mse)
    if checknoise:
        scores = (np.array([results]))
    else:
        scores = np.sqrt(np.array([results]))
    
    print('Scores',scores)
    print('Mean',scores.mean())
    print('Standard Deviation',scores.std())
    

    