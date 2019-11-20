
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
from sklearn.ensemble import RandomForestRegressor


def prepdf_or_featureselection(mydata,myfeature_importances = None,prep=True):
    
    #create location feature for data using optics clustering
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

    #print(mydata.shape[1])#39
    mydata = pd.concat([mydata,optics_df_1hot],axis=1)
    
    
    #print(mydata.shape[1])#42
    #drop unneccessary columns in our case
    mydata = mydata.drop(['city','Latitude','Longitude','change_hunits','studio_1000_1499', 'studio_1500_more',
       'studio_750_999', 'onebed_1000_1499', 'onebed_1500_more',
       'onebed_750_999', 'twobed_1000_1499', 'twobed_1500_more',
       'twobed_750_999', 'threebed_1000_1499', 'threebed_1500_more',
       'threebed_750_999'],axis=1)
    feature_data = mydata.copy()
    mydata = mydata.drop('med_rental_rate',axis =1)
    


    if prep:
        mydatacolumns = mydata.columns
        #print(mydata.shape[1])#37

        #prepare data section



        imputer = IterativeImputer(max_iter = 10 ,random_state =22,min_value=0)
        mydata = imputer.fit_transform(mydata)
        #scale only numerical attrbs which are everything but the columns which were appended earlier
        #print(len(location_labels),mydata.shape[1])
        num_attrbs = mydata.shape[1]-len(location_labels)

        ct_columns = list(range(num_attrbs))
        

        ct = ColumnTransformer(
            [('scale1',RobustScaler(),ct_columns)],
            remainder = 'passthrough')

        mydata = ct.fit_transform(mydata)
        myfeature_selection = 'onlyprep_selected'
    
    else:
        num_pipeline = Pipeline([
            ('imputer',IterativeImputer(max_iter = 10,random_state = 22,min_value=0)),
            ('rob_scaler',RobustScaler())
        ])
        #only num attributes
        num_attrbs = mydata.shape[1]-len(location_labels)
        ct_columns = list(range(num_attrbs))
        
        full_pipeline = ColumnTransformer(
            [('num',num_pipeline,ct_columns)],
            remainder = 'passthrough')
        #Thanks to Aurelien Geron https://github.com/ageron for TopFeatureSelector
        k= 10
        feature_importances = myfeature_importances
        def indices_of_top_k(arr, k):
            return np.sort(np.argpartition(np.array(arr), -k)[-k:])

        class TopFeatureSelector(BaseEstimator, TransformerMixin):
            def __init__(self, feature_importances, k):
                self.feature_importances = feature_importances
                self.k = k
            def fit(self, X, y=None):
                self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
                return self
            def transform(self, X):
                return X[:, self.feature_indices_]
        
        prepare_select_and_predict_pipeline = Pipeline([
            ('preparation',full_pipeline),
            ('feature_selection',TopFeatureSelector(feature_importances,k)),
            ('rf_reg',RandomForestRegressor(random_state=22,n_estimators = 100))
        ])
        
        param_grid = [{'feature_selection__k':list(range(1,len(feature_importances)+1))}]
        
        grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline,param_grid,cv=5,scoring = 'neg_mean_squared_error',n_jobs = -1)
        
        grid_search_prep.fit(feature_data.drop('med_rental_rate',axis = 1),feature_data['med_rental_rate'].copy())
        myfeature_selection = grid_search_prep.best_params_
        mydata = 'onlyprep_selected'
        mydatacolumns = 'onlyprep_selected'
    return {'mydata':mydata,'mydatacolumns':mydatacolumns,'myfeature_selection':myfeature_selection}
