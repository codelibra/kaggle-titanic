
# coding: utf-8



base_folder = '/Users/shiv/.bin/kaggle-titanic-test'


# # Imports


import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().magic(u'matplotlib inline')
import seaborn as sns

import numpy as np
import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing




# # Data Loading


data = pd.read_csv(base_folder + '/train.csv')



def status(feature):
    print 'Processing',feature,': ok'



# Combined training and test data
# This can be easily broken back to train and test easily
def get_combined_data():
    train = pd.read_csv(base_folder + '/train.csv')
    test = pd.read_csv(base_folder + '/test.csv')
    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived',1,inplace=True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    return combined,train,targets
combined,train,targets = get_combined_data()









def get_titles():
    global combined
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    combined['Title'] = combined.Title.map(Title_Dictionary)
get_titles()



def process_age():
    global combined
    # fill the missing ages with the median value just calculated
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

    status('age')
process_age()




def split_age():
    # new columns m planning to create are age ranges
    # 10-20, 20-30 something like that
    combined['20-40'] = combined['Age'].apply(lambda x: 1 if x>=20 and x<=40 else 0)
    combined['70-80'] = combined['Age'].apply(lambda x: 1 if x>=70 and x<=80 else 0)
def split_fare():
    # new columns m planning to create are age ranges
    # 10-20, 20-30 something like that
    combined['below-80'] = combined['Fare'].apply(lambda x: 1 if x<80 else 0)
split_fare()
split_age()




def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')


    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)

    status('names')

process_names()

def process_fares():
    global combined
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    status('fare')
process_fares()



def process_embarked():
    global combined
    combined.Embarked.fillna('S',inplace=True)
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)

    status('embarked')
process_embarked()



def process_cabin():

    global combined

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    combined = pd.concat([combined,cabin_dummies],axis=1)

    combined.drop('Cabin',axis=1,inplace=True)

    status('cabin')
process_cabin()



def process_sex():
    global combined
    # mapping string values to numerical one
    sex_dummies = pd.get_dummies(combined['Sex'],prefix='Sex')
    combined = pd.concat([combined,sex_dummies],axis=1)
    combined['Sex'] = combined['Sex'].map({'male':0,'female':2})

    status('sex')
process_sex()

combined


def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)

    status('pclass')
process_pclass()



def process_ticket():
    global combined

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'


    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)
    status('ticket')

process_ticket()



def process_family():
    global combined
    # introducing a new feature
    #the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)

    status('family')
process_family()


# # Shortcomings
# * Added many features without knowing whether it will help or not?
# * Like family size, titile etc...
#
# ### Better idea
# * If merging of both the trainig and test is required for cross validation, do it in the beginning itself!
# * Name all methods where there is transformation of data to __transform__ what function does

# ## Normalising all features


combined.head()
def scale_all_features():
    global combined
    combined['Age'] = combined['Age']/max(combined['Age']) - min(combined['Age'])
    combined['Fare'] = combined['Fare']/max(combined['Fare']) - min(combined['Fare'])
    print 'Features scaled successfully !'
scale_all_features()
combined

### Feature selection

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=200)
clf = grid_search.fit(train, targets)



features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
cols =  features.sort(['importance'],ascending=False)['feature']


cols
model = SelectFromModel(grid_search, prefit=True)
train_new = model.transform(train)
print train_new.shape
print train.shape

test_new = model.transform(test)

# # Model


forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8,9],
                 'n_estimators': [100, 200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



from sklearn.ensemble import ExtraTreesClassifier
ext = ExtraTreesClassifier()

parameter_grid = {
                 'max_depth' : [4,5,6,7,8,9],
                 'n_estimators': [100, 200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(ext,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2')

parameter_grid = {
                 'tol' : [0.1,0.01,0.001,10,1],
                 'max_iter': [100, 200,210,240,250],
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(lr,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=100)

cross_validation = StratifiedKFold(targets, n_folds=5)
adaboost.fit(train_new, targets)

print('Best score: {}'.format(cross_val_score(adaboost,train_new,targets,cv=10)))



from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('rf', forest),('etc',ext),('lr', lr), ('adb', adaboost)], voting='soft',
                        weights=[2,1,1,1])
eclf1 = eclf1.fit(train_new, targets)
predictions=eclf1.predict(test_new)
predictions

test_predictions=eclf1.predict(test_new)
test_predictions=test_predictions.astype(int)



eclf1.score(train_new, targets)



ext.fit(train_new,targets)
test_predictions = ext.predict(test_new)


# ## Ensemble stacking and blending


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }



# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = StratifiedKFold(targets,n_folds= NFOLDS)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_

# Class to extend XGboost classifer



# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# ### What does this out of fold thing fucking does??

# * First of all.. enumerate(kf). What does this do?
# * This function returns indices of the 90%train and 10%test set which it generated from each fold of the same training set data.
# * Using that data we can split the overall training data itself in to traing and test set for each fold... and  then utilize it for generating our k fold results if required.
#
# * calculates mean predicts over the i folds in the test set
# * calculates predictions over all folds of the training set
# * finally returns both the predictions


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        # get the training of fold number i from training set
        x_tr = train_new[train_index]
        # get the targets of fold i from training set
        y_tr = targets[train_index]
        # get the remaining 10% test set from the ith fold
        x_te = train_new[test_index]

        # train the classifier on the training set
        clf.train(x_tr, y_tr)

        # store results of predictions over the ith test set at proper locations
        # oof_train will contain all the predictions over the test set once all n_fold iterations are over
        oof_train[test_index] = clf.predict(x_te)
        # over the complete test set classifier trained so far will predict
        # ith entry of oof_test_skf will contain predictions from classifier trained till ith fold
        oof_test_skf[i, :] = clf.predict(x_test)

    # calculate mean of all the predictions done in the i folds and store them as final results in oof_test
    oof_test[:] = oof_test_skf.mean(axis=0)
    # predictions on training set, mean predictions on the test set
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# ### Training all features


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, train_new, targets, test_new) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,train_new, targets, test_new) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, train_new, targets, test_new) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,train_new, targets, test_new) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,train_new, targets, test_new) # Support Vector Classifier

print("Training is complete")


# ## Feature importances


rf_feature = rf.feature_importances(train_new,targets)
et_feature = et.feature_importances(train_new, targets)
ada_feature = ada.feature_importances(train_new, targets)
gb_feature = gb.feature_importances(train_new,targets)



rf_feature = [ 0.06897607 , 0.31750647 , 0.10071629  ,0.02489255 , 0.02146597  ,0.15484932,
  0.02079024  ,0.01582646 , 0.05257318 , 0.04320778 , 0.0889579 ,  0.04838885,
  0.04184892]
et_feature = [ 0.02517265  ,0.44867491 , 0.05259319  ,0.02153428  ,0.02317999 , 0.04183662,
  0.02265657,  0.03095267 , 0.08328287  ,0.04941365 , 0.10086282 , 0.03224508,
  0.06759468]
ada_feature = [ 0.39  , 0.02  , 0.154 , 0.004  ,0.028  ,0.346,  0. ,    0. ,    0.004,  0.002,
  0.002 , 0.05 ,  0.   ]
gb_feature = [  4.56662903e-01 ,  3.51987023e-02 ,  1.61809918e-01 ,  1.24079649e-02,
   9.20911057e-03  , 2.42530451e-01 ,  3.15165895e-04 ,  1.59087209e-02,
   1.44507445e-02  , 7.73641745e-03  , 1.45861525e-02,   1.52261027e-02,
   1.39576462e-02]







cols = ['Sex','PassengerId','Age','Fare','Pclass_3','20-40','Cabin_U','SmallFamily','Pclass_1','FamilySize','Parch','SibSp','below-80']
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })



feature_dataframe


# ## Second level predictions from first level

# * stored the predictions on training set of various classifiers into flattened array


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train



import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Portland',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# * converted into a single array of training set(891) X 4 columns(number of classifiers)


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)



x_train[0]



import xgboost as xgb
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# ## output


test = pd.read_csv('test.csv')



df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = predictions
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)



get_ipython().run_cell_magic(u'bash', u'', u'kg submit output.csv -c titanic -u sp4658@nyu.edu -p shivi2909 -m "voting classifier after changing feature selector"')



get_ipython().run_cell_magic(u'bash', u'', u'pip install kaggle-cli')



get_ipython().run_cell_magic(u'bash', u'', u'curl --upload-file ./output.csv https://transfer.sh/output.csv')



get_ipython().run_cell_magic(u'bash', u'', u'git clone --recursive https://github.com/dmlc/xgboost;cd xgboost; make -j4')
