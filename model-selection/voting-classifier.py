### Feature selection

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)



features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
cols =  features.sort(['importance'],ascending=False)['feature']

model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
test_new = model.transform(test)

# # Model
scores = []

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
scores.append(grid_search.best_score_)
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
scores.append(grid_search.best_score_)
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
scores.append(grid_search.best_score_)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=100)

cross_validation = StratifiedKFold(targets, n_folds=5)
adaboost.fit(train_new, targets)

scores.append(sum(cross_val_score(adaboost,train_new,targets,cv=10))/10)
print('Best score: {}'.format(cross_val_score(adaboost,train_new,targets,cv=10)))



import xgboost as xgb
gbm = xgb.XGBClassifier(
    learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(train_new, targets)

perf = 0
for train_index, test_index in skf:
    X_train, X_test = train_new[train_index], train_new[test_index]
    y_train, y_test = targets[train_index], targets[test_index]
    gbm.fit(X_train,y_train)
    perf = perf + gbm.score(X_test,y_test)

print "Performance"
scores.append(perf/10)


from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('rf', forest),('etc',ext),('lr', lr), ('adb', adaboost),
        ('gbm', gbm)
        ], voting='soft',
                        weights=scores)
eclf1 = eclf1.fit(train_new, targets)



predictions=eclf1.predict(test_new)



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
