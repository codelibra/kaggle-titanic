def evaluate(data, targets, test):

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l2')

    parameter_grid = {
                     'tol' : [0.1,0.01,0.001,10,1],
                     'max_iter': [100, 200,210,240,250],
                     }

    cross_validation = StratifiedKFold(targets, n_folds=10)

    grid_search = GridSearchCV(lr,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(data, targets)

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    return grid_search.predict(test)

def recover_train_test_target():
    global combined
    train0 = pd.read_csv(base_folder + '/train.csv')
    targets = train0.Survived
    train = combined[0:891]
    test = combined[891:]
    return train,test,targets
train,test,targets = recover_train_test_target()


predictions = evaluate(train_new , targets, test_new)

predictions = evaluate(train._get_numeric_data() , targets, test._get_numeric_data())
