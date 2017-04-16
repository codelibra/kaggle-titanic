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

def eval_model(model=eclf1):
    skf = StratifiedKFold(targets, n_folds=10)
    perf = 0
    for train_index, test_index in skf:
        X_train, X_test = train_new[train_index], train_new[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        eclf1.fit(X_train,y_train)
        perf = perf + eclf1.score(X_test,y_test)
        print eclf1.score(X_test,y_test)


    print "Performance"
    print perf/10

def recover_train_test_target():
    global combined
    train0 = pd.read_csv(base_folder + '/train.csv')
    targets = train0.Survived
    train = combined[0:891]
    test = combined[891:]
    return train,test,targets
train,test,targets = recover_train_test_target()


predictions = evaluate(train_new , targets, test_new)
