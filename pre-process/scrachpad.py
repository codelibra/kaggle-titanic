combined.head()


cols



mean = combined['Fare'].mean()
std = combined['Fare'].std()

mean
std


combined['distance_fare'] = [ abs((x-mean)/std) for x in combined['Fare']]
combined['distance_fare']

combined = combined.drop('distance_fare', axis=1)
scores

combined.columns




skf = StratifiedKFold(targets, n_folds=10)


perf = 0
for train_index, test_index in skf:
    X_train, X_test = train_new[train_index], train_new[test_index]
    y_train, y_test = targets[train_index], targets[test_index]
    gbm.fit(X_train,y_train)
    perf = perf + gbm.score(X_test,y_test)
    print gbm.score(X_test,y_test)


print "Performance"
print perf/10
