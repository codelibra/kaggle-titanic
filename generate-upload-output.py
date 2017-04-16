df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = predictions
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

!kg submit output.csv -c titanic -u sp4658@nyu.edu -p shivi2909 -m "rf"
