
# filled the empty age with median value of age
data['Age'].fillna(data['Age'].median(), inplace=True)
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()



#plot the survived male , female and dead male,female
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar', figsize=(15,8))



# * It can be clearly seen from the above graph that females survied more than men.
# * Much more than men actually!
# * Should the values be just 0 and 1 for male and female.. or should the diffrence be more?


# dead and survived based on age of people
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], color = ['g','r'],
         bins = 10,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

# * what can be seen here is..
# * those in the range 20-40 are more likely to be dead.
# * those in teh range 70-80 are almost always dead
# * 0-20 there is not much diff i think
# * making these as features would be a good idea?
#


# plotting number of survivors based on the fare they gave
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], color = ['g','r'],
         bins = 10,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# * Not exactly sure whether making <50 a feature will be a good idea? Although people less than 50 have high death rate!!
# * But over the complete data set we cannot say anything substancial from the fare alone


# depending upon age the rate of survival
# clearly see that, lower part of reds and above part is green suggesting... lower fares were killed early!
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=20,)


# * Now i know that indivisually age between 20-40 are killed more.
# * Also i know indivisually those with lower fares are also killed more.
# * <font color="red">Should i have indivisual features of these? or should i combine both into single feature and that will be better predictor?</font>


ax = plt.subplot()
ax.set_ylabel('Survived')
ax.set_xlabel('Pclass')
ax.hist([data[data['Survived']==1]['Pclass'],data[data['Survived']==0]['Pclass']],color = ['g','r'],)


# * So from the above we see pclass3 is mostly dead. Other classes are not giving much info.


# Plotting how fares versus pclass goes?
ax = plt.subplot()
ax.set_ylabel('Average fare')
# we are plotting the mean cause the mean would show overall co-relation
#rather than indivisual data points which may be unclear
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)



survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
