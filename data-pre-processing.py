
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

mean = combined['Fare'].mean()
std = combined['Fare'].std()
combined['distance_fare'] = [ abs((x-mean)/std) for x in combined['Fare']]

def scale_all_features():
    global combined
    combined['Age'] = combined['Age']/max(combined['Age']) - min(combined['Age'])
    combined['Fare'] = combined['Fare']/max(combined['Fare']) - min(combined['Fare'])
    print 'Features scaled successfully !'
scale_all_features()
