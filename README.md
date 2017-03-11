# Kaggle titanic

1. Trying out first [Kaggle](https://www.kaggle.com/) dataset exploration.
# References

[feature engineering](http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html) .
[Blending](https://github.com/emanuele/kaggle_pbr/blob/master/blend.py)
[Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
[Ensembling](http://mlwave.com/kaggle-ensembling-guide/)
[Voting](https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline)
[Stacking/Ensembling guide](https://www.kaggle.com/shivendra91/titanic/introduction-to-ensembling-stacking-in-python/editnb)
[Random Forest](https://www.kaggle.com/benhamner/titanic/random-forest-benchmark-r/code)
Lot more of them....

# Problem Overview  https://www.kaggle.com/c/titanic
We need to predict for each testing data whether the person was able to survive or not based on the model learnt from training data.

## VARIABLE DESCRIPTIONS
1. survival        Survival (0 = No; 1 = Yes)
2. pclass          Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
3. name            Name
4. sex             Sex
5. age             Age
6. sibsp           Number of Siblings/Spouses Aboard
7. parch           Number of Parents/Children Aboard
8. ticket          Ticket Number
9. fare            Passenger Fare
10. cabin           Cabin
11. embarked        Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## SPECIAL NOTES
1. Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

2. Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

3. With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

 Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
 Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
 Parent:   Mother or Father of Passenger Aboard Titanic
 Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

4. Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  

5. Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.
