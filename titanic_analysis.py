import pandas as pd
from seaborn import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 数据加载
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

train_data.Age = train_data.Age.fillna(train_data.Age.mean())
test_data.Age = test_data.Age.fillna(test_data.Age.mean())

train_data.Embarked = train_data.Embarked.fillna('S')

train_data.Cabin = train_data.Cabin.fillna('U')
test_data.Cabin = test_data.Cabin.fillna('U')

test_data['Fare'] = test_data['Fare'].fillna(test_data.Fare.mean())

test_data['Title'] = test_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
train_data['Title'] = train_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))

test_data['Title'] = test_data['Title'].map(Title_Dict)
train_data['Title'] = train_data['Title'].map(Title_Dict)

test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2
test_data['Embarked'] = test_data['Embarked'].astype('float')

train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2
train_data['Embarked'] = train_data['Embarked'].astype('float')

test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1
test_data['Sex'] = test_data['Sex'].astype('float')

train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
train_data['Sex'] = train_data['Sex'].astype('float')

title_mapping = {
    'Mr': 1,
    'Miss': 2,
    'Mrs': 3,
    'Master': 4,
    'Officer': 5,
    'Royalty': 6

}
for k, v in title_mapping.items():
    test_data['Title'][test_data['Title'] == k] = v
test_data['Title'] = test_data['Title'].astype('float')

for k, v in title_mapping.items():
    train_data['Title'][train_data['Title'] == k] = v
train_data['Title'] = train_data['Title'].astype('float')

all_variables = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                     'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title']
predictors = ['Pclass', 'Sex', 'Parch', 'Fare', 'Embarked', 'Title']

trainlabel =train_data['Survived']
traindata, testdata = train_data[predictors], test_data[predictors]

trainSet, testSet, trainlabel, testlabel = train_test_split(traindata, trainlabel,
                                                            test_size=0.2, random_state=12345)

clf = LogisticRegression()
clf.fit(trainSet, trainlabel)
test_accuracy = clf.score(testSet, testlabel) * 100
print("正确率为   %s%%" % test_accuracy)

res = clf.predict(testdata)
test_data['Survived']=res;

test_data[['PassengerId','Survived']].to_csv(r"D:\gender_submission.csv",index=False)






