##### 年齢の欠損値補完4(RandomForestRegressorで予測)

import pandas as pd
import numpy as np

##### データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

##### 前準備
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 家族人数のグループ分け
def family_size_group(size):
    if size == 1:   return 'Alone'
    elif size <= 4: return 'Small'
    else:           return 'Large'

##### 特徴量エンジニアリング(家族人数)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['FamilySizeGroup'] = all_data['FamilySize'].apply(family_size_group)

##### 特徴量エンジニアリング(敬称抽出)
all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
all_data['Title'] = all_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Jonkheer', 'Lady', 'Major', 'Sir'],'Rare')

##### 特徴量エンジニアリング(Sexを数値化)
all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})

##### 特徴量エンジニアリング(Fareの欠損値補完(これはTestデータに1件だけなので中央値で。特にこだわりなし))
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

##################### AgeをRandomForestRegressorで推定 ここから
##### 推定に使用する項目を指定
age_pred_data = all_data[['Age', 'Pclass', 'Sex', 'Fare', 'SibSp', 'Parch', 'Title', 'FamilySizeGroup']]

##### ラベル特徴量をOne-Hotエンコーディング
age_pred_data = pd.get_dummies(age_pred_data)

##### Ageがわかっているデータに分離し、numpyに変換
age_known  = age_pred_data[age_pred_data['Age'].notnull()].values
age_unknown= age_pred_data[age_pred_data['Age'].isnull()].values

##### 学習用データをX_age, y_ageに分離
X_age = age_known[:, 1:] # Age以外の特徴量
y_age = age_known[:, 0]  # Age(目的変数)

##### ランダムフォレスト(回帰)で推定モデルを構築
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X_age, y_age)

##### 欠損値のAge予測実行
predicted_ages = rfr.predict(age_unknown[:, 1:])

##### 元のall_dataに補完
all_data.loc[all_data['Age'].isnull(), 'Age'] = predicted_ages
#####################AgeをRandomForestRegressorで推定 ここまで

###### 年齢のbin化(予測後のAgeに対して実行)
#labels = ['Child', 'Teen', 'Adult', 'Mid', 'Senior']
#bins = [0, 12, 18, 31, 60, 100]
#all_data['AgeBin'] = pd.cut(all_data['Age'], bins=bins, labels=labels, right=False).astype(str)

####### 本番モデル用のOne-Hot Encoding
#all_data = pd.get_dummies(all_data, columns=['Title', 'FamilySizeGroup', 'AgeBin'])
all_data = pd.get_dummies(all_data, columns=['Title', 'FamilySizeGroup'])

##### 元に戻す
train_data = all_data.iloc[:len(train_data)].copy()
test_data  = all_data.iloc[len(train_data):].copy()

##### 特徴量を選択
features = ["Pclass", "Sex", "Fare", 'Age'] \
         + [col for col in train_data.columns if "Title_" in col] \
         + [col for col in train_data.columns if "FamilySizeGroup_" in col]# \
        #+ [col for col in train_data.columns if "AgeBin_" in col]

##### 学習データを準備
X      = train_data[features]
y      = train_data["Survived"]
X_test = test_data[features]

##### モデル作成・学習
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1)
model.fit(X, y)

##### 予測
predictions = model.predict(X_test).astype(int)

##### 提出ファイル作成
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

##### 提出ファイル出力
output.to_csv("submission-99R-001.csv", index=False)

##### 完了
print("submission-99R-001.csv を作成しました")
