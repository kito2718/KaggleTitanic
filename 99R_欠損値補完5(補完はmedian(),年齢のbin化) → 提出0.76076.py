##### 4. Ageの欠損値補完処理(補完はmedian(),年齢のbin化)を追加

import pandas as pd

##### データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

##### 前準備
all_data = pd.concat([train_data, test_data], ignore_index=True)

##### 家族人数のグループ分け
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

##### 年齢の欠損値補完処理 ここから
all_data['Age'] = all_data['Age'].fillna(train_data['Age'].median())
labels = ['Child', 'Teen', 'Adult', 'Mid', 'Senior']
bins = [0, 12, 18, 31, 60, 100]
all_data['AgeBin'] = pd.cut(all_data['Age'], bins=bins, labels=labels, right=False)
all_data['AgeBin'] = all_data['AgeBin'].astype(str)  # Ageに100以上 or 0未満があった時の保険
##### 年齢の欠損値補完処理 ここまで

####### 本番モデル用のOne-Hot Encoding
all_data = pd.get_dummies(all_data, columns=['Title', 'FamilySizeGroup', 'AgeBin'])

##### 元に戻す
train_data = all_data.iloc[:len(train_data)].copy()
test_data  = all_data.iloc[len(train_data):].copy()

##### 特徴量を選択
features = ["Pclass", "Sex", "Fare"] \
         + [col for col in train_data.columns if "Title_" in col] \
         + [col for col in train_data.columns if "FamilySizeGroup_" in col] \
         + [col for col in train_data.columns if "AgeBin_" in col]

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
output.to_csv("submission-99R-4-3-1.csv", index=False)

##### 完了
print("submission.csv 99R-4-3-1 を作成しました")
