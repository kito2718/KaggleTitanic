##### 4. Ageの欠損値補完処理(補完はmedian(),年齢のbin化)を追加

import pandas as pd

##### データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv') 
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

##### 家族人数のグループ分け
def family_size_group(size):
    if size == 1:   return 'Alone'
    elif size <= 4: return 'Small'
    else:           return 'Large'

age_median = train_data['Age'].median()
####### One-Hot Encoding(敬称追加+家族人数)処理 ここから
for df in [train_data, test_data]:
    ####### 家族人数列 生成
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    ####### 家族人数をグループ分け
    df['FamilySizeGroup'] = df['FamilySize'].apply(family_size_group)
    ####### 敬称抽出→Title列生成
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    ####### 敬称をまとめる
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Jonkheer', 'Lady', 'Major', 'Sir'],'Rare')
    ####### 年齢の欠損値補完処理 ここから
    df['Age'] = df['Age'].fillna(age_median)
    labels = ['Child', 'Teen', 'Adult', 'Mid', 'Senior']
    bins = [0, 12, 18, 31, 60, 100]
    df['AgeBin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['AgeBin'] = df['AgeBin'].astype(str)  # Ageに100以上 or 0未満があった時の保険
    ####### 年齢の欠損値補完処理 ここまで

##### trainとtestの全データを縦にくっつけて、`all_data`を取得する。
all_data = pd.concat([train_data, test_data], ignore_index=True)

##### One-Hot Encoding
all_data = pd.get_dummies(all_data, columns=['Title', 'FamilySizeGroup', 'AgeBin'])
##### 元に戻す
train_data = all_data.iloc[:len(train_data)].copy()
test_data  = all_data.iloc[len(train_data):].copy()
print(train_data.columns)
####### One-Hot Encoding(敬称追加+家族人数)処理 ここまで

##### 特徴量を選択
features = ["Pclass", "Sex", "Fare"] \
         + [col for col in train_data.columns if "Title_" in col] \
         + [col for col in train_data.columns if "FamilySizeGroup_" in col] \
         + [col for col in train_data.columns if "AgeBin_" in col]

##### 学習データを準備
X = train_data[features].copy()
y = train_data["Survived"]

X_test = test_data[features].copy()

print('Fare_count=',X["Fare"].count(),' Fare is null count=',X["Fare"].isnull().sum())
fare_median = X["Fare"].median()
X["Fare"] = X["Fare"].fillna(fare_median)
X_test["Fare"] = X_test["Fare"].fillna(fare_median)

##### Sex を数値化
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

##### モデル作成
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1)

##### 学習
model.fit(X, y)

##### 予測
predictions = model.predict(X_test)
predictions = predictions.astype(int)

##### 提出ファイル作成
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

##### 提出ファイル出力
output.to_csv("submission-4-3-1.csv", index=False)

##### 完了
print("submission.csv 004-3-1 を作成しました")
