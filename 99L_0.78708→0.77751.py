import pandas as pd
import numpy as np

##### データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

##### 前準備
all_data = pd.concat([train_data, test_data], ignore_index=True)

##### 特徴量エンジニアリング(家族人数)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data.loc[(all_data['FamilySize']>=2) & (all_data['FamilySize']<=4), 'FamilySizeGroup'] = 2
all_data.loc[(all_data['FamilySize']>=5) & (all_data['FamilySize']<=7) | (all_data['FamilySize']==1), 'FamilySizeGroup'] = 1  # == に注意
all_data.loc[(all_data['FamilySize']>=8), 'FamilySizeGroup'] = 0

##### 特徴量エンジニアリング(敬称抽出)
all_data['Title'] = all_data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
all_data['Title'].replace(['Mlle'], 'Miss', inplace=True)
all_data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
all_data['Title'].replace(['Jonkheer'], 'Master', inplace=True)
all_data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
all_data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)

##### 特徴量エンジニアリング(Fareの欠損値補完(これはTestデータに1件だけなので中央値で。特にこだわりなし))
# 欠損値を Embarked='S', Pclass=3 の平均値で補完
fare=all_data.loc[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3), 'Fare'].median()
all_data['Fare']=all_data['Fare'].fillna(fare)

##################### AgeをRandomForestRegressorで推定 ここから
##### 推定に使用する項目を指定
age_pred_data = all_data[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']]

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

####### 本番モデル用のOne-Hot Encoding
all_data = pd.get_dummies(all_data, columns=['Sex', 'Embarked', 'Title'])

##### 元に戻す
train_data = all_data[all_data['Survived'].notnull()]
test_data  = all_data[all_data['Survived'].isnull()]

##### 特徴量を選択
features = ["Pclass", 'Age', 'Fare', 'FamilySizeGroup', ] \
         + [col for col in train_data.columns if "Sex_" in col] \
         + [col for col in train_data.columns if "Embarked_" in col] \
         + [col for col in train_data.columns if "Title_" in col]

##### 学習データを準備
X      = train_data[features]
y      = train_data["Survived"]
test_x = test_data[features]

##### モデル作成・学習
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

# 採用する特徴量を25個から20個に絞り込む
from sklearn.feature_selection import SelectKBest
select = SelectKBest(k = 20)

clf = RandomForestClassifier(random_state = 10, 
                             warm_start = True,  # 既にフィットしたモデルに学習を追加 
                             n_estimators = 26,
                             max_depth = 6, 
                             max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

# フィット結果の表示
cv_result = cross_validate(pipeline, X, y, cv= 10)
print('mean_score = ', np.mean(cv_result['test_score']))
print('mean_std = ', np.std(cv_result['test_score']))

# --------　採用した特徴量 ---------------
# 採用の可否状況
mask= select.get_support()

# 項目のリスト
list_col = list(all_data[features].columns)

# 項目別の採用可否の一覧表
for i, j in enumerate(list_col):
    print('No'+str(i+1), j,'=',  mask[i])

# シェイプの確認
X_selected = select.transform(X)
print('X.shape={}, X_selected.shape={}'.format(X.shape, X_selected.shape))

##### 予測
predictions = pipeline.predict(test_x)

##### 提出ファイル作成
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": predictions.astype(np.int32)
})

##### 提出ファイル出力
submission.to_csv("submission-99L-008.csv", index=False)

##### 完了
print("submission-99L-008.csv を作成しました")
