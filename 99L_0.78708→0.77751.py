import pandas as pd
import numpy as np

##### データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data2 = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

##### 前準備
all_data = pd.concat([train_data, test_data2], ignore_index=True)

##### 特徴量エンジニアリング(家族人数)
# Family = SibSp + Parch + 1 を特徴量とし、グルーピング
all_data['Family']=all_data['SibSp']+all_data['Parch']+1
all_data.loc[(all_data['Family']>=2) & (all_data['Family']<=4), 'Family_label'] = 2
all_data.loc[(all_data['Family']>=5) & (all_data['Family']<=7) | (all_data['Family']==1), 'Family_label'] = 1  # == に注意
all_data.loc[(all_data['Family']>=8), 'Family_label'] = 0


##### 特徴量エンジニアリング(敬称抽出)
all_data['Title'] = all_data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
all_data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
all_data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
all_data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
all_data['Title'].replace(['Mlle'], 'Miss', inplace=True)
all_data['Title'].replace(['Jonkheer'], 'Master', inplace=True)


##### 特徴量エンジニアリング(Fareの欠損値補完(これはTestデータに1件だけなので中央値で。特にこだわりなし))
# 欠損値を Embarked='S', Pclass=3 の平均値で補完
fare=all_data.loc[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3), 'Fare'].median()
all_data['Fare']=all_data['Fare'].fillna(fare)

##################### AgeをRandomForestRegressorで推定 ここから
# Age を Pclass, Sex, Parch, SibSp からランダムフォレストで推定
from sklearn.ensemble import RandomForestRegressor

# 推定に使用する項目を指定
age_df = all_data[['Age', 'Pclass','Sex','Parch','SibSp']]

# ラベル特徴量をワンホットエンコーディング
age_df=pd.get_dummies(age_df)

# 学習データとテストデータに分離し、numpyに変換
known_age = age_df[age_df.Age.notnull()].values  
unknown_age = age_df[age_df.Age.isnull()].values

# 学習データをX, yに分離
X = known_age[:, 1:]  
y = known_age[:, 0]

# ランダムフォレストで推定モデルを構築
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)

# 推定モデルを使って、テストデータのAgeを予測し、補完
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAges 
#####################AgeをRandomForestRegressorで推定 ここまで

# ------------- 前処理 ---------------
# 推定に使用する項目を指定
df = all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','Family_label']]

####### 本番モデル用のOne-Hot Encoding
all_data = pd.get_dummies(df)

##### 元に戻す
train_data = all_data[all_data['Survived'].notnull()]
test_data  = all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)

##### 学習データを準備
X      = train_data.values[:,1:]  
y      = train_data.values[:,0].astype(int)
test_x = test_data.values

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
list_col = list(all_data.columns[1:])

# 項目別の採用可否の一覧表
for i, j in enumerate(list_col):
    print('No'+str(i+1), j,'=',  mask[i])

# シェイプの確認
X_selected = select.transform(X)
print('X.shape={}, X_selected.shape={}'.format(X.shape, X_selected.shape))

PassengerId=test_data2['PassengerId']

##### 予測
predictions = pipeline.predict(test_x)

##### 提出ファイル作成
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": predictions.astype(np.int32)
})

##### 提出ファイル出力
submission.to_csv("submission-99L-004.csv", index=False)

##### 完了
print("submission-99L-004.csv を作成しました")
