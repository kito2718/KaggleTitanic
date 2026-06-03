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

##### 特徴量エンジニアリング(Fareの欠損値補完)
# 欠損値を Embarked='S', Pclass=3 の平均値で補完
fare=all_data.loc[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3), 'Fare'].median()
all_data['Fare']=all_data['Fare'].fillna(fare)

##################### AgeをRandomForestRegressorで推定 ここから
##### 推定に使用する項目を指定
age_pred_data = all_data[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']]

##### ラベル特徴量をOne-Hotエンコーディング
age_pred_data = pd.get_dummies(age_pred_data)

##### Ageがわかっているデータとわかってないデータに分離し、numpyに変換
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


#################### Surname ここから
##### NameからSurname(苗字)を抽出
all_data['Surname'] = all_data['Name'].map(lambda name:name.split(',')[0].strip())

##### 同じSurname(苗字)の出現頻度をカウント(出現回数が2以上なら家族)
all_data['FamilyGroup'] = all_data['Surname'].map(all_data['Surname'].value_counts()) 

##### 家族で16才以下または女性の生存率
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=16) | (all_data['Sex']=='female'))]
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
print(Female_Child_Group.value_counts())

##### 家族で16才超えかつ男性の生存率
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>16) & (all_data['Sex']=='male')]
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
print(Male_Adult_List.value_counts())

##### デッドリストとサバイブリストの作成
Dead_list=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
Survived_list=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

##### デッドリストとサバイブリストの表示
print('Dead_list = ', Dead_list)
print('Survived_list = ', Survived_list)

##### デッドリストとサバイブリストをSex, Age, Title に反映させる
all_data.loc[(all_data['Survived'].isnull()) & (all_data['Surname'].apply(lambda x:x in Dead_list    )), ['Sex','Age','Title']] = ['male',28.0,'Mr']
all_data.loc[(all_data['Survived'].isnull()) & (all_data['Surname'].apply(lambda x:x in Survived_list)), ['Sex','Age','Title']] = ['female',5.0,'Mrs']
#################### Surname ここまで

##### ----------- Ticket ----------------
##### 同一Ticketナンバーの人が何人いるかを特徴量として抽出
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].map(Ticket_Count)

##### 生存率で3つにグルーピング
all_data.loc[(all_data['TicketGroup']>=2) & (all_data['TicketGroup']<=4), 'Ticket_label'] = 2
all_data.loc[(all_data['TicketGroup']>=5) & (all_data['TicketGroup']<=8) | (all_data['TicketGroup']==1), 'Ticket_label'] = 1  
all_data.loc[(all_data['TicketGroup']>=11), 'Ticket_label'] = 0

##### ------------- Cabin ----------------
##### Cabinの先頭文字を特徴量とする(欠損値は U )
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Cabin_label']=all_data['Cabin'].str.get(0)

##### ---------- Embarked ---------------
##### 欠損値をSで補完
all_data['Embarked'] = all_data['Embarked'].fillna('S') 

# ------------- 前処理 ---------------
# 推定に使用する項目を指定
all_data = all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilySizeGroup','Cabin_label','Ticket_label']]

####### 本番モデル用のOne-Hot Encoding
all_data = pd.get_dummies(all_data)

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

##### 予測
predictions = pipeline.predict(test_x)

##### 提出ファイル作成
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": predictions.astype(np.int32)
})

##### 提出ファイル出力
submission.to_csv("submission-99L-0.80622_001.csv", index=False)

##### 完了
print("submission-99L-0.80622_001.csv を作成しました")