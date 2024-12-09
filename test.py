import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def feature_engineering(df):
    df['trans_date'] = pd.to_datetime(df['trans_date']).astype(int) / 10**9
    df['hour'] = pd.to_datetime(df['trans_time']).dt.hour
    df['minute'] = pd.to_datetime(df['trans_time']).dt.minute
    df['second'] = pd.to_datetime(df['trans_time']).dt.second
    df['age'] = pd.to_datetime('today').year - pd.to_datetime(df['dob']).dt.year
    
    df.drop(columns=['trans_time', 'dob'], inplace=True, errors='ignore')
    return df

train = feature_engineering(train)
test = feature_engineering(test)

X_train = train.drop(columns=['is_fraud', 'id', 'trans_num'])
y_train = train['is_fraud']
X_test = test.drop(columns=['id', 'trans_num'])

X_test = X_test[X_train.columns]

object_cols = X_train.select_dtypes(include=['object']).columns
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
    le.classes_ = np.append(le.classes_, 'Unknown')  
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'scale_pos_weight': 2, 
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}...")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / skf.n_splits

oof_preds_binary = (oof_preds > 0.5).astype(int)
f1 = f1_score(y_train, oof_preds_binary)
print(f"Out-of-Fold F1 Score: {f1:.4f}")

submission = pd.DataFrame({
    'id': test['id'],
    'is_fraud': (test_preds > 0.5).astype(int)
})

submission.to_csv('submission.csv', index=False)
print("优化后的预测结果已保存为 submission.csv")