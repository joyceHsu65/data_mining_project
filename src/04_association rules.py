# src/04_association rules.py
"""
Association Rule Mining (Apriori)
=================================

目的：
- 以 Apriori 探勘停車場設施與收費特徵間的關聯。
- 篩選與「經營型態（type2）」有顯著關聯的規則。

輸入：
- data/processed/preprocessing3_taipei_paring_lot_availble.csv

輸出：
- results/04_rules.csv（通過篩選的關聯規則表）

主要步驟：
1) 載入處理後資料並刪除無關欄位
2) 特徵二元化以符合 Apriori 格式
3) 執行 Apriori（min_support=0.05, max_len=4）
4) 根據 lift > 1.1 篩選規則
5) 僅保留後件為「type2=2.0」之結果並輸出 CSV

建議執行：
- python "src/04_association rules.py"
"""

import pandas as pd
parking = pd.read_csv("data/processed/preprocessing3_taipei_paring_lot_availble.csv")

# 資料預處理
parking.drop(["area", "serviceTime", "parking_fare_classification"], axis=1, inplace=True)
parking.set_index("id", inplace=True)
parking.info()

parking["type"] = parking["type"].apply(
    lambda x: 0 if x == 1 else 1
)

#對設施欄位應用 parking_space()
def parking_space(x):
    if x==0:
        return 0
    else:
        return 1
facility_cols = [
    'totalcar', 'totalmotor', 'totalbike',
    'Pregnancy_First', 'Handicap_First',
    'totallargemotor', 'ChargingStation'
]
for col in facility_cols:
    parking[col] = parking[col].apply(parking_space)

#對收費欄位應用 parking_fare()
def parking_fare(x):
    if x==-1:
        return 0
    else:
        return 1
fare_cols = [
    'farecar_weekday', 'farecar_night', 'farecar_haliday',
    'farecar_month', 'faremotor_day/hour',
    'faremotor_month', 'largemotor_month'
]
for col in fare_cols:
    parking[col] = parking[col].apply(parking_fare)

'''
parking['type2'] = parking['type2'].map({1: "TYPE2=1停管處", 2: "TYPE2=2非停管處"})

# 將目標變數轉為一個類型一欄
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
type2 = ohe.fit_transform(parking[["type2"]])
type2 = pd.DataFrame(type2)
type2.columns = ["TYPE2=1停管處", "TYPE2=2非停管處"]

X1 = parking.drop(["type2"], axis=1)
basket = pd.concat([X1, type2], axis=1)
'''

#=======================================================
# 重抽樣
print(parking['type2'].value_counts(normalize=True))
    #2.0    0.733098, 1.0    0.266902

# 將少的取出放回方式，抽樣成與多的一樣 oversampling欠抽樣
from imblearn.over_sampling import RandomOverSampler
X = parking.drop(["type2"], axis=1)
y = parking['type2']
ros = RandomOverSampler(random_state=20250610)
X_resample, y_resample = ros.fit_resample(X, y)
print(y_resample.value_counts())

# 建立重抽樣後的新 DataFrame
basket = X_resample.copy()
basket['type2'] = y_resample


#=======================================================
# 轉為關聯法則之資料
def to_items(row):
    return [f"{col}={row[col]}" for col in row.index]

transactions = basket.apply(to_items, axis=1).tolist()

# 使用 TransactionEncoder 將交易格式轉成 One-hot 矩陣
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_tran = pd.DataFrame(te_array, columns=te.columns_)



#============================================
# 關聯法則
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_itemset = apriori(basket_tran, min_support=0.05, use_colnames=True)
frequent_itemset = frequent_itemset[frequent_itemset['itemsets'].apply(len) <= 4]
print(frequent_itemset.shape)
rules = association_rules(frequent_itemset, metric="lift", min_threshold=1.1)

rules = rules[rules['consequents'].apply(lambda x: 'type2=2.0' in x)]

rules.to_csv("results/04_rules.csv", index=False, encoding="utf_8_sig")

