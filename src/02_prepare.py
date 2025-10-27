#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taipei Parking Lot Data Preprocessing Pipeline
==============================================

目的：
- 將台北市停車場原始資料（raw CSV + 手工拆分之費用表）清理並整併，
- 統一型態與類別編碼，
- 依收費型態進行遺失值規則化填補 + 中位數補值，
- 產出乾淨資料集供後續 ML / Data Mining 使用。

輸入：
- data/raw/raw_taipei_paring_lot_Info.csv
- data/processed/preprocessing2_fare.csv

輸出：
- preprocessing3_taipei_paring_lot_availble.csv（最終乾淨資料）

步驟摘要：
1) 載入資料
2) 刪除無用/不完整/重複性高之欄位
3) 清理 fare 表（欄名、型態）
4) 合併主表與 fare
5) 類別欄位標準化（type2 → 公/民營、serviceTime → 24hr）
6) 篩除無目標/僅大客車場站（專案不討論）
7) 依收費型態分類 + 規則化填補 (-1)
8) 其餘遺失值以中位數補值
9) 輸出 + 摘要檢查

作者：徐胤瑄
"""
import pandas as pd
parking = pd.read_csv("data/raw/raw_taipei_paring_lot_Info.csv")
#parking.info()

# 處理欄位
## 刪除parking無需使用之欄位
'''
    name: 停車場名稱
    payex: 停車場收費方式
    summary: 停車場停車格詳細資訊（與後面重複）
    address: 停車場地址
    tel: 停車場電話
    tw97x, tw97y: 經緯度
    totalbus: 總公車停車位數（較少停車場提供公車停車位，因此該專案無討論）
    Taxi_OneHR_Free: 是否有計程車一小時免費停車服務（全部為0）
    AED_Equipment: 是否有AED設備（全部為0）
    CellSignal_Enhancement: 是否有手機訊號增強設備（全部為0）
    Accessibility_Elevator: 是否有無障礙電梯（全部為0）
    Phone_Charge: 是否有手機充電服務（全部為0）
    Child_Pickup_Area: 是否有兒童接送區域（全部為0）
    FareInfo: 收費資訊（不完整，該專案使用payex欄位）
    EntranceCoord: 入口座標（不完整，該欄位含經緯度與地址）
    ptype: 停車場類型（不完整，且資料出處無說明該欄位用途）
'''
parking.drop(["name", "payex", "summary", "address", "tel", "tw97x", "tw97y", "totalbus", "Taxi_OneHR_Free", "AED_Equipment", "CellSignal_Enhancement", "Accessibility_Elevator", "Phone_Charge", "Child_Pickup_Area", "FareInfo", "EntranceCoord", "ptype"], axis=1, inplace=True)
parking.info()
#parking.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing1_taipei_paring_lot_Info.csv", index=False, encoding="utf_8_sig")


## 處理fare
'''
    fare 是手動分類停車場的費用資訊，從"payex"拆分出來的，分類為：
    - farecar_weekday: 小客車平日費用
    - farecar_night: 小客車夜間費用
    - farecar_haliday: 小客車假日費用
    - farecar_month: 小客車月租費用
    - faremotor_day/hour: 重型機車日/時費用
    - faremotor_month: 重型機車月租費用
    - largemotor_month: 大型機車月租費用
'''
fare = pd.read_csv("data/processed/preprocessing2_fare.csv")
fare.drop(["Unnamed: 0"], axis=1, inplace=True)
fare.columns = ["farecar_weekday", "farecar_night", "farecar_haliday", "farecar_month", "faremotor_day/hour", "faremotor_month", "largemotor_month"]
fare.info()
fare["farecar_weekday"] = (
    fare["farecar_weekday"]
    .astype(str)  # 先保證是字串
    .str.replace(",", "", regex=False)  
    .str.extract(r"(\d+\.?\d*)")[0]     
    .astype(float)                      
)

## 合併：篩選後欄位與fare
parking_new = pd.concat([parking, fare], axis=1)
parking_new.info()


# 轉型態為簡單的分類
## 處理目標變數type2
'''
    type2可分為公營與民營兩大類：
        公營又可分為本處自營、本處委外、市屬機關學校自營、市屬機關學校委外。
        民營則為民營停車場。
'''
print(parking_new["type2"].value_counts(dropna=False))
mapping = {
    "民營停車場": 2,
    "本處委外停車場": 1,
    "本處自營停車場": 1,
    "市屬機關學校委外": 1,
    "市屬機關學校自營": 1
}
parking_new["type2"] = parking_new["type2"].map(mapping)

## 處理serviceTime
print(parking_new["serviceTime"].value_counts(dropna=False))
parking_new["serviceTime"] = parking_new["serviceTime"].apply(
    lambda x: 1 if x == "24小時" else 0
)


# 處理遺失值資料
## 刪除沒有目標變數的資料
parking_new = parking_new.dropna(subset=["type2"])

## 刪除僅提供給大客車的停車場
mask = parking_new[["farecar_weekday", "farecar_month", "faremotor_day/hour", "faremotor_month"]].isna().all(axis=1)
print(parking_new[mask])  # 顯示符合條件的 row
parking_new_mask = pd.DataFrame(parking_new[mask])
parking_new = parking_new[~mask]

# 將停車場收費方式分類為不同類型
'''
    停車場收費方式分類為：
    - monthly_rent: 僅提供月租
    - car_and_largemotor: 僅提供小客車或重型機車
    - motor: 僅提供機車
    - mixed_parking_lot: 混合式停車場（提供多種收費方式）
'''
## 建立分類函數
def classify_parking_type(row):
    # 條件一：僅提供月租
    if pd.isna(row["farecar_weekday"]) and pd.isna(row["farecar_night"]) and pd.isna(row["farecar_haliday"]) and pd.isna(row["faremotor_day/hour"]):
        return "monthly_rent"
    # 條件二：僅提供小客車或重型機車
    elif pd.isna(row["faremotor_day/hour"]) and pd.isna(row["faremotor_month"]):
        return "car_and_largemotor"
    # 條件三：僅提供機車
    elif pd.isna(row["farecar_weekday"]) and pd.isna(row["farecar_night"]) and pd.isna(row["farecar_haliday"]) and pd.isna(row["farecar_month"]) and pd.isna(row["largemotor_month"]):
        return "motor"
    # 其他為混合式
    else:
        return "mixed_parking_lot"

## 套用分類函數
parking_new["parking_fare_classification"] = parking_new.apply(classify_parking_type, axis=1)
print(parking_new["parking_fare_classification"].value_counts(dropna=False))

# 依據停車場收費方式處理遺失值
## 定義條件對應的規定遺失值欄位
fixed_fill_rules = {
    "monthly_rent": ["farecar_weekday", "farecar_night", "farecar_haliday", "faremotor_day/hour"],
    "car_and_largemotor": ["faremotor_day/hour", "faremotor_month"],
    "motor": ["farecar_weekday", "farecar_night", "farecar_haliday", "farecar_month", "largemotor_month"],
    "mixed_parking_lot": []
}
## 若不屬於該資料被分類到的欄位，補上 -1
parking_new2 = parking_new.copy()
for group, cols_to_fill in fixed_fill_rules.items():
    mask = parking_new2["parking_fare_classification"] == group
    for col in cols_to_fill:
        parking_new2.loc[mask & parking_new2[col].isna(), col] = -1
#parking_new2.info()

## 個別處理剩下的遺失值（處理剩下屬於該分類應有的欄位，但還為遺失值的狀態，因此統一取中位數）
import numpy as np
parking_new2["farecar_weekday"]=np.where(parking_new2["farecar_weekday"].isnull(), np.nanmedian(parking_new2["farecar_weekday"]), parking_new2["farecar_weekday"])
parking_new2["farecar_night"]=np.where(parking_new2["farecar_night"].isnull(), np.nanmedian(parking_new2["farecar_night"]), parking_new2["farecar_night"])
parking_new2["farecar_haliday"]=np.where(parking_new2["farecar_haliday"].isnull(), np.nanmedian(parking_new2["farecar_haliday"]), parking_new2["farecar_haliday"])
parking_new2["farecar_month"]=np.where(parking_new2["farecar_month"].isnull(), np.nanmedian(parking_new2["farecar_month"]), parking_new2["farecar_month"])
parking_new2["faremotor_day/hour"]=np.where(parking_new2["faremotor_day/hour"].isnull(), np.nanmedian(parking_new2["faremotor_day/hour"]), parking_new2["faremotor_day/hour"])
parking_new2["faremotor_month"]=np.where(parking_new2["faremotor_month"].isnull(), np.nanmedian(parking_new2["faremotor_month"]), parking_new2["faremotor_month"])
parking_new2["largemotor_month"]=np.where(parking_new2["largemotor_month"].isnull(), np.nanmedian(parking_new2["largemotor_month"]), parking_new2["largemotor_month"])


# 輸出
parking_new2.info()
parking_new2.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing3_taipei_paring_lot_availble.csv", index=False, encoding="utf_8_sig")
