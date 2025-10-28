# src/01_output_csv.py
"""
Raw → CSV Converter
======================================

使用資料：2025/05/06
Python：3.12.2（Spyder）

目的：
- 將台北市停車場原始 JSON 轉為 CSV 格式，供後續清理與整併。

輸入：
- data/raw/TCMSV_alldesc-2.json

輸出：
- data/raw/raw_taipei_paring_lot_Info.csv  # 原始資料轉 CSV

主要步驟：
1) 載入 JSON
2) 提取欄位並整理為 DataFrame
3) 存成 CSV（UTF-8、含欄名）

建議執行：
- 直接在 Spyder 執行，或於終端機：python src/01_output_csv.py
"""

import json
import pandas as pd

with open("data/raw/TCMSV_alldesc-2.json", "r", encoding="utf-8") as file:
    raw1_json = json.load(file)
data = raw1_json["data"]["park"]  # 這一層是 list of dict
data = pd.DataFrame(data)
data.info()
data.to_csv("data/raw/raw_taipei_paring_lot_Info.csv", index=False, encoding="utf_8_sig")



         