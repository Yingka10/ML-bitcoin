import os
import shutil
import pandas as pd

# 設定路徑
csv_path = "labels.csv"  # 包含 filename 和 label 欄位
image_folder = "kline_images"  # 所有圖片都在這裡
output_folder = "KLineImages/train"  # 分類後的資料夾

# 讀取 CSV
df = pd.read_csv(csv_path)

# 數字轉換成文字 label（1 → up, 0 → down）
df["label"] = df["label"].map({1: "up", 0: "down"})

# 建立輸出資料夾
for label in ["up", "down"]:
    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

# 依據 label 分類圖片
for _, row in df.iterrows():
    filename = row["filename"]
    label = row["label"]
    src_path = os.path.join(image_folder, filename)
    dst_path = os.path.join(output_folder, label, filename)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"檔案不存在：{src_path}")
