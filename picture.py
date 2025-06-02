import os
import pandas as pd
import mplfinance as mpf

# 載入資料
df = pd.read_csv("bitcoin_fear_greed_2018_2024.csv")

# 確保資料有日期索引
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 創建圖片資料夾
img_dir = "kline_images"
os.makedirs(img_dir, exist_ok=True)

# 滑動視窗大小
window_size = 5

# 儲存標籤資料
label_data = []

# 產生圖片與標籤
for i in range(window_size, len(df) - 1):
    window_df = df.iloc[i - window_size:i]

    # 收盤價變化決定 label
    five_days_ago_open = df.iloc[i-window_size]['Open']
    next_day_close = df.iloc[i + 1]['Close']
    if next_day_close > five_days_ago_open:
        label = 1 # 明天收盤價上漲
    else:
        label = 0

    # 使用最後一天日期當作檔名
    date_str = df.index[i].strftime("%Y-%m-%d")
    filename = f"{date_str}.png"
    filepath = os.path.join(img_dir, filename)

    # 畫 K 線圖
    mpf.plot(
        window_df,
        type='candle',
        style='charles',
        savefig=filepath,
        axisoff=True,
        show_nontrading=True
    )

    # 存入 label
    label_data.append([filename, label])

# 儲存 label CSV
label_df = pd.DataFrame(label_data, columns=["filename", "label"])
label_df.to_csv("labels.csv", index=False)


