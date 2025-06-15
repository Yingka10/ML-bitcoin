# 匯入必要模組
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import requests
import json
from datetime import datetime, timezone
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from configparser import ConfigParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pycoingecko import CoinGeckoAPI
import mplfinance as mpf
import io
import base64
import tempfile
from statsmodels.tsa.arima.model import ARIMA

# 初始化 Flask 應用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 讀取 config
config = ConfigParser()
config.read("config.ini")

# 初始化 LLM
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=config["Gemini"]["API_KEY"]
)
# 初始化 CoinGecko API
cg = CoinGeckoAPI()
# 載入模型
keras_model = load_model("kline_forecast.keras")
sklearn_model = joblib.load("fear_greed_regression_model.pkl")

def get_fear_greed_data(days=5, end_date=None):
    """獲取指定日期前的 Fear & Greed Index 數據"""
    try:
        # 請求更多天數以確保有足夠數據
        url = f"https://api.alternative.me/fng/?limit={days+10}&format=json"
        response = requests.get(url)
        data = response.json()
        
        if not data.get('data'):
            raise ValueError("無法獲取 Fear & Greed Index 數據")
        
        # 轉換所有數據為日期格式
        fgi_data = []
        for item in data['data']:
            date = datetime.fromtimestamp(int(item['timestamp']), tz=timezone.utc)
            fgi_data.append({
                'date': date,
                'value': int(item['value']),
                'classification': item['value_classification']
            })
        
        # 根據指定日期篩選數據
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            fgi_data = [
                item for item in fgi_data 
                if item['date'].date() <= end_dt.date()
            ]
        
        # 取最近的指定天數
        return sorted(fgi_data, key=lambda x: x['date'], reverse=True)[:days]
        
    except Exception as e:
        print(f"獲取 Fear & Greed Index 數據時發生錯誤: {e}")
        return None

def get_bitcoin_price_data(days=5, end_date=None):
    """獲取指定日期前的比特幣價格數據"""
    try:
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            # 計算起始時間（多獲取幾天確保數據足夠）
            start_timestamp = int((end_dt.timestamp() - (days + 5) * 86400))
            end_timestamp = int(end_dt.timestamp() + 86399)
            
            print(f"獲取數據範圍: {datetime.fromtimestamp(start_timestamp)} 到 {datetime.fromtimestamp(end_timestamp)}")
            
            data = cg.get_coin_market_chart_range_by_id(
                id='bitcoin',
                vs_currency='usd',
                from_timestamp=start_timestamp,
                to_timestamp=end_timestamp
            )
        else:
            data = cg.get_coin_market_chart_by_id('bitcoin', 'usd', days=days+2)

        if not data or 'prices' not in data:
            raise ValueError("無法獲取價格數據")

        # 處理價格數據
        prices = {}
        for timestamp_ms, price in data['prices']:
            date = datetime.fromtimestamp(timestamp_ms/1000, tz=timezone.utc)
            prices[date.date()] = price

        # 根據日期篩選並排序
        sorted_dates = sorted(prices.keys(), reverse=True)
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            sorted_dates = [d for d in sorted_dates if d <= end_date]

        # 取指定天數的數據
        selected_dates = sorted_dates[:days]
        
        if len(selected_dates) < days:
            raise ValueError(f"數據不足，只找到 {len(selected_dates)} 天的數據")

        return [{
            'date': datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc),
            'price': round(prices[date], 2)
        } for date in selected_dates]

    except Exception as e:
        print(f"獲取比特幣價格數據時發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def get_bitcoin_5day_ohlc_chart(end_date=None):
    """獲取指定日期前的K線圖數據"""
    try:
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            # 多獲取幾天確保數據足夠
            start_timestamp = int(end_dt.timestamp() - 8 * 86400)  # 獲取8天數據
            end_timestamp = int(end_dt.timestamp()+ 86399)
            
            print(f"獲取OHLC數據範圍: {datetime.fromtimestamp(start_timestamp)} 到 {datetime.fromtimestamp(end_timestamp)}")
            
            # 使用 market chart range API
            data = cg.get_coin_market_chart_range_by_id(
                id='bitcoin',
                vs_currency='usd',
                from_timestamp=start_timestamp,
                to_timestamp=end_timestamp
            )
            
            if not data or 'prices' not in data:
                raise ValueError("無法獲取價格數據")

            # 創建時間序列數據
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True)
            prices_df.set_index('timestamp', inplace=True)

            # 重新取樣為小時頻率
            hourly_df = prices_df['price'].resample('1H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()

            # 轉換為日頻率
            df = hourly_df.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
            
        else:
            ohlc = cg.get_coin_ohlc_by_id(
                id='bitcoin',
                vs_currency='usd',
                days='7'
            )
            
            if not ohlc:
                raise ValueError("無法獲取 OHLC 數據")

            df = pd.DataFrame(ohlc, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

        print(f"原始數據點數: {len(df)}")
        print(f"原始數據範圍: {df.index[0]} 到 {df.index[-1]}")

        # 取最後5天的數據
        daily_df = df.tail(5)

        print(f"最終K線數量: {len(daily_df)}")
        print(f"最終日期範圍: {daily_df.index[0]} 到 {daily_df.index[-1]}")

        if len(daily_df) < 5:
            raise ValueError(f"數據不足，只有 {len(daily_df)} 條K線")

        # 生成圖表
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mpf.plot(
            daily_df,
            type='candle',
            style='charles',
            ax=ax,
            volume=False,
            show_nontrading=True
        )

        # 保存圖表
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format='png',
            bbox_inches='tight',
            dpi=150,
            facecolor='black'
        )
        plt.close(fig)
        
        return base64.b64encode(buf.getvalue()).decode(), daily_df

    except Exception as e:
        print(f"獲取K線圖數據時發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None
        

def predict_next_price(fgi_values, price_values):
    if len(fgi_values) != 5 or len(price_values) != 5:
        raise ValueError("請提供5天的 fgi 和 price 資料")
    columns = list(sklearn_model.feature_names_in_)
    # 依照模型訓練時的順序組裝資料
    data = {}
    for col in columns:
        if col.startswith('fgi_lag'):
            idx = int(col.replace('fgi_lag', '')) - 1
            data[col] = fgi_values[idx]
        elif col.startswith('Close_lag'):
            idx = int(col.replace('Close_lag', '')) - 1
            data[col] = price_values[idx]
    input_data = pd.DataFrame([data], columns=columns)
    prediction = sklearn_model.predict(input_data)
    return prediction[0],

def predict_next_price_arima(fgi_values, price_values, order=(1,1,0)):
    """使用 ARIMA 模型預測下一天價格"""
    if len(fgi_values) != 5 or len(price_values) != 5:
        raise ValueError("請提供5天的 fgi 和 price 資料")

    # exog要是2D array，shape=(5,1)
    endog_train = np.array(price_values[::-1])
    exog_train = np.array(fgi_values[::-1]).reshape(-1, 1)
    
    # 建立並擬合ARIMA模型
    model = ARIMA(endog=endog_train, exog=exog_train, order=order)
    model_fit = model.fit()
    
    # 計算模型評估指標
    aic = model_fit.aic
    bic = model_fit.bic
    mse = np.mean((endog_train - model_fit.fittedvalues) ** 2)
    rmse = np.sqrt(mse)
    
    # 用今天的FGI預測下一天價格
    next_exog = np.array([fgi_values[-1]]).reshape(1, -1)
    pred = model_fit.forecast(steps=1, exog=next_exog)[0]
    
    return {
        'prediction': pred,
        'metrics': {
            'AIC': aic,
            'BIC': bic,
            'RMSE': rmse,
            'order': order
        }
    }

def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(224,224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.
    prediction = keras_model.predict(img_tensor)
    return int(prediction[0][0] > 0.5)

def generate_llm_analysis(fgi_values, prices, image_class, image_prob, regression_price, arima_results):
    """生成LLM綜合分析"""
    regression_r2 = 0.9921031529442089 
    regression_mse = 2182245.496926982
    
    user_input = f"""
    請根據以下市場數據進行整合分析：

    1. 原始數據序列 (從最新到最舊):
        過去5天 FGI 指數: {[fgi_values[0], fgi_values[1], fgi_values[2], fgi_values[3], fgi_values[4]]}
        過去5天收盤價 (USD): {[prices[0], prices[1], prices[2], prices[3], prices[4]]}
        
    2. 預測模型結果：
        回歸模型預測價格：${regression_price:.2f}
        回歸模型信心指數(R² Score)：{regression_r2:.2f} 
        回歸模型評估 (MSE)：{regression_mse:.2f}
        ARIMA模型預測價格：${arima_results['prediction']:.2f}
        ARIMA模型評估：
        - AIC: {arima_results['metrics']['AIC']:.2f}
        - BIC: {arima_results['metrics']['BIC']:.2f}
        - RMSE: {arima_results['metrics']['RMSE']:.2f}
            
    3.  圖形趨勢預測結果：
        K線趨勢判斷：{'上漲' if image_class == 1 else '下跌'}
        圖形判斷信心度：{image_prob:.2%}

    請進行綜合分析：
    
    1.  市場現況總結： 結合FGI和價格序列的變化，描述當前的市場氣氛和價格動能。
    2.  多個模型預測價格的差異分析：比較三種模型（回歸、ARIMA、Keras）的預測結果
    3.  風險等級評估：從「極低、低、中等、高、極高」中選擇一個風險等級，並簡要說明理由
    4.  具體操作建議（買入/持有/觀望/賣出）：基於你的綜合分析，提出明確的操作建議

    請以數字編號條列分析重點，最後要有總結分析。
    """

    role_description = """
    你是市場趨勢分析師，根據 Fear & Greed Index、價格預測、圖像判斷與模型預測結果，請輸出合理且簡潔的結論。
    請不要使用任何 Markdown 語法（如 *、-、#、** 等），請用純文字條列理由，每一點之間需換行。
    """

    messages = [
        ("system", role_description),
        ("human", user_input),
    ]

    response = llm_gemini.invoke(messages)
    # 返回所有內容，以便建立對話歷史
    return response.content, user_input, role_description

@app.route('/api/combined-data/5')
def api_combined_data():
    end_date = request.args.get('date')
    print(f"請求日期: {end_date}")
    
    try:
        # 獲取數據
        fgi_data = get_fear_greed_data(5, end_date)
        if not fgi_data:
            return jsonify({'success': False, 'error': '無法獲取 Fear & Greed Index 數據'})
            
        price_data = get_bitcoin_price_data(5, end_date)
        if not price_data:
            return jsonify({'success': False, 'error': '無法獲取價格數據'})
            
        chart_base64, ohlc_data = get_bitcoin_5day_ohlc_chart(end_date)
        if not chart_base64 or ohlc_data is None:
            return jsonify({'success': False, 'error': '無法生成K線圖'})

        # 返回數據
        return jsonify({
            'success': True,
            'fear_greed_data': [{
                'date': item['date'].strftime('%Y-%m-%d'),
                'value': item['value'],
                'classification': item['classification']
            } for item in fgi_data],
            'price_data': [{
                'date': item['date'].strftime('%Y-%m-%d'),
                'price': item['price']
            } for item in price_data],
            'chart': chart_base64
        })
    except Exception as e:
        print(f"API錯誤: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/analyze-chart', methods=['POST'])
def api_analyze_chart():
    try:
        data = request.get_json()
        chart_base64 = data.get('chart_data')
        if not chart_base64:
            return jsonify({'success': False, 'error': '未收到圖表數據'})
            
        # 使用 with 語句確保檔案正確關閉
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(base64.b64decode(chart_base64))
            temp_file_path = temp_file.name
            
        try:
            # 進行圖像分析
            image_class = predict_image_class(temp_file_path)
        finally:
            # 確保在分析完後刪除檔案
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"刪除臨時檔案時發生錯誤: {e}")
                
        return jsonify({
            'success': True,
            'prediction': image_class
        })
        
    except Exception as e:
        print(f"圖像分析過程中發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            print("開始處理 POST 請求...")
            
            if 'api_data' not in request.form:
                raise ValueError("未接收到 API 數據")
                
            if request.form.get('use_api_image') != 'true':
                raise ValueError("使用 API 圖像標記未設置")
                
            # 解析 API 數據
            api_data = json.loads(request.form['api_data'])
            print("收到的 API 數據:", api_data)

            regression_r2 = 0.9921031529442089
            regression_mse = 2182245.496926982

            
            # 確保有5天數據，並強制轉換為數值型
            fgi_values = [float(item['value']) for item in api_data['fear_greed_data']] 
            prices = [float(item['price']) for item in api_data['price_data']]    
            chart_base64 = api_data.get('chart')
            
            # 進行圖像分析
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_file.write(base64.b64decode(chart_base64))
                    temp_file_path = temp_file.name
                    print("臨時檔案路徑:", temp_file_path)
                    image_class = predict_image_class(temp_file_path)
                    img = image.load_img(temp_file_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 添加 batch 維度
                    image_prob = float(keras_model.predict(img_array, verbose=0)[0][0])  # 使用 float() 確保輸出是純量
                    print("圖像預測機率:", image_prob)
                    print("圖像預測趨勢:",'上漲' if image_class == 1 else '下跌')

            finally:
                # 確保在完成後刪除臨時檔案
                if temp_file:
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        print(f"刪除臨時檔案時發生錯誤: {e}")
            
            # 進行價格預測
            predicted_price, = predict_next_price(fgi_values, prices)
            print("REGRESSION 預測價格:", predicted_price)

            arima_results = predict_next_price_arima(fgi_values, prices)
            print("ARIMA 預測價格:", arima_results['prediction'])
            print("ARIMA 模型評估指標:", arima_results['metrics'])
            
            # 生成分析結果，並獲取對話歷史所需元件
            analysis, user_input, role_description = generate_llm_analysis(
                fgi_values, 
                prices, 
                image_class,
                image_prob,
                predicted_price, 
                arima_results
            )

            # 返回所有結果給模板，包括用於對話的初始上下文
            return render_template(
                'result.html', 
                prediction=analysis,
                image_trend='上漲' if image_class == 1 else '下跌',
                image_prob=image_prob,
                regression_price=predicted_price,
                regression_r2=regression_r2,
                regression_mse=regression_mse,
                arima_price=arima_results['prediction'],
                arima_metrics=arima_results['metrics'],
                user_input=user_input, # 傳遞初始用戶提問
                role_description=role_description # 傳遞系統角色設定
            )
            
        except Exception as e:
            error_msg = f"分析過程中發生錯誤: {str(e)}"
            print(f"錯誤詳情: {error_msg}")
            import traceback
            print(traceback.format_exc())
            return render_template('result.html', prediction=error_msg)
    
    return render_template('index.html')

# 【新增】處理聊天請求的 API 端點
@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    try:
        data = request.get_json()
        messages = data.get('messages') # 接收包含完整歷史的對話列表

        if not messages:
            return jsonify({'success': False, 'error': '未收到對話歷史'}), 400

        # 直接將帶有歷史紀錄的 messages 傳遞給模型
        response = llm_gemini.invoke(messages)
        
        return jsonify({
            'success': True, 
            'reply': response.content
        })

    except Exception as e:
        print(f"Chat API 錯誤: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True, port=5001)