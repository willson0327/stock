from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def calculate_moving_average(data, window):
    return data.rolling(window=window).mean()

def predict_future(close_price, days):
    X = np.arange(len(close_price)).reshape(-1, 1)
    y = close_price
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(close_price), len(close_price) + days).reshape(-1, 1)
    future_y = model.predict(future_X)
    return future_y, model.coef_[0]

def generate_stock_chart(ticker):
    # 参数配置
    start_date = '2023-01-01'
    end_date = '2024-12-31'

    # 下载历史数据
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        return None, None

    # 计算移动平均线
    data['ma20'] = calculate_moving_average(data['Close'], 20)
    data['ma50'] = calculate_moving_average(data['Close'], 50)

    # 线性回归预测
    future_y, slope = predict_future(data['Close'], 30)
    future_dates = pd.date_range(start=data.index[-1], periods=31)[1:]
    future_data = pd.DataFrame({'Date': future_dates, 'Close': future_y}, index=future_dates)

    # 生成图表
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['Close'], label='Close Price', color='blue')
    ax.plot(data['ma20'], label='20-day MA', color='orange', alpha=0.3, linestyle='--')
    ax.plot(data['ma50'], label='50-day MA', color='green', alpha=0.3, linestyle='--')
    ax.plot(future_data['Close'], label='Prediction', linestyle='-.', color='purple')
    prediction_text = 'Predicting future rise' if slope > 0 else 'Predicting future decline'
    color = 'red' if slope > 0 else 'green'
    ax.text(0.95, 1.05, prediction_text, transform=ax.transAxes, color=color, ha='right', va='top')
    ax.set_title(f'Stock Analysis for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    # 将图表保存到内存中，并转换为 base64 编码的字符串
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 清除图表，释放资源
    plt.close()

    return plot_data, data.to_html()

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_data = None
    table_data = None

    if request.method == 'POST':
        ticker = request.form['ticker'] + ".TW"
        plot_data, table_data = generate_stock_chart(ticker)

    return render_template('index.html', plot_data=plot_data, table_data=table_data)

if __name__ == '__main__':
    app.run(debug=True)
