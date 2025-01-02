"""
اختبار نموذج التعلم العميق المتقدم
"""

import yfinance as yf
import pandas as pd
from utils.deep_learning_model import DeepLearningPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)

def evaluate_predictions(y_true, y_pred):
    """تقييم التوقعات وعرض المقاييس"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    logging.info("\nنتائج التقييم:")
    logging.info(f"MAE: ${mae:,.2f}")
    logging.info(f"MSE: ${mse:,.2f}")
    logging.info(f"RMSE: ${rmse:,.2f}")
    logging.info(f"R²: {r2:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")
    
    return mae, mse, rmse, r2, mape

def plot_predictions(dates, y_true, y_pred, title):
    """رسم التوقعات مقابل القيم الحقيقية"""
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label='القيم الحقيقية', color='blue', linewidth=2)
    plt.plot(dates, y_pred, label='التوقعات', color='red', linestyle='--', linewidth=2)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('التاريخ', fontsize=12)
    plt.ylabel('السعر ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_data):
    """تقييم أداء النموذج"""
    try:
        # تجهيز بيانات الاختبار
        X_price, X_volume, X_technical, X_market, y_true = model.prepare_data(test_data)
        if any(x is None for x in [X_price, X_volume, X_technical, X_market, y_true]):
            raise ValueError("فشل في تجهيز بيانات الاختبار")
            
        # التنبؤ
        y_pred = model.predict(X_price, X_volume, X_technical, X_market)
        if y_pred is None:
            raise ValueError("فشل في التنبؤ")
            
        # التأكد من تطابق الأبعاد
        if len(y_true) != len(y_pred):
            y_true = y_true[:len(y_pred)]
            
        # حساب مقاييس الأداء
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        # طباعة النتائج
        logging.info("\nنتائج تقييم النموذج:")
        logging.info(f"متوسط الخطأ المطلق (MAE): ${mae:,.2f}")
        logging.info(f"الجذر التربيعي لمتوسط مربع الخطأ (RMSE): ${rmse:,.2f}")
        logging.info(f"متوسط النسبة المئوية للخطأ المطلق (MAPE): {mape:.2f}%")
        logging.info(f"معامل التحديد (R²): {r2:.4f}")
        
        # رسم النتائج
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='القيم الفعلية', color='blue', alpha=0.7)
        plt.plot(y_pred, label='التنبؤات', color='red', alpha=0.7)
        plt.title('مقارنة بين القيم الفعلية والتنبؤات')
        plt.xlabel('الفترة الزمنية')
        plt.ylabel('السعر (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.close()
        
        return mae, rmse, mape, r2
        
    except Exception as e:
        logging.error(f"خطأ في تقييم النموذج: {str(e)}")
        return None, None, None, None

def predict_future(model, last_data, days=7):
    """التنبؤ بأسعار البيتكوين للأيام القادمة"""
    try:
        future_dates = pd.date_range(start=last_data.index[-1] + pd.Timedelta(days=1), periods=days)
        predictions = []
        current_data = last_data.copy()
        last_close = current_data['close'].iloc[-1]
        
        for i in range(days):
            # تجهيز البيانات
            X_price, X_volume, X_technical, X_market, _ = model.prepare_data(current_data)
            if any(x is None for x in [X_price, X_volume, X_technical, X_market]):
                raise ValueError("فشل في تجهيز البيانات للتنبؤ")
            
            # التنبؤ
            try:
                prediction = model.predict(X_price[-1:], X_volume[-1:], X_technical[-1:], X_market[-1:])
                if prediction is None:
                    raise ValueError("فشل في التنبؤ")
                prediction = prediction[0]
            except Exception as e:
                logging.error(f"خطأ في التنبؤ لليوم {i+1}: {str(e)}")
                return None, None
            
            predictions.append(prediction)
            
            # تحديث البيانات للتنبؤ التالي
            new_row = pd.DataFrame({
                'close': [prediction],
                'high': [prediction * 1.01],
                'low': [prediction * 0.99],
                'open': [prediction],
                'volume': [current_data['volume'].mean()]
            }, index=[future_dates[i]])
            
            current_data = pd.concat([current_data[1:], new_row])
            
            # حساب نسبة التغير
            change_pct = ((prediction - last_close) / last_close) * 100
            
            # عرض التنبؤ
            logging.info(f"\nتوقعات اليوم {i+1}:")
            logging.info(f"التاريخ: {future_dates[i].strftime('%Y-%m-%d')}")
            logging.info(f"السعر المتوقع: ${prediction:,.2f}")
            logging.info(f"نسبة التغير: {change_pct:+.2f}%")
            logging.info(f"الاتجاه: {'▲ صعود' if change_pct > 0 else '▼ هبوط'}")
        
        # رسم التوقعات المستقبلية
        plt.figure(figsize=(15, 7))
        plt.plot(future_dates, predictions, marker='o', linestyle='--', color='green', linewidth=2, markersize=8)
        
        # إضافة آخر 30 يوم من البيانات التاريخية
        historical_dates = last_data.index[-30:]
        historical_prices = last_data['close'].values[-30:]
        plt.plot(historical_dates, historical_prices, color='blue', linewidth=2, label='البيانات التاريخية')
        
        plt.axvline(x=last_data.index[-1], color='red', linestyle=':', label='بداية التوقعات')
        
        plt.title('توقعات أسعار Bitcoin للأيام القادمة', fontsize=14, pad=20)
        plt.xlabel('التاريخ', fontsize=12)
        plt.ylabel('السعر ($)', fontsize=12)
        plt.legend(['التوقعات المستقبلية', 'البيانات التاريخية', 'بداية التوقعات'], fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('future_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return predictions, future_dates
        
    except Exception as e:
        logging.error(f"خطأ في التنبؤ: {str(e)}")
        return None, None

def main():
    """الدالة الرئيسية للتدريب والتقييم"""
    try:
        # تهيئة النموذج
        model = DeepLearningPredictor(sequence_length=60)
        
        # تحميل البيانات
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            raise ValueError("فشل في تحميل البيانات")
        
        # معالجة القيم المفقودة في البيانات الأصلية
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # تقسيم البيانات
        train_size = int(len(df) * 0.8)
        val_size = int(len(df) * 0.1)
        
        train_data = df[:train_size]
        val_data = df[train_size:train_size + val_size]
        test_data = df[train_size + val_size:]
        
        # تدريب النموذج
        logging.info("بدء تدريب النموذج...")
        model.train(train_data, val_data, epochs=50, batch_size=32)
        
        # تقييم النموذج
        logging.info("\nتقييم النموذج على بيانات الاختبار...")
        mae, rmse, mape, r2 = evaluate_model(model, test_data)
        
        if all(metric is not None for metric in [mae, rmse, mape, r2]):
            logging.info("\nالتنبؤ بالأسعار المستقبلية...")
            predictions, future_dates = predict_future(model, df, days=7)
            
            if predictions is not None:
                logging.info("\nتم الانتهاء من التدريب والتقييم والتنبؤ بنجاح!")
                
    except Exception as e:
        logging.error(f"خطأ في التنفيذ: {str(e)}")

if __name__ == "__main__":
    main()
