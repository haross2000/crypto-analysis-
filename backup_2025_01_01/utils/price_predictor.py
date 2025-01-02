import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class PricePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def prepare_data(self, df):
        """تحضير البيانات للتنبؤ"""
        try:
            if df is None or df.empty:
                return None
            
            # استخدام المؤشرات الفنية للتنبؤ
            features = ['close', 'volume', 'RSI', 'MACD', 'MA20', 'MA50']
            data = df[features].values
            
            # تطبيع البيانات
            scaled_data = self.scaler.fit_transform(data)
            
            return scaled_data
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None
    
    def predict_price(self, df):
        """التنبؤ بالسعر التالي"""
        try:
            if df is None or df.empty:
                return None
            
            # حساب المؤشرات المتقدمة
            returns = df['close'].pct_change()
            avg_return = returns.mean()
            volatility = returns.std()
            
            # حساب مؤشرات الزخم
            momentum = returns.rolling(window=10).mean().iloc[-1]
            
            # تحليل الاتجاه
            current_price = df['close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            
            # حساب قوة الاتجاه
            trend_strength = 0
            if ma20 > ma50:
                trend_strength = (ma20 - ma50) / ma50
            elif ma20 < ma50:
                trend_strength = (ma50 - ma20) / ma50
            
            # حساب مؤشر التقلب النسبي
            rsi = df['RSI'].iloc[-1]
            rsi_trend = 1 if rsi > 50 else -1
            
            # دمج جميع العوامل للتنبؤ
            price_change = (
                avg_return * 0.2 +  # متوسط العائد
                momentum * 0.3 +    # الزخم
                trend_strength * 0.3 +  # قوة الاتجاه
                (rsi_trend * 0.2)   # اتجاه RSI
            )
            
            # تعديل التغير بناءً على التقلب
            price_change *= (1 + volatility)
            
            # حساب السعر المتوقع
            predicted_price = current_price * (1 + price_change)
            
            # حساب نطاق الثقة
            confidence_range = volatility * current_price
            
            return {
                'predicted_price': predicted_price,
                'confidence_low': predicted_price - confidence_range,
                'confidence_high': predicted_price + confidence_range,
                'trend_strength': trend_strength,
                'volatility': volatility
            }
            
        except Exception as e:
            print(f"Error predicting price: {str(e)}")
            return None
