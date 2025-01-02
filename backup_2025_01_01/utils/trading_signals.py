import numpy as np
from enum import Enum

class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradingSignals:
    def __init__(self):
        self.rsi_oversold = 30
        self.rsi_overbought = 70
    
    def calculate_trend(self, df):
        """حساب اتجاه السعر"""
        try:
            current_price = df['close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            
            # حساب قوة الاتجاه
            trend_strength = ((current_price - ma50) / ma50) * 100
            
            return trend_strength
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 0
    
    def analyze_momentum(self, df):
        """تحليل قوة الزخم"""
        try:
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            signal = df['Signal_Line'].iloc[-1]
            
            momentum_score = 0
            
            # تحليل RSI
            if rsi < self.rsi_oversold:
                momentum_score += 1
            elif rsi > self.rsi_overbought:
                momentum_score -= 1
            
            # تحليل MACD
            if macd > signal:
                momentum_score += 1
            elif macd < signal:
                momentum_score -= 1
            
            return momentum_score
        except Exception as e:
            print(f"Error analyzing momentum: {str(e)}")
            return 0
    
    def calculate_volatility(self, df):
        """حساب التقلب"""
        try:
            # حساب التقلب النسبي
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(24)  # تحويل إلى تقلب يومي
            return volatility
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0
    
    def generate_trading_signal(self, df, predicted_price):
        """توليد إشارة التداول"""
        try:
            if df is None or df.empty:
                return {'signal': Signal.HOLD, 'confidence': 0}
            
            current_price = df['close'].iloc[-1]
            
            # حساب نسبة التغير المتوقعة
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            # تحليل المؤشرات الفنية
            trend = self.calculate_trend(df)
            momentum = self.analyze_momentum(df)
            volatility = self.calculate_volatility(df)
            
            # حساب النتيجة النهائية
            signal_score = price_change + trend + (momentum * 2)
            
            # تعديل الثقة بناءً على التقلب
            confidence = abs(signal_score) * (1 - volatility)
            confidence = min(max(confidence, 0), 100)  # تحديد النطاق بين 0 و 100
            
            # تحديد الإشارة
            if signal_score > 1 and confidence > 30:
                signal = Signal.BUY
            elif signal_score < -1 and confidence > 30:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            
            return {
                'signal': signal,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error generating trading signal: {str(e)}")
            return {'signal': Signal.HOLD, 'confidence': 0}
    
    def calculate_target_prices(self, current_price, signal, confidence):
        """حساب أهداف الربح ووقف الخسارة"""
        try:
            if signal == Signal.HOLD or confidence < 30:
                return None, None
            
            # حساب النسب بناءً على الثقة
            risk_ratio = confidence / 100  # كلما زادت الثقة، زاد الهدف
            
            if signal == Signal.BUY:
                take_profit = current_price * (1 + (0.02 * risk_ratio))
                stop_loss = current_price * (1 - (0.01 * risk_ratio))
            else:  # SELL
                take_profit = current_price * (1 - (0.02 * risk_ratio))
                stop_loss = current_price * (1 + (0.01 * risk_ratio))
            
            return take_profit, stop_loss
            
        except Exception as e:
            print(f"Error calculating target prices: {str(e)}")
            return None, None
