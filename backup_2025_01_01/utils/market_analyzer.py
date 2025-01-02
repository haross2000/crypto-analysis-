"""
محلل بيانات السوق
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import talib
from scipy import stats

class MarketAnalyzer:
    def __init__(self):
        """تهيئة محلل السوق"""
        logging.info("تم تهيئة محلل السوق")

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل السوق باستخدام المؤشرات الفنية"""
        try:
            if df.empty:
                return {}

            results = {}
            
            # تحويل الأعمدة إلى float
            ohlc = df[['open', 'high', 'low', 'close']].astype(float)
            
            try:
                # RSI
                rsi = talib.RSI(ohlc['close'])
                if not pd.isna(rsi.iloc[-1]):
                    last_rsi = float(rsi.iloc[-1])
                    results['RSI'] = {
                        'value': last_rsi,
                        'signal': 'بيع' if last_rsi > 70 else 'شراء' if last_rsi < 30 else 'محايد',
                        'strength': abs(50 - last_rsi) / 50,  # قوة الإشارة
                        'description': 'مؤشر القوة النسبية'
                    }
            except Exception as e:
                logging.error(f"خطأ في حساب RSI: {str(e)}")

            try:
                # MACD
                macd = df['MACD'].iloc[-1]
                signal = df['MACD_Signal'].iloc[-1]
                hist = df['MACD_Hist'].iloc[-1]
                
                if not pd.isna(macd) and not pd.isna(signal):
                    results['MACD'] = {
                        'value': macd,
                        'signal': 'شراء' if hist > 0 and macd > signal else 'بيع' if hist < 0 and macd < signal else 'محايد',
                        'strength': abs(hist) / abs(macd) if abs(macd) > 0 else 0,
                        'description': 'مؤشر تقارب وتباعد المتوسطات المتحركة'
                    }
            except Exception as e:
                logging.error(f"خطأ في حساب MACD: {str(e)}")

            try:
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(ohlc['close'])
                if not any(pd.isna([upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]])):
                    last_close = float(ohlc['close'].iloc[-1])
                    last_upper = float(upper.iloc[-1])
                    last_lower = float(lower.iloc[-1])
                    
                    bb_position = (last_close - last_lower) / (last_upper - last_lower) if (last_upper - last_lower) > 0 else 0.5
                    results['Bollinger Bands'] = {
                        'value': bb_position,
                        'signal': 'بيع' if bb_position > 0.8 else 'شراء' if bb_position < 0.2 else 'محايد',
                        'strength': abs(0.5 - bb_position) * 2,
                        'description': 'حزم بولينجر'
                    }
            except Exception as e:
                logging.error(f"خطأ في حساب Bollinger Bands: {str(e)}")

            try:
                # Stochastic RSI
                fastk, fastd = talib.STOCHRSI(ohlc['close'])
                if not pd.isna(fastk.iloc[-1]):
                    last_k = float(fastk.iloc[-1])
                    
                    results['Stochastic RSI'] = {
                        'value': last_k,
                        'signal': 'بيع' if last_k > 80 else 'شراء' if last_k < 20 else 'محايد',
                        'strength': abs(50 - last_k) / 50,
                        'description': 'مؤشر ستوكاستك RSI'
                    }
            except Exception as e:
                logging.error(f"خطأ في حساب Stochastic RSI: {str(e)}")

            try:
                # ADX
                adx = talib.ADX(ohlc['high'], ohlc['low'], ohlc['close'])
                if not pd.isna(adx.iloc[-1]):
                    last_adx = float(adx.iloc[-1])
                    
                    results['ADX'] = {
                        'value': last_adx,
                        'signal': 'قوي' if last_adx > 25 else 'ضعيف',
                        'strength': last_adx / 100,
                        'description': 'مؤشر اتجاه الحركة'
                    }
            except Exception as e:
                logging.error(f"خطأ في حساب ADX: {str(e)}")

            try:
                # OBV
                obv = talib.OBV(ohlc['close'], df['volume'].astype(float))
                if not pd.isna(obv.iloc[-1]):
                    obv_sma = pd.Series(obv).rolling(20).mean()
                    last_obv = float(obv.iloc[-1])
                    last_obv_sma = float(obv_sma.iloc[-1])
                    
                    results['OBV'] = {
                        'value': last_obv,
                        'signal': 'شراء' if last_obv > last_obv_sma else 'بيع' if last_obv < last_obv_sma else 'محايد',
                        'strength': abs(last_obv - last_obv_sma) / abs(last_obv) if abs(last_obv) > 0 else 0,
                        'description': 'مؤشر توازن الحجم'
                    }
            except Exception as e:
                logging.error(f"خطأ في حساب OBV: {str(e)}")

            return results

        except Exception as e:
            logging.error(f"خطأ في تحليل السوق: {str(e)}")
            return {}
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """حساب الزخم"""
        try:
            df_reset = df.reset_index()
            # حساب ROC لعدة فترات
            roc_10 = ((float(df_reset['close'].iloc[-1]) - float(df_reset['close'].iloc[-10])) / 
                     float(df_reset['close'].iloc[-10])) * 100
            roc_20 = ((float(df_reset['close'].iloc[-1]) - float(df_reset['close'].iloc[-20])) / 
                     float(df_reset['close'].iloc[-20])) * 100
            
            # المتوسط المرجح
            momentum = (roc_10 * 0.6 + roc_20 * 0.4)
            
            # تطبيع النتيجة إلى نطاق 0-100
            return max(min((momentum + 100) / 2, 100), 0)
        except Exception as e:
            logging.error(f"خطأ في حساب الزخم: {str(e)}")
            return 50.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """حساب قوة الاتجاه"""
        try:
            df_reset = df.reset_index()
            # حساب خط الاتجاه
            x = np.arange(len(df_reset))
            slope, _, r_value, _, _ = stats.linregress(x, df_reset['close'].astype(float))
            
            # حساب قوة الاتجاه باستخدام R²
            r_squared = r_value ** 2
            
            # تحويل الميل إلى درجة
            angle = np.arctan(slope) * 180 / np.pi
            normalized_angle = (angle + 90) / 180  # تطبيع الزاوية إلى نطاق 0-1
            
            # الجمع بين قوة الاتجاه وزاويته
            trend_strength = (r_squared * 0.7 + normalized_angle * 0.3) * 100
            
            return max(min(trend_strength, 100), 0)
        except Exception as e:
            logging.error(f"خطأ في حساب قوة الاتجاه: {str(e)}")
            return 50.0
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """حساب مؤشر القوة النسبية"""
        try:
            df_reset = df.reset_index()
            delta = df_reset['close'].astype(float).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except Exception as e:
            logging.error(f"خطأ في حساب مؤشر القوة النسبية: {str(e)}")
            return 50.0
    
    def _calculate_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> float:
        """حساب مؤشر تدفق الأموال"""
        try:
            df_reset = df.reset_index()
            df_reset = df_reset.astype({
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            })
            
            typical_price = (df_reset['high'] + df_reset['low'] + df_reset['close']) / 3
            money_flow = typical_price * df_reset['volume']
            
            delta = typical_price.diff()
            positive_flow = (money_flow.where(delta > 0, 0)).rolling(window=period).sum()
            negative_flow = (money_flow.where(delta < 0, 0)).rolling(window=period).sum()
            
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))
            
            return float(mfi.iloc[-1])
        except Exception as e:
            logging.error(f"خطأ في حساب مؤشر تدفق الأموال: {str(e)}")
            return 50.0
    
    def _calculate_market_strength(self, momentum: float, trend_strength: float,
                                 rsi: float, relative_volume: float, mfi: float) -> float:
        """حساب قوة السوق الإجمالية"""
        weights = {
            'momentum': 0.25,
            'trend_strength': 0.25,
            'rsi': 0.2,
            'relative_volume': 0.15,
            'mfi': 0.15
        }
        
        # تطبيع المدخلات
        normalized_volume = min(relative_volume * 50, 100)  # تحويل الحجم النسبي إلى درجة
        
        # حساب المتوسط المرجح
        market_strength = (
            momentum * weights['momentum'] +
            trend_strength * weights['trend_strength'] +
            rsi * weights['rsi'] +
            normalized_volume * weights['relative_volume'] +
            mfi * weights['mfi']
        )
        
        return max(min(market_strength, 100), 0)
    
    def _calculate_risk_level(self, volatility: float, price_change: float,
                            volume_volatility: float, rsi: float) -> float:
        """حساب مستوى المخاطرة"""
        # تطبيع المدخلات
        normalized_volatility = min(volatility * 100, 100)
        normalized_price_change = min(abs(price_change) * 100, 100)
        normalized_volume_volatility = min(volume_volatility * 100, 100)
        
        # حساب المخاطر من RSI (القيم المتطرفة تشير إلى مخاطر أعلى)
        rsi_risk = abs(rsi - 50) * 2
        
        weights = {
            'volatility': 0.35,
            'price_change': 0.25,
            'volume_volatility': 0.2,
            'rsi': 0.2
        }
        
        risk_level = (
            normalized_volatility * weights['volatility'] +
            normalized_price_change * weights['price_change'] +
            normalized_volume_volatility * weights['volume_volatility'] +
            rsi_risk * weights['rsi']
        )
        
        return max(min(risk_level, 100), 0)
    
    def _determine_market_state(self, market_strength: float, risk_level: float,
                              price_change: float) -> str:
        """تحديد حالة السوق"""
        if market_strength >= 70:
            return 'قوي'
        elif market_strength <= 30 and risk_level >= 70:
            return 'ضعيف وخطر'
        elif market_strength <= 30 and risk_level < 70:
            return 'ضعيف مع استقرار'
        elif price_change > 0:
            return 'متوازن مع ميل صعودي'
        elif price_change < 0:
            return 'متوازن مع ميل هبوطي'
        else:
            return 'متوازن'
