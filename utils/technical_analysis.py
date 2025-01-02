"""
محلل المؤشرات الفنية
"""

import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import ta

class TechnicalAnalyzer:
    """محلل المؤشرات الفنية"""
    
    def __init__(self):
        """تهيئة المحلل الفني"""
        logging.info("تم تهيئة محلل المؤشرات الفنية")
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        تحليل البيانات وحساب المؤشرات الفنية
        
        المعاملات:
            df (pd.DataFrame): إطار البيانات مع أعمدة OHLCV
            
        العائد:
            Dict[str, Any]: قاموس يحتوي على نتائج التحليل
        """
        try:
            if df.empty:
                logging.warning("لا توجد بيانات للتحليل")
                return {}
                
            logging.info("بدء التحليل الفني")
            results = {}
            
            # التأكد من وجود الأعمدة المطلوبة
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logging.error(f"العمود المطلوب {col} غير موجود في البيانات")
                    return {}
            
            # تحويل الأعمدة إلى أرقام
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # حذف الصفوف التي تحتوي على قيم NaN
            df = df.dropna()
            
            if len(df) < 50:
                logging.warning("عدد نقاط البيانات غير كافٍ للتحليل")
                return {}
            
            # حساب المتوسطات المتحركة
            df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['MA50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['MA200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # حساب المتوسطات المتحركة الأسية
            df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # حساب مؤشرات الزخم
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            df['STOCH_K'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['STOCH_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            
            # حساب مؤشر MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
            
            # حساب مؤشرات التقلب
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # حساب مؤشر ATR
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # حساب مؤشرات الحجم
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['ADI'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            
            # حساب مؤشرات إضافية
            df['ROC'] = ta.momentum.roc(df['close'], window=9)  # معدل التغير
            df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)  # مؤشر تدفق المال
            df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)  # مؤشر القناة السلعية
            
            # تحليل الاتجاه
            last_close = df['close'].iloc[-1]
            ma20_last = df['MA20'].iloc[-1]
            ma50_last = df['MA50'].iloc[-1]
            ma200_last = df['MA200'].iloc[-1]
            rsi_last = df['RSI'].iloc[-1]
            stoch_k_last = df['STOCH_K'].iloc[-1]
            stoch_d_last = df['STOCH_D'].iloc[-1]
            atr_last = df['ATR'].iloc[-1]
            roc_last = df['ROC'].iloc[-1]
            mfi_last = df['MFI'].iloc[-1]
            cci_last = df['CCI'].iloc[-1]
            
            # تحليل الاتجاه المتقدم
            trend_score = 0
            trend_signals = []
            
            # تحليل المتوسطات المتحركة
            if last_close > ma20_last:
                trend_score += 1
                trend_signals.append("السعر فوق MA20")
            if ma20_last > ma50_last:
                trend_score += 1
                trend_signals.append("MA20 فوق MA50")
            if ma50_last > ma200_last:
                trend_score += 2
                trend_signals.append("MA50 فوق MA200")
            
            # تحليل RSI
            if 30 < rsi_last < 70:
                trend_score += 1
                trend_signals.append("RSI في النطاق الصحي")
            elif rsi_last <= 30:
                trend_signals.append("RSI يشير إلى تشبع البيع")
            elif rsi_last >= 70:
                trend_signals.append("RSI يشير إلى تشبع الشراء")
            
            # تحليل Stochastic
            if stoch_k_last > stoch_d_last and stoch_k_last < 80:
                trend_score += 1
                trend_signals.append("Stochastic في وضع إيجابي")
            elif stoch_k_last < stoch_d_last and stoch_k_last > 20:
                trend_score -= 1
                trend_signals.append("Stochastic في وضع سلبي")
            
            # تحليل MACD
            macd_last = df['MACD'].iloc[-1]
            macd_signal_last = df['MACD_signal'].iloc[-1]
            if macd_last > macd_signal_last:
                trend_score += 1
                trend_signals.append("MACD إيجابي")
            else:
                trend_score -= 1
                trend_signals.append("MACD سلبي")
            
            # تحليل ROC
            if roc_last > 0:
                trend_score += 1
                trend_signals.append("معدل التغير إيجابي")
            else:
                trend_score -= 1
                trend_signals.append("معدل التغير سلبي")
            
            # تحليل MFI
            if 20 < mfi_last < 80:
                trend_score += 1
                trend_signals.append("تدفق المال متوازن")
            elif mfi_last <= 20:
                trend_signals.append("تدفق المال يشير إلى تشبع البيع")
            elif mfi_last >= 80:
                trend_signals.append("تدفق المال يشير إلى تشبع الشراء")
            
            # تحليل CCI
            if -100 < cci_last < 100:
                trend_score += 1
                trend_signals.append("CCI في النطاق المتوازن")
            elif cci_last <= -100:
                trend_signals.append("CCI يشير إلى تشبع البيع")
            elif cci_last >= 100:
                trend_signals.append("CCI يشير إلى تشبع الشراء")
            
            # تحديد الاتجاه النهائي
            if trend_score >= 5:
                trend = 'صعود قوي'
            elif trend_score >= 2:
                trend = 'صعود'
            elif trend_score <= -5:
                trend = 'هبوط قوي'
            elif trend_score <= -2:
                trend = 'هبوط'
            else:
                trend = 'متذبذب'
            
            # حساب قوة الاتجاه
            trend_strength = abs(trend_score) / 8 * 100  # تحويل إلى نسبة مئوية
            
            # تجميع النتائج
            results = {
                'trend': trend,
                'trend_strength': trend_strength,
                'trend_signals': trend_signals,
                'current_price': last_close,
                'ma': {
                    'ma20': ma20_last,
                    'ma50': ma50_last,
                    'ma200': ma200_last
                },
                'ema': {
                    'ema20': df['EMA20'].iloc[-1],
                    'ema50': df['EMA50'].iloc[-1]
                },
                'rsi': rsi_last,
                'stoch': {
                    'k': stoch_k_last,
                    'd': stoch_d_last
                },
                'macd': {
                    'macd': macd_last,
                    'signal': macd_signal_last,
                    'hist': df['MACD_hist'].iloc[-1]
                },
                'bollinger': {
                    'upper': df['BB_upper'].iloc[-1],
                    'middle': df['BB_middle'].iloc[-1],
                    'lower': df['BB_lower'].iloc[-1],
                    'width': df['BB_width'].iloc[-1]
                },
                'volume': {
                    'obv': df['OBV'].iloc[-1],
                    'adi': df['ADI'].iloc[-1]
                },
                'additional': {
                    'roc': roc_last,
                    'mfi': mfi_last,
                    'cci': cci_last
                },
                'atr': atr_last
            }
            
            return results
            
        except Exception as e:
            logging.error(f"خطأ في التحليل الفني: {str(e)}")
            traceback.print_exc()
            return {}
            
    def _make_decision(self, signals: Dict[str, Any], patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """اتخاذ قرار التداول بناءً على الإشارات والأنماط"""
        try:
            logging.info("بدء اتخاذ قرار التداول")
            
            if patterns is None:
                patterns = []

            # تحليل الإشارات
            score = 0.0
            weight = 0.0
            
            # تحليل المتوسطات المتحركة
            current_price = signals.get('Current_Price', 0)
            ma20 = signals.get('MA20', 0)
            ma50 = signals.get('MA50', 0)
            
            if current_price and ma20 and ma50:
                # تحليل موقع السعر من المتوسطات
                if current_price > ma20 > ma50:  # اتجاه صعودي
                    score += 2.0
                elif current_price > ma20 and ma20 < ma50:  # بداية اتجاه صعودي
                    score += 1.0
                elif current_price < ma20 < ma50:  # اتجاه هبوطي
                    score -= 2.0
                elif current_price < ma20 and ma20 > ma50:  # بداية اتجاه هبوطي
                    score -= 1.0
                weight += 1.0
            
            # تحليل مؤشرات القوة
            rsi = signals.get('RSI', 50)
            if rsi > 70:
                score -= 1.0
            elif rsi < 30:
                score += 1.0
            weight += 1.0
            
            # تحليل MACD
            macd = signals.get('MACD', 0)
            macd_signal = signals.get('MACD_Signal', 0)
            if macd and macd_signal:
                if macd > macd_signal:
                    score += 1.0
                else:
                    score -= 1.0
                weight += 1.0
            
            # تحليل Bollinger Bands
            bb_upper = signals.get('Bollinger_Upper', 0)
            bb_lower = signals.get('Bollinger_Lower', 0)
            if current_price and bb_upper and bb_lower:
                if current_price > bb_upper:
                    score -= 1.0
                elif current_price < bb_lower:
                    score += 1.0
                weight += 1.0
            
            # تحليل الأنماط
            pattern_score = 0.0
            pattern_weight = 0.0
            pattern_confidence = 0.0
            
            for pattern in patterns:
                strength = pattern.get('strength', 0)
                direction = pattern.get('direction', '')
                confidence = pattern.get('confidence', 0)
                
                if direction == 'صعود':
                    pattern_score += strength
                elif direction == 'هبوط':
                    pattern_score -= strength
                
                pattern_weight += 1.0
                pattern_confidence = max(pattern_confidence, confidence)
            
            # حساب النتيجة النهائية
            final_score = 0.0
            if weight > 0:
                final_score = score / weight
            
            pattern_final_score = 0.0
            if pattern_weight > 0:
                pattern_final_score = pattern_score / pattern_weight
            
            # تحديد القرار
            final_signal = (final_score + pattern_final_score) / 2
            decision = self._determine_final_decision(final_signal)
            
            # حساب مستويات التداول
            volatility = signals.get('Volatility', 0)
            if volatility:
                if decision == 'شراء':
                    entry_price = current_price
                    stop_loss = entry_price - (volatility * 1.5)
                    take_profit = entry_price + (volatility * 2.5)
                elif decision == 'بيع':
                    entry_price = current_price
                    stop_loss = entry_price + (volatility * 1.5)
                    take_profit = entry_price - (volatility * 2.5)
                else:
                    entry_price = current_price
                    stop_loss = current_price * 0.95
                    take_profit = current_price * 1.05
            else:
                entry_price = current_price
                stop_loss = current_price * 0.95
                take_profit = current_price * 1.05
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            
            # تجميع التوصيات
            recommendations = []
            for pattern in patterns:
                recommendations.append({
                    'name': pattern.get('name', ''),
                    'direction': pattern.get('direction', ''),
                    'target': pattern.get('target', 0),
                    'strength': pattern.get('strength', 0)
                })
            
            # إنشاء النتيجة
            result = {
                'decision': decision,
                'confidence': abs(final_signal),
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'risk_reward_ratio': float(risk_reward_ratio),
                'analysis': {
                    'technical_score': float(abs(final_score)),
                    'market_score': float(abs(pattern_confidence)),
                    'pattern_score': float(abs(pattern_final_score)),
                    'risk_score': float(1 - abs(final_signal))
                },
                'recommendations': recommendations,
                'signals': {
                    'RSI': float(rsi),
                    'MACD': float(macd),
                    'MACD_Signal': float(macd_signal),
                    'Bollinger_Upper': float(bb_upper),
                    'Bollinger_Middle': float(ma20),
                    'Bollinger_Lower': float(bb_lower),
                    'Current_Price': float(current_price),
                    'MA20': float(ma20),
                    'MA50': float(ma50)
                }
            }
            
            logging.info(f"تم اتخاذ القرار: {result}")
            return result
            
        except Exception as e:
            logging.error(f"خطأ في اتخاذ القرار: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                'decision': 'محايد',
                'confidence': 0.0,
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'risk_reward_ratio': 1.0,
                'analysis': {
                    'technical_score': 0.0,
                    'market_score': 0.0,
                    'pattern_score': 0.0,
                    'risk_score': 0.5
                },
                'recommendations': [],
                'signals': {
                    'RSI': 50.0,
                    'MACD': 0.0,
                    'MACD_Signal': 0.0,
                    'Bollinger_Upper': 0.0,
                    'Bollinger_Middle': 0.0,
                    'Bollinger_Lower': 0.0,
                    'Current_Price': 0.0,
                    'MA20': 0.0,
                    'MA50': 0.0
                }
            }
            
    def _determine_final_decision(self, final_signal: float) -> str:
        """تحديد القرار النهائي بناءً على الإشارة النهائية"""
        try:
            if final_signal > 0.5:
                return 'شراء'
            elif final_signal < -0.5:
                return 'بيع'
            else:
                return 'محايد'
        except Exception as e:
            logging.error(f"خطأ في تحديد القرار النهائي: {str(e)}")
            return 'محايد'

    def analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الأنماط في البيانات"""
        try:
            patterns = []
            
            # تحليل نمط الرأس والكتفين
            head_shoulders = self._detect_head_shoulders(df)
            if head_shoulders:
                patterns.append(head_shoulders)
            
            # تحليل نمط المثلث
            triangle = self._detect_triangle(df)
            if triangle:
                patterns.append(triangle)
            
            # تحليل نمط القمة المزدوجة
            double_top = self._detect_double_top(df)
            if double_top:
                patterns.append(double_top)
            
            # تحليل نمط القاع المزدوج
            double_bottom = self._detect_double_bottom(df)
            if double_bottom:
                patterns.append(double_bottom)
            
            return {'patterns': patterns}
        except Exception as e:
            logging.error(f"خطأ في تحليل الأنماط: {str(e)}")
            return {'patterns': []}
            
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """تحليل نمط الرأس والكتفين"""
        try:
            # تحليل نمط الرأس والكتفين
            # هذا مجرد مثال بسيط، يمكن تحسينه
            high = df['high'].values
            if len(high) < 5:
                return None
                
            # البحث عن قمم محلية
            peaks = []
            for i in range(2, len(high)-2):
                if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                    peaks.append((i, high[i]))
            
            if len(peaks) >= 3:
                # التحقق من نمط الرأس والكتفين
                for i in range(len(peaks)-2):
                    left_shoulder = peaks[i][1]
                    head = peaks[i+1][1]
                    right_shoulder = peaks[i+2][1]
                    
                    if head > left_shoulder and head > right_shoulder and abs(left_shoulder - right_shoulder) / left_shoulder < 0.1:
                        return {
                            'name': 'رأس وكتفين',
                            'direction': 'هبوط',
                            'strength': 0.8,
                            'confidence': 0.7,
                            'target': min(df['low'].values[peaks[i][0]:peaks[i+2][0]])
                        }
            
            return None
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط الرأس والكتفين: {str(e)}")
            return None
            
    def _detect_triangle(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """تحليل نمط المثلث"""
        try:
            if len(df) < 10:
                return None
                
            highs = df['high'].values[-10:]
            lows = df['low'].values[-10:]
            
            # حساب خط الاتجاه العلوي والسفلي
            high_slope = (highs[-1] - highs[0]) / len(highs)
            low_slope = (lows[-1] - lows[0]) / len(lows)
            
            # التحقق من نمط المثلث
            if abs(high_slope) < 0.001 and low_slope > 0:
                return {
                    'name': 'مثلث صاعد',
                    'direction': 'صعود',
                    'strength': 0.7,
                    'confidence': 0.6,
                    'target': highs[-1] + (highs[-1] - lows[-1])
                }
            elif high_slope < 0 and abs(low_slope) < 0.001:
                return {
                    'name': 'مثلث هابط',
                    'direction': 'هبوط',
                    'strength': 0.7,
                    'confidence': 0.6,
                    'target': lows[-1] - (highs[-1] - lows[-1])
                }
            
            return None
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط المثلث: {str(e)}")
            return None
            
    def _detect_double_top(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """تحليل نمط القمة المزدوجة"""
        try:
            if len(df) < 10:
                return None
                
            highs = df['high'].values[-10:]
            
            # البحث عن قمتين متساويتين تقريباً
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                for i in range(len(peaks)-1):
                    if abs(peaks[i][1] - peaks[i+1][1]) / peaks[i][1] < 0.02:
                        return {
                            'name': 'قمة مزدوجة',
                            'direction': 'هبوط',
                            'strength': 0.9,
                            'confidence': 0.8,
                            'target': min(df['low'].values[peaks[i][0]:peaks[i+1][0]])
                        }
            
            return None
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط القمة المزدوجة: {str(e)}")
            return None
            
    def _detect_double_bottom(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """تحليل نمط القاع المزدوج"""
        try:
            if len(df) < 10:
                return None
                
            lows = df['low'].values[-10:]
            
            # البحث عن قاعين متساويين تقريباً
            troughs = []
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 2:
                for i in range(len(troughs)-1):
                    if abs(troughs[i][1] - troughs[i+1][1]) / troughs[i][1] < 0.02:
                        return {
                            'name': 'قاع مزدوج',
                            'direction': 'صعود',
                            'strength': 0.9,
                            'confidence': 0.8,
                            'target': max(df['high'].values[troughs[i][0]:troughs[i+1][0]])
                        }
            
            return None
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط القاع المزدوج: {str(e)}")
            return None
