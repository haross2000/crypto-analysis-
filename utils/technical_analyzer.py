"""
وحدة تحليل المؤشرات الفنية
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import talib  # إضافة استيراد talib
class TechnicalAnalyzer:
    """
    محلل المؤشرات الفنية
    
    يقوم هذا المحلل بحساب وتحليل المؤشرات الفنية الرئيسية:
    
    1. المتوسطات المتحركة (Moving Averages):
       - EMA-20: المتوسط المتحرك الأسي القصير المدى
       - EMA-50: المتوسط المتحرك الأسي المتوسط المدى
       - يتم حساب النسب المئوية للتغير عن السعر السابق
    
    2. مؤشر القوة النسبية (RSI):
       - يقيس قوة وسرعة حركة السعر
       - القيم فوق 70 تشير إلى ذروة شراء (احتمال هبوط)
       - القيم تحت 30 تشير إلى ذروة بيع (احتمال صعود)
       - يظهر الاتجاه (صاعد/هابط) ومدة البقاء في المنطقة
    
    3. مؤشر MACD:
       - يجمع بين ثلاثة متوسطات متحركة أسية
       - خط MACD: الفرق بين EMA-12 و EMA-26
       - خط الإشارة: EMA-9 لخط MACD
       - الهستوجرام: الفرق بين MACD والإشارة
       - يتم عرض القيم كنسب مئوية من السعر
    
    ملاحظات مهمة:
    - جميع النسب المئوية محدودة بين -5% و +5%
    - يتم تنقية القيم غير المنطقية واستبدالها بصفر
    - الألوان تعكس حالة المؤشر (أخضر: إيجابي، أحمر: سلبي)
    """
    
    def __init__(self):
        """تهيئة المحلل الفني"""
        logging.info("تهيئة محلل المؤشرات الفنية")
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة المؤشرات الفنية إلى DataFrame

        Args:
            df (pd.DataFrame): DataFrame مع بيانات الأسعار

        Returns:
            pd.DataFrame: DataFrame مع المؤشرات الفنية المضافة
        """
        try:
            if df is None or df.empty:
                logging.error("DataFrame فارغ أو غير موجود")
                return df

            # المتوسطات المتحركة
            df['SMA_20'] = self.calculate_sma(df, 20)
            df['SMA_50'] = self.calculate_sma(df, 50)
            df['EMA_20'] = self.calculate_ema(df, 20)
            df['EMA_50'] = self.calculate_ema(df, 50)

            # مؤشر القوة النسبية
            rsi_data = self.calculate_rsi(df)
            df['RSI'] = rsi_data['value']
            df['RSI_Trend'] = rsi_data['trend']
            df['RSI_Zone_Duration'] = rsi_data['zone_duration']

            # مؤشر MACD
            macd_line, signal_line, macd_hist = self.calculate_macd(df)
            df['MACD'] = macd_line
            df['MACD_Signal'] = signal_line
            df['MACD_Hist'] = macd_hist

            # حزم بولينجر
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower

            # مؤشر Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(df)
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d

            # مؤشر ADX
            df['ADX'] = self.calculate_adx(df)

            # المدى الحقيقي المتوسط
            df['ATR'] = self.calculate_atr(df)

            logging.info("تم إضافة المؤشرات الفنية بنجاح")
            return df

        except Exception as e:
            logging.error(f"خطأ في إضافة المؤشرات الفنية: {str(e)}")
            return df
    
    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """حساب المتوسط المتحرك البسيط كنسبة مئوية من التغير"""
        try:
            if 'close' not in df.columns:
                logging.error("عمود 'close' غير موجود في DataFrame")
                return pd.Series(index=df.index)
                
            close_prices = df['close'].astype(float)
            sma = close_prices.rolling(window=period).mean()
            
            # حساب النسبة المئوية للتغير عن السعر الحالي
            # استخدام القيمة المطلقة في المقام لتجنب الإشارات الخاطئة
            pct_change = ((close_prices - sma) / sma) * 100
            
            # تنظيف القيم غير المنطقية
            pct_change = pct_change.apply(lambda x: min(max(x, -5), 5) if not pd.isna(x) else 0)
            
            return pct_change
            
        except Exception as e:
            logging.error(f"خطأ في حساب المتوسط المتحرك البسيط: {str(e)}")
            return pd.Series(index=df.index)

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """حساب المتوسط المتحرك الأسي كنسبة مئوية من التغير"""
        try:
            if 'close' not in df.columns:
                logging.error("عمود 'close' غير موجود في DataFrame")
                return pd.Series(index=df.index)
                
            close_prices = df['close'].astype(float)
            ema = close_prices.ewm(span=period, adjust=False).mean()
            
            # حساب النسبة المئوية للتغير عن السعر الحالي
            # استخدام القيمة المطلقة في المقام لتجنب الإشارات الخاطئة
            pct_change = ((close_prices - ema) / ema) * 100
            
            # تنظيف القيم غير المنطقية
            pct_change = pct_change.apply(lambda x: min(max(x, -5), 5) if not pd.isna(x) else 0)
            
            return pct_change
            
        except Exception as e:
            logging.error(f"خطأ في حساب المتوسط المتحرك الأسي: {str(e)}")
            return pd.Series(index=df.index)

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> dict:
        """حساب مؤشر القوة النسبية RSI مع معلومات إضافية"""
        try:
            if 'close' not in df.columns:
                logging.error("عمود 'close' غير موجود في DataFrame")
                return {'value': 50, 'trend': 'متعادل', 'zone_duration': 0}

            close_prices = df['close'].astype(float)
            delta = close_prices.diff()
            
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # حساب الاتجاه
            rsi_trend = self._calculate_rsi_trend(rsi)
            
            # حساب مدة البقاء في المنطقة الحالية
            zone_duration = self._calculate_rsi_zone_duration(rsi)
            
            current_rsi = rsi.iloc[-1]
            
            return {
                'value': current_rsi,
                'trend': rsi_trend,
                'zone_duration': zone_duration
            }
            
        except Exception as e:
            logging.error(f"خطأ في حساب مؤشر RSI: {str(e)}")
            return {'value': 50, 'trend': 'متعادل', 'zone_duration': 0}

    def _calculate_rsi_trend(self, rsi_series: pd.Series) -> str:
        """حساب اتجاه مؤشر RSI"""
        try:
            # حساب المتوسط المتحرك للـ RSI
            rsi_ma = rsi_series.rolling(window=5).mean()
            
            current_rsi = rsi_series.iloc[-1]
            prev_rsi_ma = rsi_ma.iloc[-2]
            
            # تحديد الاتجاه
            if current_rsi > prev_rsi_ma:
                if current_rsi > 70:
                    return "صاعد بقوة"
                return "صاعد"
            elif current_rsi < prev_rsi_ma:
                if current_rsi < 30:
                    return "هابط بقوة"
                return "هابط"
            return "متعادل"
            
        except Exception as e:
            logging.error(f"خطأ في حساب اتجاه RSI: {str(e)}")
            return "متعادل"

    def _calculate_rsi_zone_duration(self, rsi_series: pd.Series) -> int:
        """حساب مدة البقاء في المنطقة الحالية"""
        try:
            current_rsi = rsi_series.iloc[-1]
            duration = 0
            
            if current_rsi >= 70:  # منطقة ذروة الشراء
                for i in range(len(rsi_series)-1, -1, -1):
                    if rsi_series.iloc[i] >= 70:
                        duration += 1
                    else:
                        break
            elif current_rsi <= 30:  # منطقة ذروة البيع
                for i in range(len(rsi_series)-1, -1, -1):
                    if rsi_series.iloc[i] <= 30:
                        duration += 1
                    else:
                        break
            
            return duration
            
        except Exception as e:
            logging.error(f"خطأ في حساب مدة البقاء: {str(e)}")
            return 0

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """حساب مؤشر MACD باستخدام talib"""
        try:
            if 'close' not in df.columns:
                logging.error("عمود 'close' غير موجود في DataFrame")
                return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)
            
            close_prices = df['close'].astype(float)
            
            # حساب MACD باستخدام talib
            macd_line, signal_line, hist = talib.MACD(close_prices, 
                                                     fastperiod=fast, 
                                                     slowperiod=slow, 
                                                     signalperiod=signal)
            
            # تحويل القيم إلى نسب مئوية من السعر
            macd_pct = (macd_line / close_prices.shift(1)) * 100
            signal_pct = (signal_line / close_prices.shift(1)) * 100
            hist_pct = (hist / close_prices.shift(1)) * 100
            
            # تنظيف القيم
            macd_pct = pd.Series(macd_pct, index=df.index).fillna(0)
            signal_pct = pd.Series(signal_pct, index=df.index).fillna(0)
            hist_pct = pd.Series(hist_pct, index=df.index).fillna(0)
            
            return macd_pct, signal_pct, hist_pct
            
        except Exception as e:
            logging.error(f"خطأ في حساب مؤشر MACD: {str(e)}")
            return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> tuple:
        """حساب نطاقات بولينجر"""
        try:
            if 'close' not in df.columns:
                logging.error("عمود 'close' غير موجود في DataFrame")
                return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)
            
            close_prices = df['close'].astype(float)
            middle = close_prices.rolling(window=period).mean()
            std_dev = close_prices.rolling(window=period).std()
            
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            return upper, middle, lower
        except Exception as e:
            logging.error(f"خطأ في حساب نطاقات بولينجر: {str(e)}")
            return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)

    def calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """حساب مؤشر Stochastic"""
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logging.error("بعض الأعمدة المطلوبة غير موجودة في DataFrame")
                return pd.Series(index=df.index), pd.Series(index=df.index)
            
            low14 = df['low'].astype(float).rolling(window=period).min()
            high14 = df['high'].astype(float).rolling(window=period).max()
            
            k = ((df['close'].astype(float) - low14) / (high14 - low14)) * 100
            d = k.rolling(window=3).mean()
            
            return k, d
        except Exception as e:
            logging.error(f"خطأ في حساب مؤشر Stochastic: {str(e)}")
            return pd.Series(index=df.index), pd.Series(index=df.index)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """حساب مؤشر ADX"""
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logging.error("بعض الأعمدة المطلوبة غير موجودة في DataFrame")
                return pd.Series(index=df.index)
            
            df = df.astype(float)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            max_range = ranges.max(axis=1)
            
            plus_dm = df['high'] - df['high'].shift(1)
            minus_dm = df['low'].shift(1) - df['low']
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            plus_di = plus_dm.ewm(span=period, adjust=False).mean() / max_range.ewm(span=period, adjust=False).mean()
            minus_di = minus_dm.ewm(span=period, adjust=False).mean() / max_range.ewm(span=period, adjust=False).mean()
            
            dx = np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            return adx
        except Exception as e:
            logging.error(f"خطأ في حساب مؤشر ADX: {str(e)}")
            return pd.Series(index=df.index)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """حساب المدى الحقيقي المتوسط"""
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logging.error("بعض الأعمدة المطلوبة غير موجودة في DataFrame")
                return pd.Series(index=df.index)
            
            df = df.astype(float)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            max_range = ranges.max(axis=1)
            
            atr = max_range.ewm(span=period, adjust=False).mean()
            
            return atr
        except Exception as e:
            logging.error(f"خطأ في حساب المدى الحقيقي المتوسط: {str(e)}")
            return pd.Series(index=df.index)
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل المؤشرات الفنية وإرجاع الإشارات مع التحقق من القيم"""
        try:
            if df.empty:
                logging.warning("لا يمكن تحليل المؤشرات: البيانات فارغة")
                return {}
            
            # إضافة المؤشرات
            df = self.add_indicators(df)
            signals = {}
            
            # تحليل MACD
            if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                try:
                    last_row = df.iloc[-1]
                    macd = last_row['MACD']
                    signal = last_row['MACD_Signal']
                    hist = last_row['MACD_Hist']
                    
                    signals['MACD'] = {
                        'value': round(macd, 4),
                        'signal': round(signal, 4),
                        'histogram': round(hist, 4),
                        'trend': 'صاعد' if hist > 0 else 'هابط' if hist < 0 else 'متعادل'
                    }
                except Exception as e:
                    logging.error(f"خطأ في تحليل MACD: {str(e)}")
            
            # تحليل RSI
            if 'RSI' in df.columns:
                try:
                    rsi = df['RSI'].iloc[-1]
                    rsi_trend = df['RSI_Trend'].iloc[-1]
                    rsi_zone_duration = df['RSI_Zone_Duration'].iloc[-1]
                    
                    signals['RSI'] = {
                        'value': round(rsi, 2),
                        'trend': rsi_trend,
                        'zone_duration': rsi_zone_duration,
                        'status': 'ذروة شراء' if rsi > 70 
                                else 'ذروة بيع' if rsi < 30 
                                else 'متعادل'
                    }
                except Exception as e:
                    logging.error(f"خطأ في تحليل RSI: {str(e)}")
            
            return signals
            
        except Exception as e:
            logging.error(f"خطأ في تحليل المؤشرات: {str(e)}")
            return {}
            
    def _determine_macd_signal(self, macd: float, signal: float, hist: float) -> str:
        """تحديد إشارة MACD"""
        try:
            if abs(macd) < 0.0001 or abs(signal) < 0.0001:  # تجنب القيم الصغيرة جداً
                return 'متعادل'
                
            if hist > 0:
                return 'شراء' if hist > abs(macd * 0.1) else 'متعادل'  # يجب أن يكون الفرق كبير بما يكفي
            elif hist < 0:
                return 'بيع' if abs(hist) > abs(macd * 0.1) else 'متعادل'
            else:
                return 'متعادل'
                
        except Exception as e:
            logging.error(f"خطأ في تحديد إشارة MACD: {str(e)}")
            return 'متعادل'
            
    def _determine_ma_trend(self, price: float, sma_20: float, sma_50: float) -> str:
        """تحديد اتجاه المتوسطات المتحركة"""
        try:
            # حساب النسب المئوية للفروق
            diff_20 = ((price - sma_20) / sma_20) * 100
            diff_50 = ((price - sma_50) / sma_50) * 100
            ma_diff = ((sma_20 - sma_50) / sma_50) * 100
            
            # تحديد الاتجاه بناءً على الفروق
            if abs(diff_20) < 0.1 and abs(diff_50) < 0.1:  # تجنب القيم الصغيرة جداً
                return 'متعادل'
                
            if diff_20 > 0.5 and diff_50 > 0.5 and ma_diff > 0:
                return 'صعود قوي'
            elif diff_20 < -0.5 and diff_50 < -0.5 and ma_diff < 0:
                return 'هبوط قوي'
            elif diff_20 > 0.2 or (ma_diff > 0 and diff_50 > 0):
                return 'صعود'
            elif diff_20 < -0.2 or (ma_diff < 0 and diff_50 < 0):
                return 'هبوط'
            else:
                return 'متعادل'
                
        except Exception as e:
            logging.error(f"خطأ في تحديد اتجاه المتوسطات المتحركة: {str(e)}")
            return 'متعادل'

    def analyze_signals(self, df: pd.DataFrame) -> dict:
        """تحليل وتقييم إشارات التداول"""
        try:
            # تحليل كل مؤشر
            analyses = {
                'MA': self._analyze_ma_signals(df),
                'RSI': self._analyze_rsi_signals(df),
                'MACD': self._analyze_macd_signals(df),
                'Volume': self._analyze_volume_signals(df)
            }
            
            # التحقق من اتجاه السعر الحالي
            current_trend = 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-5] else 'bearish'
            
            # حساب عدد المؤشرات المتوافقة مع كل اتجاه
            bullish_count = sum(1 for a in analyses.values() if a['direction'] == 'bullish')
            bearish_count = sum(1 for a in analyses.values() if a['direction'] == 'bearish')
            
            # حساب القوة المرجحة مع الأخذ في الاعتبار توافق المؤشرات
            total_weight = sum(a['weight'] for a in analyses.values())
            base_strength = sum(
                a['strength'] * a['weight'] 
                for a in analyses.values()
            ) / total_weight if total_weight > 0 else 0
            
            # تعديل القوة بناءً على توافق المؤشرات
            indicator_agreement = max(bullish_count, bearish_count) / len(analyses)
            adjusted_strength = base_strength * indicator_agreement
            
            # حساب مستوى الثقة
            confidence = (max(bullish_count, bearish_count) / len(analyses)) * 100
            
            # تحديد الاتجاه النهائي
            if bullish_count > bearish_count:
                final_direction = 'bullish'
            elif bearish_count > bullish_count:
                final_direction = 'bearish'
            else:
                final_direction = current_trend
            
            # التحقق من تأكيد الاتجاه
            trend_confirmed = (
                (final_direction == 'bullish' and current_trend == 'bullish') or
                (final_direction == 'bearish' and current_trend == 'bearish')
            )
            
            # تخفيض القوة إذا كان الاتجاه غير مؤكد
            if not trend_confirmed:
                adjusted_strength *= 0.7
                confidence *= 0.7
            
            # فلترة الإشارات الضعيفة
            if adjusted_strength < 30 or confidence < 60:
                return {
                    'decision': 'محايد',
                    'strength': adjusted_strength,
                    'confidence': confidence,
                    'reasons': ['الإشارات ضعيفة أو غير مؤكدة']
                }
            
            # تجميع أسباب القرار
            active_signals = []
            for name, analysis in analyses.items():
                if analysis['strength'] >= 30:
                    direction_ar = 'صاعد' if analysis['direction'] == 'bullish' else 'هابط'
                    active_signals.append(
                        f"{name} ({direction_ar}, قوة: {analysis['strength']:.1f}%)"
                    )
            
            # تحديد القرار النهائي
            if final_direction == 'bullish':
                if adjusted_strength >= 80 and confidence >= 80 and trend_confirmed:
                    decision = 'شراء قوي'
                elif adjusted_strength >= 50:
                    decision = 'شراء'
                else:
                    decision = 'محايد'
            else:
                if adjusted_strength >= 80 and confidence >= 80 and trend_confirmed:
                    decision = 'بيع قوي'
                elif adjusted_strength >= 50:
                    decision = 'بيع'
                else:
                    decision = 'محايد'
            
            return {
                'decision': decision,
                'strength': adjusted_strength,
                'confidence': confidence,
                'reasons': active_signals
            }
            
        except Exception as e:
            logging.error(f"خطأ في تحليل الإشارات: {str(e)}")
            return {
                'decision': 'خطأ',
                'strength': 0,
                'confidence': 0,
                'reasons': [f'حدث خطأ: {str(e)}']
            }
            
    def _analyze_ma_signals(self, df: pd.DataFrame) -> dict:
        """تحليل إشارات المتوسطات المتحركة"""
        try:
            signals = []
            close = df['close'].values
            volume = df['volume'].values
            volume_ma = df['volume'].rolling(window=20).mean().values
            
            for period in [20, 50]:
                ma = df[f'EMA_{period}'].values
                
                # حساب قوة التقاطع والاتجاه
                for i in range(5, len(df)):
                    price_change = (close[i] - close[i-5]) / close[i-5] * 100
                    volume_factor = volume[i] / volume_ma[i]
                    
                    # حساب قوة الإشارة
                    distance = abs(close[i] - ma[i]) / close[i] * 100
                    trend_strength = abs(price_change) * (1 + (volume_factor - 1) * 0.5)
                    signal_strength = min(100, max(20, (distance + trend_strength) * 0.7))
                    
                    if close[i] > ma[i] and close[i-1] <= ma[i-1]:
                        signals.append(('bullish', signal_strength))
                    elif close[i] < ma[i] and close[i-1] >= ma[i-1]:
                        signals.append(('bearish', signal_strength))
            
            if not signals:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.3}
            
            # تحديد الاتجاه والقوة النهائية
            bullish_signals = [s[1] for s in signals if s[0] == 'bullish']
            bearish_signals = [s[1] for s in signals if s[0] == 'bearish']
            
            if len(bullish_signals) > len(bearish_signals):
                return {
                    'direction': 'bullish',
                    'strength': sum(bullish_signals) / len(bullish_signals),
                    'weight': 0.3
                }
            elif len(bearish_signals) > len(bullish_signals):
                return {
                    'direction': 'bearish',
                    'strength': sum(bearish_signals) / len(bearish_signals),
                    'weight': 0.3
                }
            else:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.3}
                
        except Exception as e:
            logging.error(f"خطأ في تحليل المتوسطات المتحركة: {str(e)}")
            return {'direction': 'neutral', 'strength': 0, 'weight': 0.3}

    def _analyze_rsi_signals(self, df: pd.DataFrame) -> dict:
        """تحليل إشارات مؤشر RSI"""
        try:
            rsi = df['RSI'].values
            volume = df['volume'].values
            volume_ma = df['volume'].rolling(window=20).mean().values
            close = df['close'].values
            
            signals = []
            for i in range(5, len(df)):
                price_change = (close[i] - close[i-5]) / close[i-5] * 100
                volume_factor = volume[i] / volume_ma[i]
                
                # حساب قوة الإشارة
                if rsi[i] < 30:
                    distance = (30 - rsi[i])
                    trend_strength = abs(price_change) * (1 + (volume_factor - 1) * 0.5)
                    signal_strength = min(100, max(20, (distance * 3 + trend_strength) * 0.7))
                    signals.append(('bullish', signal_strength))
                elif rsi[i] > 70:
                    distance = (rsi[i] - 70)
                    trend_strength = abs(price_change) * (1 + (volume_factor - 1) * 0.5)
                    signal_strength = min(100, max(20, (distance * 3 + trend_strength) * 0.7))
                    signals.append(('bearish', signal_strength))
            
            if not signals:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.25}
            
            # تحديد الاتجاه والقوة النهائية
            bullish_signals = [s[1] for s in signals if s[0] == 'bullish']
            bearish_signals = [s[1] for s in signals if s[0] == 'bearish']
            
            if len(bullish_signals) > len(bearish_signals):
                return {
                    'direction': 'bullish',
                    'strength': sum(bullish_signals) / len(bullish_signals),
                    'weight': 0.25
                }
            elif len(bearish_signals) > len(bullish_signals):
                return {
                    'direction': 'bearish',
                    'strength': sum(bearish_signals) / len(bearish_signals),
                    'weight': 0.25
                }
            else:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.25}
                
        except Exception as e:
            logging.error(f"خطأ في تحليل RSI: {str(e)}")
            return {'direction': 'neutral', 'strength': 0, 'weight': 0.25}

    def _analyze_macd_signals(self, df: pd.DataFrame) -> dict:
        """تحليل إشارات مؤشر MACD"""
        try:
            macd = df['MACD'].values
            signal = df['MACD_Signal'].values
            hist = df['MACD_Hist'].values
            volume = df['volume'].values
            volume_ma = df['volume'].rolling(window=20).mean().values
            close = df['close'].values
            
            signals = []
            for i in range(5, len(df)):
                price_change = (close[i] - close[i-5]) / close[i-5] * 100
                volume_factor = volume[i] / volume_ma[i]
                
                # حساب قوة الإشارة
                if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                    hist_strength = abs(hist[i]) / abs(macd[i]) * 100 if macd[i] != 0 else 50
                    trend_strength = abs(price_change) * (1 + (volume_factor - 1) * 0.5)
                    signal_strength = min(100, max(20, (hist_strength + trend_strength) * 0.7))
                    signals.append(('bullish', signal_strength))
                elif macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                    hist_strength = abs(hist[i]) / abs(macd[i]) * 100 if macd[i] != 0 else 50
                    trend_strength = abs(price_change) * (1 + (volume_factor - 1) * 0.5)
                    signal_strength = min(100, max(20, (hist_strength + trend_strength) * 0.7))
                    signals.append(('bearish', signal_strength))
            
            if not signals:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.25}
            
            # تحديد الاتجاه والقوة النهائية
            bullish_signals = [s[1] for s in signals if s[0] == 'bullish']
            bearish_signals = [s[1] for s in signals if s[0] == 'bearish']
            
            if len(bullish_signals) > len(bearish_signals):
                return {
                    'direction': 'bullish',
                    'strength': sum(bullish_signals) / len(bullish_signals),
                    'weight': 0.25
                }
            elif len(bearish_signals) > len(bullish_signals):
                return {
                    'direction': 'bearish',
                    'strength': sum(bearish_signals) / len(bearish_signals),
                    'weight': 0.25
                }
            else:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.25}
                
        except Exception as e:
            logging.error(f"خطأ في تحليل MACD: {str(e)}")
            return {'direction': 'neutral', 'strength': 0, 'weight': 0.25}

    def _analyze_volume_signals(self, df: pd.DataFrame) -> dict:
        """تحليل إشارات الحجم"""
        try:
            volume = df['volume'].values
            close = df['close'].values
            volume_ma = df['volume'].rolling(window=20).mean().values
            
            signals = []
            for i in range(5, len(df)):
                if volume[i] > volume_ma[i] * 1.5:
                    price_change = (close[i] - close[i-5]) / close[i-5] * 100
                    
                    # حساب قوة الإشارة
                    signal_strength = min(100, max(20, (price_change * 0.7)))
                    
                    if price_change > 0:
                        signals.append(('bullish', signal_strength))
                    elif price_change < 0:
                        signals.append(('bearish', signal_strength))
            
            if not signals:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.2}
            
            # تحديد الاتجاه والقوة النهائية
            bullish_signals = [s[1] for s in signals if s[0] == 'bullish']
            bearish_signals = [s[1] for s in signals if s[0] == 'bearish']
            
            if len(bullish_signals) > len(bearish_signals):
                return {
                    'direction': 'bullish',
                    'strength': sum(bullish_signals) / len(bullish_signals),
                    'weight': 0.2
                }
            elif len(bearish_signals) > len(bullish_signals):
                return {
                    'direction': 'bearish',
                    'strength': sum(bearish_signals) / len(bearish_signals),
                    'weight': 0.2
                }
            else:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.2}
                
        except Exception as e:
            logging.error(f"خطأ في تحليل الحجم: {str(e)}")
            return {'direction': 'neutral', 'strength': 0, 'weight': 0.2}
            
            signals = []
            for i in range(5, len(df)):
                if volume[i] > volume_ma[i] * 1.5:
                    # حساب عدد الشموع المؤكدة للاتجاه
                    confirmation_count = 0
                    price_direction = close[i] > close[i-1]
                    for j in range(max(0, i-5), i+1):
                        if (close[j] > close[j-1]) == price_direction:
                            confirmation_count += 1
                    
                    confirmation_factor = confirmation_count / 6
                    volume_ratio = volume[i] / volume_ma[i]
                    
                    if close[i] > close[i-1]:
                        # حساب قوة إشارة الشراء
                        volume_strength = (volume_ratio - 1.5) * 30
                        signal_strength = min(100, max(20,
                            (volume_strength * 0.4 + trend_strength * 0.3 + confirmation_factor * 30) * sr_factor
                        ))
                        signals.append(('bullish', signal_strength))
                        
                    else:
                        # حساب قوة إشارة البيع
                        volume_strength = (volume_ratio - 1.5) * 30
                        signal_strength = min(100, max(20,
                            (volume_strength * 0.4 + trend_strength * 0.3 + confirmation_factor * 30) * sr_factor
                        ))
                        signals.append(('bearish', signal_strength))
            
            if not signals:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.2}
            
            # تحديد الاتجاه والقوة النهائية
            recent_signals = [(d, s) for d, s in signals[-3:]]  # آخر 3 إشارات فقط
            bullish_signals = [s for d, s in recent_signals if d == 'bullish']
            bearish_signals = [s for d, s in recent_signals if d == 'bearish']
            
            if len(bullish_signals) > len(bearish_signals):
                return {
                    'direction': 'bullish',
                    'strength': sum(bullish_signals) / len(bullish_signals) if bullish_signals else 0,
                    'weight': 0.2
                }
            elif len(bearish_signals) > len(bullish_signals):
                return {
                    'direction': 'bearish',
                    'strength': sum(bearish_signals) / len(bearish_signals) if bearish_signals else 0,
                    'weight': 0.2
                }
            else:
                return {'direction': 'neutral', 'strength': 0, 'weight': 0.2}
                
        except Exception as e:
            logging.error(f"خطأ في تحليل الحجم: {str(e)}")
            return {'direction': 'neutral', 'strength': 0, 'weight': 0.2}
