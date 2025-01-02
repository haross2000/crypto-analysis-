"""
محلل أنماط الشموع اليابانية والأنماط الفنية
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.signal import argrelextrema
from dataclasses import dataclass
import talib

@dataclass
class Pattern:
    name: str
    direction: str
    strength: float
    description: str
    price_target: float = 0.0
    stop_loss: float = 0.0

class PatternAnalyzer:
    """فئة تحليل أنماط الشموع اليابانية والأنماط الفنية"""
    
    def __init__(self):
        """تهيئة محلل الأنماط"""
        logging.info("تم تهيئة محلل أنماط الشموع والأنماط الفنية")
        
        # إضافة معايير التطبيع
        self.normalization = {
            'MACD': {
                'max_value': 2.0,  # القيمة القصوى المتوقعة للMACD
                'min_value': -2.0  # القيمة الدنيا المتوقعة للMACD
            },
            'Pattern': {
                'strength_factors': {
                    'trend_alignment': 0.3,    # توافق النمط مع الاتجاه
                    'volume_confirmation': 0.3, # تأكيد الحجم
                    'price_level': 0.2,        # مستوى السعر
                    'formation_quality': 0.2    # جودة تشكل النمط
                }
            }
        }
        
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """التحقق من صحة البيانات"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_columns)
    
    def analyze(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        تحليل أنماط الشموع والأنماط الفنية
        
        المعلمات:
            df: DataFrame مع بيانات OHLCV
            
        العوائد:
            List[Dict]: قائمة بالأنماط المكتشفة
        """
        try:
            if not self._validate_dataframe(df):
                logging.error("البيانات غير صحيحة أو ناقصة")
                return []
                
            if df.empty or len(df) < 300:  # نحتاج 300 شمعة للتحليل
                logging.warning("لا توجد بيانات كافية للتحليل (مطلوب 300 شمعة)")
                return []
                
            patterns = []
            current_price = df.iloc[-1]['close']
            
            try:
                # تحليل أنماط الشموع اليابانية
                candlestick_patterns = self._analyze_candlestick_patterns(df)  # استخدام كل البيانات
                for pattern in candlestick_patterns:
                    pattern_type = 'محايد'
                    if pattern.direction == 'صعود':
                        pattern_type = 'bullish'
                    elif pattern.direction == 'هبوط':
                        pattern_type = 'bearish'
                    patterns.append({
                        'type': pattern_type,
                        'name': pattern.name,
                        'strength': pattern.strength,
                        'price': pattern.price_target or current_price,
                        'description': pattern.description
                    })
                
                # تحليل الأنماط الفنية
                chart_patterns = self._analyze_chart_patterns(df)  # استخدام كل البيانات
                for pattern in chart_patterns:
                    pattern_type = 'محايد'
                    if pattern.direction == 'صعود':
                        pattern_type = 'bullish'
                    elif pattern.direction == 'هبوط':
                        pattern_type = 'bearish'
                    patterns.append({
                        'type': pattern_type,
                        'name': pattern.name,
                        'strength': pattern.strength,
                        'price': pattern.price_target or current_price,
                        'description': pattern.description
                    })
                    
            except Exception as e:
                logging.error(f"خطأ في تحليل الأنماط: {str(e)}")
                return []
            
            # تحليل مستويات الدعم والمقاومة
            support_resistance = self._analyze_support_resistance(df)
            for pattern in support_resistance:
                pattern_type = 'محايد'
                if pattern.direction == 'صعود':
                    pattern_type = 'bullish'
                elif pattern.direction == 'هبوط':
                    pattern_type = 'bearish'
                patterns.append({
                    'type': pattern_type,
                    'name': pattern.name,
                    'strength': pattern.strength,
                    'price': pattern.price_target or current_price,
                    'description': pattern.description
                })
            
            return patterns
            
        except Exception as e:
            logging.error(f"خطأ في تحليل الأنماط: {str(e)}")
            return []
            
    def _normalize_macd(self, macd_value: float) -> float:
        """تطبيع قيمة MACD إلى نطاق 0-1"""
        max_val = self.normalization['MACD']['max_value']
        min_val = self.normalization['MACD']['min_value']
        
        # تقييد القيمة ضمن النطاق
        macd_value = max(min(macd_value, max_val), min_val)
        
        # تطبيع القيمة إلى نطاق 0-1
        normalized = (macd_value - min_val) / (max_val - min_val)
        return normalized

    def _calculate_pattern_strength(self, pattern_type: str, df: pd.DataFrame, pattern_data: dict) -> float:
        """حساب قوة النمط بشكل ديناميكي"""
        try:
            factors = self.normalization['Pattern']['strength_factors']
            strength = 0.0
            
            # 1. توافق النمط مع الاتجاه (30%)
            trend = self._calculate_trend(df)
            if pattern_type == 'bullish' and trend > 0:
                trend_score = min(abs(trend), 1.0)
            elif pattern_type == 'bearish' and trend < 0:
                trend_score = min(abs(trend), 1.0)
            else:
                trend_score = 0.0
            strength += trend_score * factors['trend_alignment']
            
            # 2. تأكيد الحجم (30%)
            volume_score = self._confirm_volume(df, pattern_type)
            strength += volume_score * factors['volume_confirmation']
            
            # 3. مستوى السعر (20%)
            price_score = self._check_price_level(df, pattern_data)
            strength += price_score * factors['price_level']
            
            # 4. جودة تشكل النمط (20%)
            quality_score = self._check_pattern_quality(pattern_data)
            strength += quality_score * factors['formation_quality']
            
            # تقريب النتيجة إلى أقرب 5%
            return round(strength * 100 / 5) * 5
            
        except Exception as e:
            logging.error(f"خطأ في حساب قوة النمط: {str(e)}")
            return 50.0  # قيمة افتراضية محايدة

    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """حساب الاتجاه باستخدام المتوسطات المتحركة"""
        try:
            # حساب EMA 20 و EMA 50
            ema_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            
            # حساب نسبة الفرق بين EMA 20 و EMA 50
            trend_strength = (ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
            
            return trend_strength
            
        except Exception as e:
            logging.error(f"خطأ في حساب الاتجاه: {str(e)}")
            return 0.0

    def _confirm_volume(self, df: pd.DataFrame, pattern_type: str) -> float:
        """التحقق من تأكيد الحجم للنمط"""
        try:
            # حساب متوسط الحجم للـ 20 فترة السابقة
            avg_volume = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            
            # حساب نسبة الحجم الحالي إلى المتوسط
            volume_ratio = current_volume / avg_volume.iloc[-1]
            
            # تطبيع النسبة إلى نطاق 0-1
            volume_score = min(volume_ratio / 2, 1.0)
            
            return volume_score
            
        except Exception as e:
            logging.error(f"خطأ في التحقق من الحجم: {str(e)}")
            return 0.5

    def _check_price_level(self, df: pd.DataFrame, pattern_data: dict) -> float:
        """التحقق من مستوى السعر"""
        try:
            # حساب مستويات الدعم والمقاومة
            highs = df['high'].rolling(window=20).max()
            lows = df['low'].rolling(window=20).min()
            current_price = df['close'].iloc[-1]
            
            # حساب الموقع النسبي للسعر
            price_range = highs.iloc[-1] - lows.iloc[-1]
            if price_range == 0:
                return 0.5
                
            relative_position = (current_price - lows.iloc[-1]) / price_range
            
            # تعديل النتيجة بناءً على نوع النمط
            if pattern_data.get('type') == 'bullish':
                score = 1 - relative_position  # أفضل للشراء عندما يكون السعر منخفض
            else:
                score = relative_position  # أفضل للبيع عندما يكون السعر مرتفع
                
            return score
            
        except Exception as e:
            logging.error(f"خطأ في التحقق من مستوى السعر: {str(e)}")
            return 0.5

    def _check_pattern_quality(self, pattern_data: dict) -> float:
        """التحقق من جودة تشكل النمط"""
        try:
            # معايير الجودة
            criteria = {
                'size': 0.4,        # حجم النمط
                'symmetry': 0.3,     # تناسق النمط
                'completion': 0.3    # اكتمال النمط
            }
            
            quality_score = 0.0
            
            # حساب درجة الحجم
            size_score = min(pattern_data.get('size', 0.5), 1.0)
            quality_score += size_score * criteria['size']
            
            # حساب درجة التناسق
            symmetry_score = pattern_data.get('symmetry', 0.5)
            quality_score += symmetry_score * criteria['symmetry']
            
            # حساب درجة الاكتمال
            completion_score = pattern_data.get('completion', 1.0)
            quality_score += completion_score * criteria['completion']
            
            return quality_score
            
        except Exception as e:
            logging.error(f"خطأ في التحقق من جودة النمط: {str(e)}")
            return 0.5

    def _analyze_japanese_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """تحليل أنماط الشموع اليابانية مع تطبيع القيم"""
        patterns = []
        
        # تحليل MACD
        macd, signal, hist = talib.MACD(df['close'])
        
        # تطبيع قيم MACD
        normalized_macd = self._normalize_macd(macd.iloc[-1])
        normalized_signal = self._normalize_macd(signal.iloc[-1])
        normalized_hist = self._normalize_macd(hist.iloc[-1])
        
        patterns.append({
            'type': 'MACD',
            'values': {
                'MACD': normalized_macd,
                'Signal': normalized_signal,
                'Histogram': normalized_hist
            },
            'strength': abs(normalized_hist)
        })
        
        # تحليل الأنماط الأخرى...
        return patterns
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """تحليل أنماط الشموع اليابانية"""
        patterns = []
        last_candles = df.iloc[-3:]  # آخر 3 شموع
        
        # تحليل نمط الدوجي
        doji = self._detect_doji(df.iloc[-1])
        if doji.strength > 0:
            patterns.append(doji)
        
        # تحليل نمط المطرقة والمطرقة المقلوبة
        hammer = self._detect_hammer(last_candles)
        if hammer.strength > 0:
            patterns.append(hammer)
        
        # تحليل نمط الابتلاع
        engulfing = self._detect_engulfing(last_candles)
        if engulfing.strength > 0:
            patterns.append(engulfing)
        
        # تحليل نمط نجمة المساء والصباح
        star = self._detect_star_pattern(last_candles)
        if star.strength > 0:
            patterns.append(star)
        
        return patterns
    
    def _analyze_chart_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """تحليل الأنماط الفنية"""
        patterns = []
        
        # تحليل نمط الرأس والكتفين
        head_shoulders = self._detect_head_shoulders(df)
        if head_shoulders.strength > 0:
            patterns.append(head_shoulders)
        
        # تحليل نمط المثلث
        triangle = self._detect_triangle(df)
        if triangle.strength > 0:
            patterns.append(triangle)
        
        # تحليل نمط القمة/القاع المزدوج
        double_pattern = self._detect_double_pattern(df)
        if double_pattern.strength > 0:
            patterns.append(double_pattern)
        
        # تحليل نمط العلم
        flag = self._detect_flag(df)
        if flag.strength > 0:
            patterns.append(flag)
        
        return patterns
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> List[Pattern]:
        """تحليل مستويات الدعم والمقاومة"""
        patterns = []
        
        # تحديد القمم والقيعان المحلية
        n = 5  # عدد الشموع للمقارنة
        highs = df['high'].values
        lows = df['low'].values
        
        # العثور على القمم المحلية
        peak_idx = argrelextrema(highs, np.greater, order=n)[0]
        peaks = highs[peak_idx]
        
        # العثور على القيعان المحلية
        valley_idx = argrelextrema(lows, np.less, order=n)[0]
        valleys = lows[valley_idx]
        
        current_price = df.iloc[-1]['close']
        
        # تحديد أقرب مستويات الدعم والمقاومة
        resistance_levels = peaks[peaks > current_price]
        support_levels = valleys[valleys < current_price]
        
        if len(resistance_levels) > 0:
            nearest_resistance = resistance_levels.min()
            patterns.append(Pattern(
                name='مستوى مقاومة',
                direction='هبوط',
                strength=0.8,
                description=f'مستوى مقاومة قوي عند {nearest_resistance:.2f}',
                price_target=nearest_resistance,
                stop_loss=current_price + (nearest_resistance - current_price) * 0.1
            ))
        
        if len(support_levels) > 0:
            nearest_support = support_levels.max()
            patterns.append(Pattern(
                name='مستوى دعم',
                direction='صعود',
                strength=0.8,
                description=f'مستوى دعم قوي عند {nearest_support:.2f}',
                price_target=nearest_support,
                stop_loss=current_price - (current_price - nearest_support) * 0.1
            ))
        
        return patterns
    
    def _detect_doji(self, candle: pd.Series) -> Pattern:
        """تحليل نمط الدوجي"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            avg_shadow = (upper_shadow + lower_shadow) / 2
            
            if body_size <= avg_shadow * 0.1:  # جسم صغير جداً مقارنة بالظلال
                return Pattern(
                    name='دوجي',
                    direction='محايد',
                    strength=0.6,
                    description='نمط دوجي يشير إلى تردد السوق',
                    price_target=candle['close']
                )
            
            return Pattern(
                name='',
                direction='',
                strength=0,
                description=''
            )
            
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط الدوجي: {str(e)}")
            return Pattern('', '', 0, '')
            
    def _detect_hammer(self, candles: pd.DataFrame) -> Pattern:
        """تحليل نمط المطرقة والمطرقة المقلوبة"""
        try:
            last_candle = candles.iloc[-1]
            body_size = abs(last_candle['close'] - last_candle['open'])
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
            
            # المطرقة: ظل سفلي طويل وظل علوي قصير
            if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                return Pattern(
                    name='مطرقة',
                    direction='صعود',
                    strength=0.7,
                    description='نمط مطرقة صعودي',
                    price_target=last_candle['close'] + body_size
                )
            
            # المطرقة المقلوبة: ظل علوي طويل وظل سفلي قصير
            if upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
                return Pattern(
                    name='مطرقة مقلوبة',
                    direction='هبوط',
                    strength=0.7,
                    description='نمط مطرقة مقلوبة هبوطي',
                    price_target=last_candle['close'] - body_size
                )
            
            return Pattern('', '', 0, '')
            
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط المطرقة: {str(e)}")
            return Pattern('', '', 0, '')
            
    def _detect_engulfing(self, candles: pd.DataFrame) -> Pattern:
        """تحليل نمط الابتلاع"""
        try:
            if len(candles) < 2:
                return Pattern('', '', 0, '')
                
            current = candles.iloc[-1]
            previous = candles.iloc[-2]
            
            current_body = abs(current['close'] - current['open'])
            previous_body = abs(previous['close'] - previous['open'])
            
            # نمط ابتلاع صعودي
            if (previous['close'] < previous['open'] and  # الشمعة السابقة حمراء
                current['close'] > current['open'] and    # الشمعة الحالية خضراء
                current_body > previous_body and          # جسم الشمعة الحالية أكبر
                current['open'] < previous['close'] and   # الشمعة الحالية تبتلع السابقة
                current['close'] > previous['open']):
                return Pattern(
                    name='ابتلاع صعودي',
                    direction='صعود',
                    strength=0.8,
                    description='نمط ابتلاع صعودي قوي',
                    price_target=current['close'] + current_body
                )
            
            # نمط ابتلاع هبوطي
            if (previous['close'] > previous['open'] and  # الشمعة السابقة خضراء
                current['close'] < current['open'] and    # الشمعة الحالية حمراء
                current_body > previous_body and          # جسم الشمعة الحالية أكبر
                current['open'] > previous['close'] and   # الشمعة الحالية تبتلع السابقة
                current['close'] < previous['open']):
                return Pattern(
                    name='ابتلاع هبوطي',
                    direction='هبوط',
                    strength=0.8,
                    description='نمط ابتلاع هبوطي قوي',
                    price_target=current['close'] - current_body
                )
            
            return Pattern('', '', 0, '')
            
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط الابتلاع: {str(e)}")
            return Pattern('', '', 0, '')
            
    def _detect_star_pattern(self, candles: pd.DataFrame) -> Pattern:
        """تحليل نمط نجمة المساء والصباح"""
        if len(candles) < 3:
            return Pattern('', '', 0, '')
            
        first = candles.iloc[-3]
        second = candles.iloc[-2]
        third = candles.iloc[-1]
        
        # نجمة الصباح
        if (first['close'] < first['open'] and  # شمعة هبوطية
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # شمعة صغيرة
            third['close'] > third['open'] and  # شمعة صعودية
            second['low'] < first['low']):  # فجوة هبوطية
            return Pattern(
                name='نجمة الصباح',
                direction='صعود',
                strength=0.9,
                description='نمط نجمة الصباح يشير إلى احتمال انعكاس صعودي قوي',
                price_target=third['close'] + abs(third['close'] - third['open'])
            )
        
        # نجمة المساء
        if (first['close'] > first['open'] and  # شمعة صعودية
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # شمعة صغيرة
            third['close'] < third['open'] and  # شمعة هبوطية
            second['high'] > first['high']):  # فجوة صعودية
            return Pattern(
                name='نجمة المساء',
                direction='هبوط',
                strength=0.9,
                description='نمط نجمة المساء يشير إلى احتمال انعكاس هبوطي قوي',
                price_target=third['close'] - abs(third['close'] - third['open'])
            )
        
        return Pattern('', '', 0, '')
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Pattern:
        """تحليل نمط الرأس والكتفين"""
        if len(df) < 30:
            return Pattern('', '', 0, '')
            
        # تحديد القمم المحلية
        highs = df['high'].values
        peak_idx = argrelextrema(highs, np.greater, order=5)[0]
        
        if len(peak_idx) < 3:
            return Pattern('', '', 0, '')
        
        # البحث عن نمط الرأس والكتفين
        for i in range(len(peak_idx)-2):
            left_shoulder = highs[peak_idx[i]]
            head = highs[peak_idx[i+1]]
            right_shoulder = highs[peak_idx[i+2]]
            
            # التحقق من شروط النمط
            if (head > left_shoulder and 
                head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                
                neckline = min(df['low'].iloc[peak_idx[i]:peak_idx[i+2]+1])
                current_price = df.iloc[-1]['close']
                
                if current_price < neckline:
                    target = neckline - (head - neckline)
                    return Pattern(
                        name='رأس وكتفين',
                        direction='هبوط',
                        strength=0.9,
                        description='نمط رأس وكتفين مكتمل يشير إلى اتجاه هبوطي قوي',
                        price_target=target,
                        stop_loss=head
                    )
        
        return Pattern('', '', 0, '')
    
    def _detect_triangle(self, df: pd.DataFrame) -> Pattern:
        """تحليل نمط المثلث"""
        if len(df) < 20:
            return Pattern('', '', 0, '')
            
        highs = df['high'].values
        lows = df['low'].values
        
        # حساب خط الاتجاه العلوي والسفلي
        x = np.arange(len(df))
        high_slope, high_intercept = np.polyfit(x[-20:], highs[-20:], 1)
        low_slope, low_intercept = np.polyfit(x[-20:], lows[-20:], 1)
        
        # التحقق من تقارب الخطوط
        if abs(high_slope - low_slope) < 0.1:
            current_price = df.iloc[-1]['close']
            breakout_level = high_slope * len(df) + high_intercept
            
            if current_price > breakout_level:
                return Pattern(
                    name='مثلث متماثل',
                    direction='صعود',
                    strength=0.8,
                    description='اختراق صعودي لنمط المثلث المتماثل',
                    price_target=breakout_level + (breakout_level - (low_slope * len(df) + low_intercept)),
                    stop_loss=low_slope * len(df) + low_intercept
                )
            elif current_price < low_slope * len(df) + low_intercept:
                return Pattern(
                    name='مثلث متماثل',
                    direction='هبوط',
                    strength=0.8,
                    description='اختراق هبوطي لنمط المثلث المتماثل',
                    price_target=low_slope * len(df) + low_intercept - (breakout_level - (low_slope * len(df) + low_intercept)),
                    stop_loss=breakout_level
                )
        
        return Pattern('', '', 0, '')
    
    def _detect_double_pattern(self, df: pd.DataFrame) -> Pattern:
        """تحليل نمط القمة/القاع المزدوج"""
        if len(df) < 20:
            return Pattern('', '', 0, '')
            
        highs = df['high'].values
        lows = df['low'].values
        
        # تحديد القمم والقيعان المحلية
        peak_idx = argrelextrema(highs, np.greater, order=5)[0]
        valley_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(peak_idx) >= 2:
            # البحث عن قمة مزدوجة
            last_two_peaks = highs[peak_idx[-2:]]
            if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.02:
                neckline = min(df['low'].iloc[peak_idx[-2]:peak_idx[-1]])
                current_price = df.iloc[-1]['close']
                
                if current_price < neckline:
                    target = neckline - (last_two_peaks[0] - neckline)
                    return Pattern(
                        name='قمة مزدوجة',
                        direction='هبوط',
                        strength=0.9,
                        description='نمط قمة مزدوجة مكتمل يشير إلى اتجاه هبوطي',
                        price_target=target,
                        stop_loss=max(last_two_peaks)
                    )
        
        if len(valley_idx) >= 2:
            # البحث عن قاع مزدوج
            last_two_valleys = lows[valley_idx[-2:]]
            if abs(last_two_valleys[0] - last_two_valleys[1]) / last_two_valleys[0] < 0.02:
                neckline = max(df['high'].iloc[valley_idx[-2]:valley_idx[-1]])
                current_price = df.iloc[-1]['close']
                
                if current_price > neckline:
                    target = neckline + (neckline - last_two_valleys[0])
                    return Pattern(
                        name='قاع مزدوج',
                        direction='صعود',
                        strength=0.9,
                        description='نمط قاع مزدوج مكتمل يشير إلى اتجاه صعودي',
                        price_target=target,
                        stop_loss=min(last_two_valleys)
                    )
        
        return Pattern('', '', 0, '')
    
    def _detect_flag(self, df: pd.DataFrame) -> Pattern:
        """تحليل نمط العلم"""
        if len(df) < 20:
            return Pattern('', '', 0, '')
            
        # حساب الاتجاه العام
        x = np.arange(len(df))
        slope, intercept = np.polyfit(x[-20:], df['close'].values[-20:], 1)
        
        # حساب التذبذب
        std = df['close'].rolling(5).std().iloc[-1]
        avg_std = df['close'].rolling(5).std().mean()
        
        if std < avg_std * 0.7:  # تضييق في التذبذب
            if slope > 0:
                return Pattern(
                    name='علم صعودي',
                    direction='صعود',
                    strength=0.7,
                    description='نمط علم في اتجاه صعودي',
                    price_target=df.iloc[-1]['close'] * 1.1,
                    stop_loss=df.iloc[-1]['close'] * 0.95
                )
            elif slope < 0:
                return Pattern(
                    name='علم هبوطي',
                    direction='هبوط',
                    strength=0.7,
                    description='نمط علم في اتجاه هبوطي',
                    price_target=df.iloc[-1]['close'] * 0.9,
                    stop_loss=df.iloc[-1]['close'] * 1.05
                )
        
        return Pattern('', '', 0, '')
