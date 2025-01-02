import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Pattern:
    name: str
    start_idx: int
    end_idx: int
    pattern_type: str  # bullish/bearish
    confidence: float
    target_price: float
    stop_loss: float

class ChartPatternAnalyzer:
    def __init__(self):
        self.patterns_config = {
            'double_top': {'min_distance': 10, 'max_distance': 50, 'tolerance': 0.02},
            'double_bottom': {'min_distance': 10, 'max_distance': 50, 'tolerance': 0.02},
            'head_shoulders': {'min_distance': 20, 'max_distance': 60, 'tolerance': 0.03},
            'inverse_head_shoulders': {'min_distance': 20, 'max_distance': 60, 'tolerance': 0.03},
            'triangle': {'min_points': 5, 'max_distance': 50, 'tolerance': 0.02},
            'wedge': {'min_points': 5, 'max_distance': 50, 'tolerance': 0.02},
            'channel': {'min_points': 4, 'max_distance': 40, 'tolerance': 0.02},
            'flag': {'min_points': 4, 'max_distance': 20, 'tolerance': 0.02}
        }
    
    def analyze_all_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """تحليل جميع الأنماط المعروفة في البيانات"""
        patterns = []
        
        # تحليل القمم والقيعان
        peaks, troughs = self._find_peaks_and_troughs(df)
        
        # البحث عن الأنماط
        patterns.extend(self._find_double_top(df, peaks))
        patterns.extend(self._find_double_bottom(df, troughs))
        patterns.extend(self._find_head_and_shoulders(df, peaks, troughs))
        patterns.extend(self._find_inverse_head_and_shoulders(df, peaks, troughs))
        patterns.extend(self._find_triangles(df))
        patterns.extend(self._find_wedges(df))
        patterns.extend(self._find_channels(df))
        patterns.extend(self._find_flags(df))
        
        return patterns
    
    def _find_peaks_and_troughs(self, df: pd.DataFrame, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """تحديد القمم والقيعان في السعر"""
        highs = df['high'].values
        lows = df['low'].values
        
        peaks = []
        troughs = []
        
        for i in range(window, len(df) - window):
            if all(highs[i] > highs[i-window:i]) and all(highs[i] > highs[i+1:i+window+1]):
                peaks.append(i)
            if all(lows[i] < lows[i-window:i]) and all(lows[i] < lows[i+1:i+window+1]):
                troughs.append(i)
        
        return np.array(peaks), np.array(troughs)
    
    def _find_double_top(self, df: pd.DataFrame, peaks: np.ndarray) -> List[Pattern]:
        """البحث عن نمط القمة المزدوجة"""
        patterns = []
        config = self.patterns_config['double_top']
        
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                distance = peaks[j] - peaks[i]
                
                if config['min_distance'] <= distance <= config['max_distance']:
                    price1 = df['high'].iloc[peaks[i]]
                    price2 = df['high'].iloc[peaks[j]]
                    
                    if abs(price1 - price2) / price1 <= config['tolerance']:
                        # حساب خط العنق
                        neckline = min(df['low'].iloc[peaks[i]:peaks[j]])
                        target = neckline - (price1 - neckline)
                        stop_loss = max(price1, price2) * 1.02
                        
                        patterns.append(Pattern(
                            name='Double Top',
                            start_idx=peaks[i],
                            end_idx=peaks[j],
                            pattern_type='bearish',
                            confidence=0.8,
                            target_price=target,
                            stop_loss=stop_loss
                        ))
        
        return patterns
    
    def _find_double_bottom(self, df: pd.DataFrame, troughs: np.ndarray) -> List[Pattern]:
        """البحث عن نمط القاع المزدوج"""
        patterns = []
        config = self.patterns_config['double_bottom']
        
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                distance = troughs[j] - troughs[i]
                
                if config['min_distance'] <= distance <= config['max_distance']:
                    price1 = df['low'].iloc[troughs[i]]
                    price2 = df['low'].iloc[troughs[j]]
                    
                    if abs(price1 - price2) / price1 <= config['tolerance']:
                        # حساب خط العنق
                        neckline = max(df['high'].iloc[troughs[i]:troughs[j]])
                        target = neckline + (neckline - price1)
                        stop_loss = min(price1, price2) * 0.98
                        
                        patterns.append(Pattern(
                            name='Double Bottom',
                            start_idx=troughs[i],
                            end_idx=troughs[j],
                            pattern_type='bullish',
                            confidence=0.8,
                            target_price=target,
                            stop_loss=stop_loss
                        ))
        
        return patterns
    
    def _find_head_and_shoulders(self, df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> List[Pattern]:
        """البحث عن نمط الرأس والكتفين"""
        patterns = []
        config = self.patterns_config['head_shoulders']
        
        for i in range(len(peaks) - 2):
            # البحث عن ثلاث قمم متتالية
            head_idx = peaks[i+1]
            left_shoulder_idx = peaks[i]
            right_shoulder_idx = peaks[i+2]
            
            if (config['min_distance'] <= head_idx - left_shoulder_idx <= config['max_distance'] and
                config['min_distance'] <= right_shoulder_idx - head_idx <= config['max_distance']):
                
                head = df['high'].iloc[head_idx]
                left_shoulder = df['high'].iloc[left_shoulder_idx]
                right_shoulder = df['high'].iloc[right_shoulder_idx]
                
                # التحقق من أن الرأس أعلى من الكتفين
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder <= config['tolerance']):
                    
                    # حساب خط العنق
                    neckline = min(df['low'].iloc[left_shoulder_idx:right_shoulder_idx])
                    target = neckline - (head - neckline)
                    stop_loss = head * 1.02
                    
                    patterns.append(Pattern(
                        name='Head and Shoulders',
                        start_idx=left_shoulder_idx,
                        end_idx=right_shoulder_idx,
                        pattern_type='bearish',
                        confidence=0.85,
                        target_price=target,
                        stop_loss=stop_loss
                    ))
        
        return patterns
    
    def _find_inverse_head_and_shoulders(self, df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> List[Pattern]:
        """البحث عن نمط الرأس والكتفين المعكوس"""
        patterns = []
        config = self.patterns_config['inverse_head_shoulders']
        
        for i in range(len(troughs) - 2):
            # البحث عن ثلاث قيعان متتالية
            head_idx = troughs[i+1]
            left_shoulder_idx = troughs[i]
            right_shoulder_idx = troughs[i+2]
            
            if (config['min_distance'] <= head_idx - left_shoulder_idx <= config['max_distance'] and
                config['min_distance'] <= right_shoulder_idx - head_idx <= config['max_distance']):
                
                head = df['low'].iloc[head_idx]
                left_shoulder = df['low'].iloc[left_shoulder_idx]
                right_shoulder = df['low'].iloc[right_shoulder_idx]
                
                # التحقق من أن الرأس أدنى من الكتفين
                if (head < left_shoulder and head < right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder <= config['tolerance']):
                    
                    # حساب خط العنق
                    neckline = max(df['high'].iloc[left_shoulder_idx:right_shoulder_idx])
                    target = neckline + (neckline - head)
                    stop_loss = head * 0.98
                    
                    patterns.append(Pattern(
                        name='Inverse Head and Shoulders',
                        start_idx=left_shoulder_idx,
                        end_idx=right_shoulder_idx,
                        pattern_type='bullish',
                        confidence=0.85,
                        target_price=target,
                        stop_loss=stop_loss
                    ))
        
        return patterns
    
    def _find_triangles(self, df: pd.DataFrame) -> List[Pattern]:
        """البحث عن أنماط المثلثات"""
        patterns = []
        config = self.patterns_config['triangle']
        
        for start in range(len(df) - config['max_distance']):
            end = min(start + config['max_distance'], len(df))
            segment = df.iloc[start:end]
            
            if len(segment) < config['min_points']:
                continue
            
            highs = segment['high'].values
            lows = segment['low'].values
            
            # حساب خطوط الاتجاه
            x = np.arange(len(segment))
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            # تحديد نوع المثلث
            if abs(high_slope) < config['tolerance'] and low_slope > config['tolerance']:
                # مثلث صاعد
                target = segment['close'].iloc[-1] + abs(highs.max() - lows.min())
                stop_loss = lows.min() * 0.98
                
                patterns.append(Pattern(
                    name='Ascending Triangle',
                    start_idx=start,
                    end_idx=end-1,
                    pattern_type='bullish',
                    confidence=0.75,
                    target_price=target,
                    stop_loss=stop_loss
                ))
                
            elif high_slope < -config['tolerance'] and abs(low_slope) < config['tolerance']:
                # مثلث هابط
                target = segment['close'].iloc[-1] - abs(highs.max() - lows.min())
                stop_loss = highs.max() * 1.02
                
                patterns.append(Pattern(
                    name='Descending Triangle',
                    start_idx=start,
                    end_idx=end-1,
                    pattern_type='bearish',
                    confidence=0.75,
                    target_price=target,
                    stop_loss=stop_loss
                ))
                
            elif abs(high_slope + low_slope) < config['tolerance']:
                # مثلث متماثل
                if high_slope < 0:
                    pattern_type = 'bearish'
                    target = segment['close'].iloc[-1] - abs(highs.max() - lows.min())
                    stop_loss = highs.max() * 1.02
                else:
                    pattern_type = 'bullish'
                    target = segment['close'].iloc[-1] + abs(highs.max() - lows.min())
                    stop_loss = lows.min() * 0.98
                
                patterns.append(Pattern(
                    name='Symmetrical Triangle',
                    start_idx=start,
                    end_idx=end-1,
                    pattern_type=pattern_type,
                    confidence=0.7,
                    target_price=target,
                    stop_loss=stop_loss
                ))
        
        return patterns
    
    def _find_wedges(self, df: pd.DataFrame) -> List[Pattern]:
        """البحث عن أنماط الإسفين"""
        patterns = []
        config = self.patterns_config['wedge']
        
        for start in range(len(df) - config['max_distance']):
            end = min(start + config['max_distance'], len(df))
            segment = df.iloc[start:end]
            
            if len(segment) < config['min_points']:
                continue
            
            highs = segment['high'].values
            lows = segment['low'].values
            
            # حساب خطوط الاتجاه
            x = np.arange(len(segment))
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            # تحديد نوع الإسفين
            if high_slope < -config['tolerance'] and low_slope < -config['tolerance']:
                if abs(high_slope - low_slope) < config['tolerance']:
                    # إسفين هابط
                    target = segment['close'].iloc[-1] - abs(highs.max() - lows.min())
                    stop_loss = highs.max() * 1.02
                    
                    patterns.append(Pattern(
                        name='Falling Wedge',
                        start_idx=start,
                        end_idx=end-1,
                        pattern_type='bullish',
                        confidence=0.8,
                        target_price=target,
                        stop_loss=stop_loss
                    ))
                    
            elif high_slope > config['tolerance'] and low_slope > config['tolerance']:
                if abs(high_slope - low_slope) < config['tolerance']:
                    # إسفين صاعد
                    target = segment['close'].iloc[-1] + abs(highs.max() - lows.min())
                    stop_loss = lows.min() * 0.98
                    
                    patterns.append(Pattern(
                        name='Rising Wedge',
                        start_idx=start,
                        end_idx=end-1,
                        pattern_type='bearish',
                        confidence=0.8,
                        target_price=target,
                        stop_loss=stop_loss
                    ))
        
        return patterns
    
    def _find_channels(self, df: pd.DataFrame) -> List[Pattern]:
        """البحث عن أنماط القنوات السعرية"""
        patterns = []
        config = self.patterns_config['channel']
        
        for start in range(len(df) - config['max_distance']):
            end = min(start + config['max_distance'], len(df))
            segment = df.iloc[start:end]
            
            if len(segment) < config['min_points']:
                continue
            
            highs = segment['high'].values
            lows = segment['low'].values
            
            # حساب خطوط الاتجاه
            x = np.arange(len(segment))
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            # التحقق من توازي الخطوط
            if abs(high_slope - low_slope) < config['tolerance']:
                if high_slope > config['tolerance']:
                    # قناة صاعدة
                    target = segment['close'].iloc[-1] + abs(highs.max() - lows.min())
                    stop_loss = lows.min() * 0.98
                    pattern_type = 'bullish'
                elif high_slope < -config['tolerance']:
                    # قناة هابطة
                    target = segment['close'].iloc[-1] - abs(highs.max() - lows.min())
                    stop_loss = highs.max() * 1.02
                    pattern_type = 'bearish'
                else:
                    # قناة أفقية
                    continue
                
                patterns.append(Pattern(
                    name=f"{pattern_type.capitalize()} Channel",
                    start_idx=start,
                    end_idx=end-1,
                    pattern_type=pattern_type,
                    confidence=0.75,
                    target_price=target,
                    stop_loss=stop_loss
                ))
        
        return patterns
    
    def _find_flags(self, df: pd.DataFrame) -> List[Pattern]:
        """البحث عن أنماط الأعلام"""
        patterns = []
        config = self.patterns_config['flag']
        
        for start in range(len(df) - config['max_distance']):
            end = min(start + config['max_distance'], len(df))
            segment = df.iloc[start:end]
            
            if len(segment) < config['min_points']:
                continue
            
            # البحث عن العمود (الحركة القوية السابقة)
            pole_start = max(0, start - 20)
            pole = df.iloc[pole_start:start]
            
            if len(pole) < 5:
                continue
            
            pole_move = pole['close'].iloc[-1] - pole['close'].iloc[0]
            pole_height = abs(pole_move)
            
            if pole_height / pole['close'].iloc[0] < 0.05:  # تجاهل الحركات الصغيرة
                continue
            
            # تحليل نمط العلم
            closes = segment['close'].values
            x = np.arange(len(segment))
            slope, intercept = np.polyfit(x, closes, 1)
            
            if abs(slope) > config['tolerance']:
                if pole_move > 0 and slope < 0:
                    # علم صاعد
                    target = segment['close'].iloc[-1] + pole_height
                    stop_loss = segment['low'].min() * 0.98
                    
                    patterns.append(Pattern(
                        name='Bull Flag',
                        start_idx=start,
                        end_idx=end-1,
                        pattern_type='bullish',
                        confidence=0.8,
                        target_price=target,
                        stop_loss=stop_loss
                    ))
                    
                elif pole_move < 0 and slope > 0:
                    # علم هابط
                    target = segment['close'].iloc[-1] - pole_height
                    stop_loss = segment['high'].max() * 1.02
                    
                    patterns.append(Pattern(
                        name='Bear Flag',
                        start_idx=start,
                        end_idx=end-1,
                        pattern_type='bearish',
                        confidence=0.8,
                        target_price=target,
                        stop_loss=stop_loss
                    ))
        
        return patterns
