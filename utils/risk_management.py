import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import ta

class RiskManager:
    def __init__(self):
        """تهيئة مدير المخاطر"""
        self.min_rr_ratio = 1.5  # الحد الأدنى لنسبة المخاطرة/المكافأة
        self.max_risk_percent = 2.0  # أقصى نسبة مخاطرة من رأس المال
        self.atr_multiplier = 2.0  # مضاعف ATR لتحديد وقف الخسارة
        
    def calculate_entry_points(self, df: pd.DataFrame, technical_signals: Dict) -> Dict[str, float]:
        """
        حساب نقاط الدخول ووقف الخسارة وجني الأرباح بناءً على التحليل الفني والمخاطر
        
        المعاملات:
            df: إطار البيانات التاريخية
            technical_signals: نتائج التحليل الفني
            
        العائد:
            قاموس يحتوي على نقاط الدخول والوقف والهدف
        """
        try:
            if df.empty or not technical_signals:
                logging.warning("لا توجد بيانات كافية لحساب نقاط الدخول")
                return {}
                
            current_price = technical_signals.get('current_price')
            if not current_price:
                logging.warning("السعر الحالي غير متوفر")
                return {}
                
            # حساب ATR للتقلب
            atr = df['ATR'].iloc[-1]
            if pd.isna(atr):
                logging.warning("لا يمكن حساب ATR")
                return {}
                
            # تحديد اتجاه السوق من المتوسطات المتحركة
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            ma200 = df['MA200'].iloc[-1]
            
            if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200):
                logging.warning("المتوسطات المتحركة غير متوفرة")
                return {}
                
            # تحديد الاتجاه
            trend = 'neutral'
            if ma20 > ma50 and ma50 > ma200:
                trend = 'bullish'
            elif ma20 < ma50 and ma50 < ma200:
                trend = 'bearish'
                
            # حساب وقف الخسارة وجني الأرباح
            if trend == 'bullish':
                stop_loss = current_price - (atr * self.atr_multiplier)
                take_profit = current_price + (atr * self.atr_multiplier * self.min_rr_ratio)
            elif trend == 'bearish':
                stop_loss = current_price + (atr * self.atr_multiplier)
                take_profit = current_price - (atr * self.atr_multiplier * self.min_rr_ratio)
            else:
                logging.info("لا يوجد اتجاه واضح للسوق")
                return {}
                
            # حساب المخاطر والمكافأة
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # التحقق من نسبة المخاطرة/المكافأة
            if risk_reward_ratio < self.min_rr_ratio:
                logging.info(f"نسبة المخاطرة/المكافأة {risk_reward_ratio:.2f} أقل من الحد الأدنى {self.min_rr_ratio}")
                return {}
                
            return {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk,
                'reward_amount': reward,
                'risk_reward_ratio': risk_reward_ratio,
                'trend': trend
            }
            
        except Exception as e:
            logging.error(f"خطأ في حساب نقاط الدخول: {str(e)}")
            return {}
            
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """حساب نقاط الارتكاز"""
        try:
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_close = df['close'].iloc[-1]
            
            pivot = (last_high + last_low + last_close) / 3
            r1 = 2 * pivot - last_low
            r2 = pivot + (last_high - last_low)
            r3 = r1 + (last_high - last_low)
            s1 = 2 * pivot - last_high
            s2 = pivot - (last_high - last_low)
            s3 = s1 - (last_high - last_low)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
        except:
            return {}
            
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """تحديد مستويات الدعم والمقاومة"""
        try:
            # استخدام المتوسطات المتحركة كمستويات دعم/مقاومة ديناميكية
            df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['MA50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['MA200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # حساب مستويات بولينجر
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            # تحديد القمم والقيعان المحلية
            window = 20
            df['local_high'] = df['high'].rolling(window=window, center=True).max()
            df['local_low'] = df['low'].rolling(window=window, center=True).min()
            
            recent_highs = df['local_high'].dropna().unique()[-3:]
            recent_lows = df['local_low'].dropna().unique()[-3:]
            
            return {
                'ma20': df['MA20'].iloc[-1],
                'ma50': df['MA50'].iloc[-1],
                'ma200': df['MA200'].iloc[-1],
                'bb_upper': df['BB_upper'].iloc[-1],
                'bb_lower': df['BB_lower'].iloc[-1],
                'recent_highs': list(recent_highs),
                'recent_lows': list(recent_lows)
            }
        except:
            return {}
            
    def _calculate_volatility(self, df: pd.DataFrame, technical_signals: Dict) -> Dict[str, float]:
        """حساب مقاييس التقلب"""
        try:
            atr = technical_signals.get('atr', 0)
            bb_width = technical_signals.get('bollinger', {}).get('width', 0)
            
            # حساب التقلب التاريخي
            returns = np.log(df['close'] / df['close'].shift(1))
            hist_vol = returns.std() * np.sqrt(252)  # التقلب السنوي
            
            return {
                'atr': atr,
                'bb_width': bb_width,
                'historical_volatility': hist_vol
            }
        except:
            return {}
            
    def _optimize_entry_point(
        self,
        current_price: float,
        technical_signals: Dict,
        pivot_points: Dict,
        support_resistance: Dict,
        volatility: Dict
    ) -> Optional[float]:
        """تحسين نقطة الدخول بناءً على التحليل الفني والمخاطر"""
        try:
            trend = technical_signals.get('trend', '')
            trend_strength = technical_signals.get('trend_strength', 0)
            
            if not trend or trend_strength < 40:
                return None
                
            entry_candidates = []
            
            # إضافة السعر الحالي كنقطة دخول محتملة
            entry_candidates.append((current_price, 1.0))
            
            # استخدام نقاط الارتكاز
            if trend.startswith('صعود'):
                entry_candidates.extend([
                    (pivot_points.get('pivot', 0), 0.8),
                    (pivot_points.get('s1', 0), 0.7),
                    (pivot_points.get('s2', 0), 0.5)
                ])
            else:
                entry_candidates.extend([
                    (pivot_points.get('pivot', 0), 0.8),
                    (pivot_points.get('r1', 0), 0.7),
                    (pivot_points.get('r2', 0), 0.5)
                ])
                
            # استخدام المتوسطات المتحركة
            ma20 = support_resistance.get('ma20', 0)
            ma50 = support_resistance.get('ma50', 0)
            if ma20 and ma50:
                entry_candidates.append((ma20, 0.9))
                entry_candidates.append((ma50, 0.8))
                
            # تصفية وترجيح نقاط الدخول
            valid_entries = [
                (price, weight)
                for price, weight in entry_candidates
                if price > 0
            ]
            
            if not valid_entries:
                return current_price
                
            # اختيار أفضل نقطة دخول بناءً على الأوزان
            best_entry = max(valid_entries, key=lambda x: x[1])[0]
            
            return best_entry
            
        except:
            return current_price
            
    def _calculate_stop_and_target(
        self,
        entry_point: float,
        technical_signals: Dict,
        pivot_points: Dict,
        support_resistance: Dict,
        volatility: Dict
    ) -> Tuple[Optional[float], Optional[float]]:
        """حساب وقف الخسارة وجني الأرباح"""
        try:
            trend = technical_signals.get('trend', '')
            atr = volatility.get('atr', 0)
            
            if not trend or not atr:
                return None, None
                
            # حساب وقف الخسارة الأولي باستخدام ATR
            initial_stop = self.atr_multiplier * atr
            
            if trend.startswith('صعود'):
                # تحسين وقف الخسارة باستخدام مستويات الدعم
                stop_candidates = [
                    entry_point - initial_stop,
                    support_resistance.get('ma20', 0),
                    support_resistance.get('bb_lower', 0),
                    *support_resistance.get('recent_lows', []),
                    pivot_points.get('s1', 0),
                    pivot_points.get('s2', 0)
                ]
                
                # اختيار أقرب مستوى دعم تحت نقطة الدخول
                valid_stops = [
                    s for s in stop_candidates
                    if s > 0 and s < entry_point
                ]
                stop_loss = max(valid_stops) if valid_stops else entry_point - initial_stop
                
                # حساب هدف الربح باستخدام نسبة المخاطرة/المكافأة ومستويات المقاومة
                min_target = entry_point + (initial_stop * self.min_rr_ratio)
                target_candidates = [
                    min_target,
                    support_resistance.get('bb_upper', 0),
                    *support_resistance.get('recent_highs', []),
                    pivot_points.get('r1', 0),
                    pivot_points.get('r2', 0)
                ]
                
                # اختيار أقرب مستوى مقاومة فوق الحد الأدنى للهدف
                valid_targets = [
                    t for t in target_candidates
                    if t > min_target
                ]
                take_profit = min(valid_targets) if valid_targets else min_target
                
            else:
                # نفس المنطق للاتجاه الهابط مع عكس المستويات
                stop_candidates = [
                    entry_point + initial_stop,
                    support_resistance.get('ma20', 0),
                    support_resistance.get('bb_upper', 0),
                    *support_resistance.get('recent_highs', []),
                    pivot_points.get('r1', 0),
                    pivot_points.get('r2', 0)
                ]
                
                valid_stops = [
                    s for s in stop_candidates
                    if s > 0 and s > entry_point
                ]
                stop_loss = min(valid_stops) if valid_stops else entry_point + initial_stop
                
                min_target = entry_point - (initial_stop * self.min_rr_ratio)
                target_candidates = [
                    min_target,
                    support_resistance.get('bb_lower', 0),
                    *support_resistance.get('recent_lows', []),
                    pivot_points.get('s1', 0),
                    pivot_points.get('s2', 0)
                ]
                
                valid_targets = [
                    t for t in target_candidates
                    if t > 0 and t < min_target
                ]
                take_profit = max(valid_targets) if valid_targets else min_target
                
            return stop_loss, take_profit
            
        except:
            return None, None
