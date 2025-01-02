"""
متخذ القرار للتداول
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class MarketCondition:
    """فئة لتمثيل حالة السوق"""
    trend: str  # اتجاه السوق (صاعد، هابط، متذبذب)
    strength: float  # قوة الاتجاه (0-1)
    volatility: float  # التذبذب (0-1)
    volume_trend: str  # اتجاه الحجم (متزايد، متناقص، مستقر)
    risk_level: float  # مستوى المخاطرة (0-1)

@dataclass
class TradingDecision:
    """فئة لتمثيل قرار التداول"""
    action: str  # شراء، بيع، انتظار
    confidence: float  # مستوى الثقة في القرار (0-1)
    entry_price: float  # سعر الدخول المقترح
    stop_loss: float  # سعر وقف الخسارة
    take_profit: float  # سعر جني الأرباح
    risk_reward_ratio: float  # نسبة المخاطرة/المكافأة
    reasoning: List[str]  # أسباب القرار

class SmartDecisionMaker:
    def __init__(self):
        """تهيئة صانع القرارات الذكي"""
        # أوزان المؤشرات الفنية
        self.technical_weights = {
            'RSI': 0.15,
            'MACD': 0.15,
            'BB': 0.1,
            'MA': 0.1,
            'Volume': 0.1,
            'Patterns': 0.2,
            'Support_Resistance': 0.2
        }
        
        # عتبات المؤشرات
        self.thresholds = {
            'RSI': {'oversold': 30, 'overbought': 70},
            'Volume': {'significant': 1.5},  # مقارنة بمتوسط الحجم
            'Trend_Strength': {'weak': 0.3, 'strong': 0.7},
            'Risk_Reward': {'minimum': 2.0}  # الحد الأدنى لنسبة المخاطرة/المكافأة
        }

    def analyze_market_condition(self, 
                               df: pd.DataFrame, 
                               technical_indicators: Dict,
                               patterns: Dict) -> MarketCondition:
        """تحليل حالة السوق الحالية"""
        try:
            # تحليل الاتجاه
            ma_20 = df['close'].rolling(window=20).mean()
            ma_50 = df['close'].rolling(window=50).mean()
            current_price = df['close'].iloc[-1]
            
            if current_price > ma_20.iloc[-1] > ma_50.iloc[-1]:
                trend = "صاعد"
            elif current_price < ma_20.iloc[-1] < ma_50.iloc[-1]:
                trend = "هابط"
            else:
                trend = "متذبذب"
            
            # حساب قوة الاتجاه
            atr = technical_indicators.get('ATR', df['high'] - df['low'])
            price_range = df['high'].max() - df['low'].min()
            trend_strength = min(1.0, atr.iloc[-1] / price_range)
            
            # حساب التذبذب
            volatility = min(1.0, (df['high'].iloc[-1] - df['low'].iloc[-1]) / current_price)
            
            # تحليل اتجاه الحجم
            volume_sma = df['volume'].rolling(window=20).mean()
            if df['volume'].iloc[-1] > volume_sma.iloc[-1] * self.thresholds['Volume']['significant']:
                volume_trend = "متزايد"
            elif df['volume'].iloc[-1] < volume_sma.iloc[-1] / self.thresholds['Volume']['significant']:
                volume_trend = "متناقص"
            else:
                volume_trend = "مستقر"
            
            # حساب مستوى المخاطرة
            rsi = technical_indicators.get('RSI', pd.Series([50] * len(df)))
            risk_level = 1.0 - (abs(50 - rsi.iloc[-1]) / 50)
            
            return MarketCondition(
                trend=trend,
                strength=trend_strength,
                volatility=volatility,
                volume_trend=volume_trend,
                risk_level=risk_level
            )
            
        except Exception as e:
            logging.error(f"خطأ في تحليل حالة السوق: {str(e)}")
            return MarketCondition("متذبذب", 0.5, 0.5, "مستقر", 0.5)

    def calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """حساب مستويات الدعم والمقاومة"""
        try:
            # استخدام القمم والقيعان المحلية
            high_points = df['high'].rolling(window=5, center=True).max()
            low_points = df['low'].rolling(window=5, center=True).min()
            
            current_price = df['close'].iloc[-1]
            
            # البحث عن أقرب مستويات الدعم والمقاومة
            resistance = min([p for p in high_points if p > current_price], default=current_price * 1.05)
            support = max([p for p in low_points if p < current_price], default=current_price * 0.95)
            
            return support, resistance
            
        except Exception as e:
            logging.error(f"خطأ في حساب مستويات الدعم والمقاومة: {str(e)}")
            return df['close'].iloc[-1] * 0.95, df['close'].iloc[-1] * 1.05

    def evaluate_pattern_significance(self, 
                                   patterns: Dict, 
                                   market_condition: MarketCondition) -> float:
        """تقييم أهمية الأنماط الفنية"""
        try:
            if not patterns:
                return 0.0
                
            # تقييم قوة كل نمط
            pattern_scores = []
            for pattern, details in patterns.items():
                score = details.get('strength', 0.5)
                
                # زيادة الأهمية إذا كان النمط يتوافق مع الاتجاه العام
                if (market_condition.trend == "صاعد" and details.get('type') == 'bullish') or \
                   (market_condition.trend == "هابط" and details.get('type') == 'bearish'):
                    score *= 1.5
                    
                pattern_scores.append(score)
            
            return min(1.0, sum(pattern_scores) / len(pattern_scores))
            
        except Exception as e:
            logging.error(f"خطأ في تقييم الأنماط: {str(e)}")
            return 0.0

    def make_decision(self, 
                     df: pd.DataFrame,
                     technical_indicators: Dict,
                     patterns: Dict) -> TradingDecision:
        """اتخاذ قرار التداول النهائي"""
        try:
            # تحليل حالة السوق
            market_condition = self.analyze_market_condition(df, technical_indicators, patterns)
            
            # حساب مستويات الدعم والمقاومة
            support, resistance = self.calculate_support_resistance(df)
            
            # تقييم الأنماط
            pattern_significance = self.evaluate_pattern_significance(patterns, market_condition)
            
            # جمع المؤشرات
            current_price = df['close'].iloc[-1]
            rsi = technical_indicators.get('RSI', pd.Series([50] * len(df))).iloc[-1]
            macd = technical_indicators.get('MACD', pd.Series([0] * len(df))).iloc[-1]
            
            # حساب نقاط القرار
            buy_points = 0
            sell_points = 0
            reasoning = []
            
            # تحليل RSI
            if rsi < self.thresholds['RSI']['oversold']:
                buy_points += self.technical_weights['RSI']
                reasoning.append("RSI في منطقة ذروة البيع")
            elif rsi > self.thresholds['RSI']['overbought']:
                sell_points += self.technical_weights['RSI']
                reasoning.append("RSI في منطقة ذروة الشراء")
            
            # تحليل MACD
            if macd > 0:
                buy_points += self.technical_weights['MACD']
                reasoning.append("إشارة MACD إيجابية")
            else:
                sell_points += self.technical_weights['MACD']
                reasoning.append("إشارة MACD سلبية")
            
            # تحليل الاتجاه
            if market_condition.trend == "صاعد":
                buy_points += self.technical_weights['MA']
                reasoning.append("الاتجاه العام صاعد")
            elif market_condition.trend == "هابط":
                sell_points += self.technical_weights['MA']
                reasoning.append("الاتجاه العام هابط")
            
            # تحليل الحجم
            if market_condition.volume_trend == "متزايد":
                if market_condition.trend == "صاعد":
                    buy_points += self.technical_weights['Volume']
                    reasoning.append("حجم التداول يدعم الاتجاه الصاعد")
                elif market_condition.trend == "هابط":
                    sell_points += self.technical_weights['Volume']
                    reasoning.append("حجم التداول يدعم الاتجاه الهابط")
            
            # إضافة تأثير الأنماط
            if pattern_significance > 0:
                if any(p.get('type') == 'bullish' for p in patterns.values()):
                    buy_points += self.technical_weights['Patterns'] * pattern_significance
                    reasoning.append("أنماط فنية صاعدة")
                elif any(p.get('type') == 'bearish' for p in patterns.values()):
                    sell_points += self.technical_weights['Patterns'] * pattern_significance
                    reasoning.append("أنماط فنية هابطة")
            
            # اتخاذ القرار النهائي
            total_points = max(buy_points, sell_points)
            confidence = total_points / sum(self.technical_weights.values())
            
            if buy_points > sell_points and confidence > 0.6:
                action = "شراء"
                stop_loss = support * 0.99
                take_profit = current_price + (current_price - stop_loss) * 2
            elif sell_points > buy_points and confidence > 0.6:
                action = "بيع"
                stop_loss = resistance * 1.01
                take_profit = current_price - (stop_loss - current_price) * 2
            else:
                action = "انتظار"
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.02
                reasoning.append("المؤشرات غير حاسمة")
            
            # حساب نسبة المخاطرة/المكافأة
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            return TradingDecision(
                action=action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                reasoning=reasoning
            )
            
        except Exception as e:
            logging.error(f"خطأ في اتخاذ القرار: {str(e)}")
            return TradingDecision(
                action="انتظار",
                confidence=0.0,
                entry_price=df['close'].iloc[-1],
                stop_loss=df['close'].iloc[-1] * 0.95,
                take_profit=df['close'].iloc[-1] * 1.05,
                risk_reward_ratio=0.0,
                reasoning=["حدث خطأ في تحليل البيانات"]
            )

class DecisionMaker:
    """فئة اتخاذ قرارات التداول"""
    
    def __init__(self):
        """تهيئة متخذ القرار"""
        logging.info("تم تهيئة متخذ القرار")
        
        # تعريف الأوزان للمؤشرات المختلفة
        self.weights = {
            'Technical': 0.5,  # زيادة وزن التحليل الفني
            'Market': 0.2,    
            'Pattern': 0.2,    
            'Risk': 0.1        
        }
        
        # تعريف الأوزان للمؤشرات الفنية
        self.technical_weights = {
            'MA': 0.2,
            'RSI': 0.2,
            'MACD': 0.4,      # زيادة وزن MACD
            'Strength': 0.2
        }
        
        # إضافة عتبات للتحقق
        self.thresholds = {
            'MACD': {
                'min_signal': 0.1,     # الحد الأدنى لقوة إشارة MACD
                'max_signal': 100.0,   # الحد الأقصى لقوة إشارة MACD
                'min_hist': 0.05       # الحد الأدنى للهيستوجرام
            },
            'Pattern': {
                'min_strength': 0.6,   # الحد الأدنى لقوة النمط
                'min_confidence': 0.7  # الحد الأدنى لثقة النمط
            }
        }
        
        self.smart_decision_maker = SmartDecisionMaker()
    
    def make_decision(self, technical_signals: Dict[str, Any], pattern_signals: Dict[str, float], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        اتخاذ قرار التداول
        
        المعلمات:
            technical_signals: نتائج التحليل الفني
            pattern_signals: نتائج تحليل الأنماط
            market_data: بيانات السوق
            
        العوائد:
            Dict: قرار التداول مع التفاصيل والتوصيات
        """
        try:
            # الحصول على آخر سعر
            candles = technical_signals if isinstance(technical_signals, list) else []
            current_price = float(candles[-1]['close']) if candles else 0

            # تحليل المؤشرات الفنية
            technical_score = self._analyze_technical_signals(technical_signals)
            
            # تحليل الأنماط
            pattern_score = self._analyze_pattern_signals(pattern_signals)
            
            # تحليل السوق
            market_score = market_data.get('market_analysis', {}).get('market_score', 50)
            
            # حساب النتيجة النهائية
            final_score = (
                technical_score * self.weights['Technical'] +
                market_score * self.weights['Market'] +
                pattern_score * self.weights['Pattern']
            )

            # تحديد القرار والثقة
            decision, confidence = self._get_decision(final_score)
            
            # حساب مستويات الأسعار
            if decision != 'انتظار' and current_price > 0:
                # حساب نسبة التحرك المتوقعة بناءً على التذبذب
                volatility = market_data.get('market_analysis', {}).get('volatility', 0.02)
                expected_move = current_price * volatility
                
                if decision == 'شراء':
                    entry_price = current_price
                    stop_loss = current_price - expected_move
                    take_profit = current_price + (expected_move * 2)  # نسبة مخاطرة/مكافأة 1:2
                else:  # بيع
                    entry_price = current_price
                    stop_loss = current_price + expected_move
                    take_profit = current_price - (expected_move * 2)
                    
                risk_reward_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
            else:
                entry_price = current_price
                stop_loss = current_price
                take_profit = current_price
                risk_reward_ratio = 0
            
            # توليد أسباب القرار
            reasons = []
            if technical_score > 60:
                reasons.append("المؤشرات الفنية تشير إلى اتجاه قوي")
            elif technical_score < 40:
                reasons.append("المؤشرات الفنية تشير إلى ضعف في الاتجاه")
            
            market_analysis = market_data.get('market_analysis', {})
            if market_analysis.get('trend') == 'صاعد':
                reasons.append("السوق في اتجاه صاعد")
            elif market_analysis.get('trend') == 'هابط':
                reasons.append("السوق في اتجاه هابط")
            
            if not reasons:
                reasons.append("السوق غير واضح الاتجاه")

            return {
                'smart_decision': {
                    'action': decision,
                    'confidence': confidence,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'reasoning': reasons
                },
                'weights': self.weights,
                'scores': {
                    'technical': technical_score,
                    'market': market_score,
                    'pattern': pattern_score,
                    'final': final_score
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"خطأ في اتخاذ القرار: {str(e)}")
            return {
                'smart_decision': {
                    'action': 'انتظار',
                    'confidence': 0,
                    'entry_price': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'risk_reward_ratio': 0,
                    'reasoning': [f"حدث خطأ: {str(e)}"]
                },
                'weights': self.weights,
                'scores': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_decision(self, score: float) -> tuple[str, float]:
        """
        تحديد القرار والثقة بناءً على النتيجة النهائية وعوامل أخرى
        
        المعلمات:
            score: النتيجة النهائية (0-100)
            
        العوائد:
            tuple: (القرار، مستوى الثقة)
        """
        # حساب الثقة الأساسية من النتيجة
        if score >= 65:
            decision = 'شراء'
            # زيادة الثقة كلما زادت النتيجة عن 65
            confidence = 0.5 + min((score - 65) / 35, 1) * 0.5
            
        elif score <= 35:
            decision = 'بيع'
            # زيادة الثقة كلما قلت النتيجة عن 35
            confidence = 0.5 + min((35 - score) / 35, 1) * 0.5
            
        else:
            decision = 'انتظار'
            # حساب الثقة في منطقة الانتظار
            center_distance = abs(score - 50)
            if center_distance < 5:  # قريب جداً من المنتصف
                confidence = 0.8  # ثقة عالية في قرار الانتظار
            else:
                # تقليل الثقة تدريجياً كلما ابتعدنا عن المنتصف
                confidence = 0.8 - (center_distance - 5) / 20
        
        # تأكد من أن الثقة في النطاق 0-1
        confidence = max(min(confidence, 1.0), 0.4)  # زيادة الحد الأدنى للثقة إلى 0.4
        
        # تعديل القرار بناءً على مستوى الثقة
        if confidence >= 0.8:
            if decision != 'انتظار':
                decision = f"{decision} قوي"
        elif confidence <= 0.5 and decision != 'انتظار':
            decision = 'انتظار'
        
        return decision, confidence
    
    def _analyze_technical_signals(self, signals: Dict[str, Any]) -> float:
        """تحليل الإشارات الفنية وحساب النتيجة مع تحسين التحقق"""
        try:
            total_score = 50  # نبدأ من نقطة متعادلة
            
            # تحليل المتوسطات المتحركة
            if isinstance(signals, list) and len(signals) > 0:
                last_candle = signals[-1]
                
                # تحليل SMA
                sma_20 = float(last_candle.get('SMA_20_PCT', 0))
                sma_50 = float(last_candle.get('SMA_50_PCT', 0))
                
                # تحليل EMA
                ema_20 = float(last_candle.get('EMA_20_PCT', 0))
                ema_50 = float(last_candle.get('EMA_50_PCT', 0))
                
                # حساب متوسط المؤشرات
                ma_avg = (sma_20 + sma_50 + ema_20 + ema_50) / 4
                ma_score = 50 + (ma_avg * 100)  # تحويل النسبة المئوية إلى نقاط
                total_score += ma_score * self.technical_weights['MA']
            
            # تحليل RSI
            if isinstance(signals, list) and len(signals) > 0:
                rsi = float(last_candle.get('RSI', 50))
                
                # حساب نتيجة RSI
                if rsi > 70:  # منطقة ذروة الشراء
                    rsi_score = max(0, 100 - (rsi - 70) * 3)
                elif rsi < 30:  # منطقة ذروة البيع
                    rsi_score = min(100, (30 - rsi) * 3)
                else:  # المنطقة المتعادلة
                    rsi_score = 50 + (rsi - 50)
                
                total_score += rsi_score * self.technical_weights['RSI']
            
            # تحليل MACD
            if isinstance(signals, list) and len(signals) > 0:
                macd = float(last_candle.get('MACD', 0))
                macd_signal = float(last_candle.get('MACD_Signal', 0))
                macd_hist = float(last_candle.get('MACD_Hist', 0))
                
                # حساب نتيجة MACD
                if abs(macd_hist) > self.thresholds['MACD']['min_hist']:
                    if macd_hist > 0:  # إشارة إيجابية
                        hist_strength = min(macd_hist / 0.1, 1.0)  # تطبيع القوة
                        macd_score = 50 + (hist_strength * 50)
                    else:  # إشارة سلبية
                        hist_strength = min(abs(macd_hist) / 0.1, 1.0)
                        macd_score = 50 - (hist_strength * 50)
                else:
                    macd_score = 50  # منطقة متعادلة
                
                total_score += macd_score * self.technical_weights['MACD']
            
            # حساب قوة الاتجاه
            if isinstance(signals, list) and len(signals) > 1:
                current_close = float(signals[-1].get('close', 0))
                prev_close = float(signals[-2].get('close', 0))
                
                if current_close > prev_close:
                    strength_score = 60
                elif current_close < prev_close:
                    strength_score = 40
                else:
                    strength_score = 50
                
                total_score += strength_score * self.technical_weights['Strength']
            
            return total_score
            
        except Exception as e:
            logging.error(f"خطأ في تحليل الإشارات الفنية: {str(e)}")
            return 50  # قيمة متعادلة في حالة الخطأ
    
    def _analyze_pattern_signals(self, signals: Dict[str, float]) -> float:
        """تحليل إشارات الأنماط مع فلترة الإشارات الضعيفة"""
        try:
            total_score = 0
            total_weight = 0
            
            for pattern in signals:
                strength = float(pattern.get('strength', 0))
                confidence = float(pattern.get('confidence', 0))
                
                # فلترة الأنماط الضعيفة
                if (strength >= self.thresholds['Pattern']['min_strength'] and 
                    confidence >= self.thresholds['Pattern']['min_confidence']):
                    
                    # حساب وزن النمط (0-1)
                    pattern_weight = (strength * 0.6 + confidence * 0.4)
                    
                    # حساب النتيجة بناءً على نوع النمط وقوته
                    if pattern['type'] == 'bullish':
                        total_score += pattern_weight * 50
                    elif pattern['type'] == 'bearish':
                        total_score -= pattern_weight * 50
                    
                    total_weight += pattern_weight
            
            # إرجاع المتوسط المرجح للأنماط
            if total_weight > 0:
                return total_score / total_weight
            return 0
            
        except Exception as e:
            logging.error(f"خطأ في تحليل الأنماط: {str(e)}")
            return 0
    
    def _generate_decision_message(self, decision: str, confidence: float,
                                 market_state: str, technical_score: float,
                                 market_score: float, pattern_score: float,
                                 risk_score: float) -> str:
        """توليد رسالة تفصيلية عن القرار"""
        message = f"القرار: {decision} (الثقة: {confidence:.1%})\n"
        message += f"حالة السوق: {market_state}\n"
        message += f"التحليل الفني: {technical_score:.1f}%\n"
        message += f"قوة السوق: {market_score:.1f}%\n"
        message += f"قوة الأنماط: {pattern_score:.1f}%\n"
        message += f"مستوى المخاطرة: {risk_score:.1f}%"
        return message
    
    def _create_neutral_decision(self, reason: str) -> Dict[str, Any]:
        """إنشاء قرار محايد"""
        return {
            'decision': 'انتظار',
            'confidence': 0.0,
            'technical_score': 50,
            'market_score': 50,
            'pattern_score': 50,
            'risk_score': 50,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward_ratio': 0,
            'message': f"لا يمكن اتخاذ قرار: {reason}",
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_indicators_agreement(self, technical_score: float, market_score: float, pattern_score: float) -> float:
        """
        حساب مدى اتفاق المؤشرات المختلفة
        
        المعلمات:
            technical_score: نتيجة التحليل الفني
            market_score: نتيجة تحليل السوق
            pattern_score: نتيجة تحليل الأنماط
            
        العوائد:
            float: درجة الاتفاق (0-1)
        """
        # تحويل النتائج إلى إشارات (-1, 0, 1)
        def get_signal(score):
            if score > 0.65:
                return 1
            elif score < 0.35:
                return -1
            return 0
        
        signals = [
            get_signal(technical_score),
            get_signal(market_score),
            get_signal(pattern_score)
        ]
        
        # حساب عدد الإشارات غير الصفرية
        non_zero_signals = [s for s in signals if s != 0]
        if not non_zero_signals:
            return 0.5
        
        # حساب نسبة الاتفاق
        agreement = sum(1 for s in non_zero_signals if s == non_zero_signals[0])
        return agreement / len(non_zero_signals)
