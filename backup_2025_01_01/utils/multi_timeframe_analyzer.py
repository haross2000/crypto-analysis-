import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .technical_analysis import TechnicalAnalyzer
from .data_collector import DataCollector
from config import TRADING_CONFIG

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = {
            '5m': {'weight': 0.1, 'periods': 288},   # 24 ساعة
            '15m': {'weight': 0.15, 'periods': 192},  # 48 ساعة
            '1h': {'weight': 0.25, 'periods': 168},   # 7 أيام
            '4h': {'weight': 0.25, 'periods': 180},   # 30 يوم
            '1d': {'weight': 0.25, 'periods': 100}    # 100 يوم
        }
        self.technical_analyzer = TechnicalAnalyzer()
        self.data_collector = DataCollector()
        self.global_store = {
            'market_data': {},
            'analysis_results': {},
            'last_update': {}
        }
        
    def analyze_all_timeframes(self, pair):
        """تحليل جميع الفترات الزمنية"""
        try:
            results = {}
            
            for timeframe in self.timeframes:
                # جلب البيانات
                df = self.data_collector.get_market_data(
                    pair, 
                    timeframe=timeframe,
                    limit=self.timeframes[timeframe]['periods']
                )
                
                if df is not None and not df.empty:
                    # حساب المؤشرات الفنية
                    df = self.technical_analyzer.calculate_all_indicators(df)
                    
                    if df is not None:
                        # حفظ البيانات في المخزن العالمي
                        if pair not in self.global_store['market_data']:
                            self.global_store['market_data'][pair] = {}
                        self.global_store['market_data'][pair][timeframe] = df
                        
                        # تحليل كل فترة زمنية
                        results[timeframe] = {
                            'trend': self.technical_analyzer.analyze_trend(df),
                            'support_resistance': self.technical_analyzer.analyze_support_resistance(df),
                            'signal_strength': self.technical_analyzer.get_signal_strength(df),
                            'last_price': df['close'].iloc[-1],
                            'volume_24h': df['volume'].iloc[-24:].sum() if timeframe == '1h' else None
                        }
            
            # حفظ نتائج التحليل في المخزن العالمي
            if pair not in self.global_store['analysis_results']:
                self.global_store['analysis_results'][pair] = {}
            self.global_store['analysis_results'][pair] = results
            self.global_store['last_update'][pair] = datetime.now()
            
            return self.combine_timeframe_analysis(results)
            
        except Exception as e:
            print(f"Error in multi-timeframe analysis: {str(e)}")
            return None
    
    def combine_timeframe_analysis(self, results):
        """دمج نتائج التحليل من جميع الفترات الزمنية"""
        try:
            if not results:
                return None
            
            timeframes = ['5m', '15m', '1h', '4h', '1d']
            weights = {'5m': 0.1, '15m': 0.15, '1h': 0.2, '4h': 0.25, '1d': 0.3}
            
            # تجميع التحليل
            trend_score = 0
            volume_score = 0
            momentum_score = 0
            support_levels = []
            resistance_levels = []
            
            for tf in timeframes:
                if tf in results:
                    analysis = results[tf]
                    weight = weights[tf]
                    
                    # تحليل الاتجاه
                    if 'trend' in analysis and analysis['trend']:
                        trend = analysis['trend'].get('trend', 'متذبذب')
                        if trend == 'صاعد قوي':
                            trend_score += weight
                        elif trend == 'صاعد':
                            trend_score += weight * 0.5
                        elif trend == 'هابط قوي':
                            trend_score -= weight
                        elif trend == 'هابط':
                            trend_score -= weight * 0.5
                    
                    # تحليل الحجم
                    if 'volume_24h' in analysis and analysis['volume_24h']:
                        volume = analysis['volume_24h']
                        avg_volume = 1000000  # حجم متوسط افتراضي
                        if volume > avg_volume * 2:
                            volume_score += weight
                        elif volume < avg_volume * 0.5:
                            volume_score -= weight
                    
                    # تحليل الزخم
                    if 'signal_strength' in analysis:
                        momentum_score += analysis['signal_strength'] * weight
                    
                    # تحليل الدعم والمقاومة
                    if 'support_resistance' in analysis and analysis['support_resistance']:
                        sr = analysis['support_resistance']
                        if 'support_levels' in sr and sr['support_levels']:
                            support_levels.extend([
                                {'level': level, 'timeframe': tf, 'strength': strength}
                                for level, strength in zip(sr['support_levels'], sr['support_strengths'])
                            ])
                        if 'resistance_levels' in sr and sr['resistance_levels']:
                            resistance_levels.extend([
                                {'level': level, 'timeframe': tf, 'strength': strength}
                                for level, strength in zip(sr['resistance_levels'], sr['resistance_strengths'])
                            ])
            
            # تحديد الاتجاه النهائي
            if trend_score > 0.5:
                trend = 'صاعد قوي'
                signal = 'شراء'
            elif trend_score > 0.2:
                trend = 'صاعد'
                signal = 'شراء ضعيف'
            elif trend_score < -0.5:
                trend = 'هابط قوي'
                signal = 'بيع'
            elif trend_score < -0.2:
                trend = 'هابط'
                signal = 'بيع ضعيف'
            else:
                trend = 'متذبذب'
                signal = 'انتظار'
            
            # حساب قوة الإشارة
            strength = abs(trend_score) * 0.4 + abs(volume_score) * 0.3 + abs(momentum_score) * 0.3
            
            return {
                'trend': trend,
                'signal': signal,
                'strength': min(abs(strength) * 100, 100),  # تحويل إلى نسبة مئوية
                'support_levels': sorted(support_levels, key=lambda x: x['level']),
                'resistance_levels': sorted(resistance_levels, key=lambda x: x['level'], reverse=True)
            }
            
        except Exception as e:
            print(f"Error combining timeframe analysis: {str(e)}")
            return None
    
    def get_entry_points(self, analysis):
        """تحديد نقاط الدخول المثالية"""
        try:
            if not analysis:
                return None
            
            current_price = None
            for tf_result in analysis['timeframe_analysis'].values():
                if tf_result and 'last_price' in tf_result:
                    current_price = tf_result['last_price']
                    break
            
            if not current_price:
                return None
            
            entry_points = []
            
            # تحديد نقاط الدخول بناءً على الاتجاه
            if analysis['overall_trend'] in ['STRONG_UP', 'UP']:
                # البحث عن أقرب مستوى دعم
                for support in analysis['support_levels']:
                    if support < current_price:
                        entry_points.append({
                            'price': support,
                            'type': 'BUY',
                            'strength': 'STRONG' if analysis['trend_strength'] > 0.7 else 'MODERATE'
                        })
            
            elif analysis['overall_trend'] in ['STRONG_DOWN', 'DOWN']:
                # البحث عن أقرب مستوى مقاومة
                for resistance in analysis['resistance_levels']:
                    if resistance > current_price:
                        entry_points.append({
                            'price': resistance,
                            'type': 'SELL',
                            'strength': 'STRONG' if analysis['trend_strength'] > 0.7 else 'MODERATE'
                        })
            
            return entry_points
            
        except Exception as e:
            print(f"Error calculating entry points: {str(e)}")
            return None
    
    def get_trade_recommendation(self, pair):
        """الحصول على توصية التداول النهائية"""
        try:
            # تحليل جميع الفترات الزمنية
            analysis = self.analyze_all_timeframes(pair)
            if not analysis:
                return None
            
            # تحديد نقاط الدخول
            entry_points = self.get_entry_points(analysis)
            
            # تحديد التوصية النهائية
            signal_type = 'BUY' if analysis['signal_strength'] > TRADING_CONFIG['MIN_CONFIDENCE']/100 else \
                         'SELL' if analysis['signal_strength'] < -TRADING_CONFIG['MIN_CONFIDENCE']/100 else \
                         'HOLD'
            
            return {
                'pair': pair,
                'signal': signal_type,
                'confidence': abs(analysis['signal_strength'] * 100),
                'trend': analysis['overall_trend'],
                'trend_strength': analysis['trend_strength'] * 100,
                'entry_points': entry_points,
                'support_levels': analysis['support_levels'],
                'resistance_levels': analysis['resistance_levels'],
                'analysis_by_timeframe': analysis['timeframe_analysis']
            }
            
        except Exception as e:
            print(f"Error generating trade recommendation: {str(e)}")
            return None

    def combine_timeframe_analysis(self, pair):
        """دمج تحليل الإطارات الزمنية المختلفة"""
        try:
            timeframes = ['5m', '15m', '1h', '4h', '1d']
            combined_analysis = {
                'trend': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'momentum': {'strong': 0, 'weak': 0, 'neutral': 0},
                'volume': {'high': 0, 'low': 0, 'normal': 0},
                'support_resistance': {
                    'support': [],
                    'resistance': [],
                    'support_strength': 0,
                    'resistance_strength': 0
                },
                'patterns': []
            }
            
            weights = {'5m': 0.1, '15m': 0.15, '1h': 0.2, '4h': 0.25, '1d': 0.3}
            
            for tf in timeframes:
                key = f"{pair}_{tf}"
                if key in self.global_store['analysis_results']:
                    analysis = self.global_store['analysis_results'][key]
                    if analysis is None:
                        continue
                        
                    weight = weights[tf]
                    
                    # تحليل الاتجاه
                    if 'trend' in analysis:
                        trend = analysis['trend'].get('trend', 'neutral')
                        if trend == 'STRONG_UP':
                            combined_analysis['trend']['bullish'] += weight
                        elif trend == 'STRONG_DOWN':
                            combined_analysis['trend']['bearish'] += weight
                        else:
                            combined_analysis['trend']['neutral'] += weight
                    
                    # تحليل الزخم
                    if 'momentum' in analysis:
                        momentum = analysis['momentum'].get('momentum', 'neutral')
                        if momentum == 'strong':
                            combined_analysis['momentum']['strong'] += weight
                        elif momentum == 'weak':
                            combined_analysis['momentum']['weak'] += weight
                        else:
                            combined_analysis['momentum']['neutral'] += weight
                    
                    # تحليل الحجم
                    if 'volume' in analysis:
                        volume = analysis['volume'].get('volume_trend', 'normal')
                        if volume == 'high':
                            combined_analysis['volume']['high'] += weight
                        elif volume == 'low':
                            combined_analysis['volume']['low'] += weight
                        else:
                            combined_analysis['volume']['normal'] += weight
                    
                    # تحليل الدعم والمقاومة
                    if 'support_resistance' in analysis:
                        sr = analysis['support_resistance']
                        if sr and 'support' in sr and sr['support'] is not None:
                            combined_analysis['support_resistance']['support'].append({
                                'level': sr['support'],
                                'timeframe': tf,
                                'strength': sr.get('support_strength', 0)
                            })
                            combined_analysis['support_resistance']['support_strength'] += \
                                sr.get('support_strength', 0) * weight
                        
                        if sr and 'resistance' in sr and sr['resistance'] is not None:
                            combined_analysis['support_resistance']['resistance'].append({
                                'level': sr['resistance'],
                                'timeframe': tf,
                                'strength': sr.get('resistance_strength', 0)
                            })
                            combined_analysis['support_resistance']['resistance_strength'] += \
                                sr.get('resistance_strength', 0) * weight
                    
                    # تحليل الأنماط
                    if 'patterns' in analysis and analysis['patterns']:
                        for pattern in analysis['patterns']:
                            if pattern:
                                pattern['timeframe'] = tf
                                pattern['weight'] = weight
                                combined_analysis['patterns'].append(pattern)
            
            # تطبيع النتائج
            trend_sum = sum(combined_analysis['trend'].values())
            momentum_sum = sum(combined_analysis['momentum'].values())
            volume_sum = sum(combined_analysis['volume'].values())
            
            if trend_sum > 0:
                for k in combined_analysis['trend']:
                    combined_analysis['trend'][k] /= trend_sum
            
            if momentum_sum > 0:
                for k in combined_analysis['momentum']:
                    combined_analysis['momentum'][k] /= momentum_sum
            
            if volume_sum > 0:
                for k in combined_analysis['volume']:
                    combined_analysis['volume'][k] /= volume_sum
            
            # ترتيب مستويات الدعم والمقاومة
            if combined_analysis['support_resistance']['support']:
                combined_analysis['support_resistance']['support'].sort(
                    key=lambda x: (x['strength'], -abs(x['level'])), 
                    reverse=True
                )
            
            if combined_analysis['support_resistance']['resistance']:
                combined_analysis['support_resistance']['resistance'].sort(
                    key=lambda x: (x['strength'], abs(x['level'])), 
                    reverse=True
                )
            
            # ترتيب الأنماط
            if combined_analysis['patterns']:
                combined_analysis['patterns'].sort(
                    key=lambda x: (x['weight'], x.get('confidence', 0)), 
                    reverse=True
                )
            
            return combined_analysis
        
        except Exception as e:
            print(f"Error combining timeframe analysis: {str(e)}")
            return None
