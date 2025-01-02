import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class RiskManager:
    def __init__(self, risk_reward_min: float = 1.5, max_risk_percent: float = 2.0):
        """
        تهيئة مدير المخاطر
        
        Args:
            risk_reward_min: الحد الأدنى لنسبة المخاطرة/المكافأة (افتراضياً 1.5)
            max_risk_percent: النسبة المئوية القصوى للمخاطرة من رأس المال (افتراضياً 2%)
        """
        self.risk_reward_min = risk_reward_min
        self.max_risk_percent = max_risk_percent
    
    def calculate_entry_points(self, df: pd.DataFrame, signals: Dict) -> List[Dict]:
        """
        حساب نقاط الدخول والمخاطر المرتبطة بها
        
        Args:
            df: إطار البيانات مع أسعار OHLCV
            signals: إشارات التداول من المحلل الفني
        
        Returns:
            قائمة بنقاط الدخول، كل منها يحتوي على:
            - price: سعر الدخول
            - stop_loss: مستوى وقف الخسارة
            - take_profit: مستوى هدف الربح
            - risk: قيمة المخاطرة
            - reward: قيمة المكافأة المحتملة
            - risk_reward_ratio: نسبة المخاطرة/المكافأة
        """
        entry_points = []
        
        if df.empty or not signals:
            return entry_points
        
        # الحصول على آخر سعر
        current_price = df['close'].iloc[-1]
        
        # حساب اتجاه السوق
        trend = signals.get('trend', 'neutral')
        
        if trend == 'bullish':
            # في الاتجاه الصعودي
            stop_loss = current_price * 0.98  # وقف خسارة 2% تحت السعر الحالي
            take_profit = current_price * (1 + (self.risk_reward_min * 0.02))  # هدف ربح بناءً على نسبة المخاطرة/المكافأة
        elif trend == 'bearish':
            # في الاتجاه الهبوطي
            stop_loss = current_price * 1.02  # وقف خسارة 2% فوق السعر الحالي
            take_profit = current_price * (1 - (self.risk_reward_min * 0.02))  # هدف ربح بناءً على نسبة المخاطرة/المكافأة
        else:
            # في حالة عدم وجود اتجاه واضح
            return entry_points
        
        # حساب المخاطر والمكافأة
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # إضافة نقطة الدخول إذا كانت نسبة المخاطرة/المكافأة مقبولة
        if risk_reward_ratio >= self.risk_reward_min:
            entry_points.append({
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk': risk,
                'reward': reward,
                'risk_reward_ratio': risk_reward_ratio,
                'trend': trend
            })
        
        return entry_points
