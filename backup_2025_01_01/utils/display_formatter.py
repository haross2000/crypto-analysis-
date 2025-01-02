import json
from typing import Dict, Any

class DisplayFormatter:
    # رموز خاصة
    ARROW_UP = "↑"
    ARROW_DOWN = "↓"
    ARROW_SIDE = "↔"
    CHECK = "✓"
    CROSS = "✗"
    BULLET = "•"
    
    @staticmethod
    def _color_price_change(current: float, reference: float) -> str:
        """تلوين التغير في السعر"""
        if current > reference:
            return f"{current:,.2f}"
        elif current < reference:
            return f"{current:,.2f}"
        return f"{current:,.2f}"

    @staticmethod
    def _color_trend(trend: str) -> str:
        """تلوين الاتجاه"""
        if trend == "صاعد":
            return f"{DisplayFormatter.ARROW_UP} {trend}"
        elif trend == "هابط":
            return f"{DisplayFormatter.ARROW_DOWN} {trend}"
        return f"{DisplayFormatter.ARROW_SIDE} {trend}"

    @staticmethod
    def format_technical_analysis(data: Dict[str, Any]) -> str:
        """تنسيق بيانات التحليل الفني"""
        if not data:
            return "لا توجد بيانات للتحليل الفني"
            
        template = """
══════════ التحليل الفني ══════════
▸ الاتجاه العام: {trend}
▸ السعر الحالي: {price:,.2f}

المؤشرات الفنية:
▸ RSI: {rsi:.2f} ({rsi_signal})
▸ MACD: {macd_signal}

المتوسطات المتحركة:
▸ MA20: {ma20:,.2f}
▸ MA50: {ma50:,.2f}

نطاقات البولنجر:
▸ الحد العلوي:   {bb_upper:,.2f}
▸ الخط الأوسط:   {bb_middle:,.2f}
▸ الحد السفلي:   {bb_lower:,.2f}
"""
        trend_symbol = ""
        if data['trend'] == "صاعد":
            trend_symbol = DisplayFormatter.ARROW_UP
        elif data['trend'] == "هابط":
            trend_symbol = DisplayFormatter.ARROW_DOWN
        else:
            trend_symbol = DisplayFormatter.ARROW_SIDE
            
        return template.format(
            trend=f"{trend_symbol} {data['trend']}",
            price=data['current_price'],
            rsi=data['rsi'],
            rsi_signal=data['rsi_signal'],
            macd_signal=data['macd_signal'],
            ma20=data['ma20'],
            ma50=data['ma50'],
            bb_upper=data['bb_upper'],
            bb_middle=data['bb_middle'],
            bb_lower=data['bb_lower']
        )

    @staticmethod
    def format_patterns(data: Dict[str, Any]) -> str:
        """تنسيق بيانات الأنماط الفنية"""
        if not data:
            return "لا توجد أنماط فنية مكتشفة"
            
        output = "\n══════════ الأنماط الفنية ══════════\n"
        
        pattern_names = {
            'double_top': 'القمة المزدوجة',
            'double_bottom': 'القاع المزدوج',
            'head_and_shoulders': 'الرأس والكتفين'
        }
        
        for pattern_key, pattern_data in data.items():
            pattern_name = pattern_names.get(pattern_key, pattern_key)
            output += f"\n▸ نمط {pattern_name}:\n"
            output += f"  {DisplayFormatter.BULLET} مستوى الثقة: {pattern_data['confidence']}%\n"
            output += f"  {DisplayFormatter.BULLET} مستوى السعر: {pattern_data['price_level']:,.2f}\n"
            output += f"  {DisplayFormatter.BULLET} الهدف: {pattern_data['target']:,.2f}\n"
        
        return output

    @staticmethod
    def format_trading_signals(decision: Any) -> str:
        """تنسيق إشارات التداول"""
        if not decision:
            return "لا توجد إشارات تداول حالياً"
            
        template = """
══════════ إشارات التداول ══════════
▸ نوع الإشارة: {signal_type}
▸ مستوى الثقة: {confidence:.1f}%

نقاط الدخول والخروج:
▸ سعر الدخول: {entry:,.2f}
▸ وقف الخسارة: {stop:,.2f}

مستويات الربح:
{take_profits}

معلومات إضافية:
▸ نسبة المخاطرة/المكافأة: {risk_reward:.2f}
▸ الإطار الزمني: {timeframe}
▸ تأكيد الأنماط: {pattern_conf}
▸ تأكيد الحجم: {volume_conf}

ملاحظات:
{notes}
"""
        if hasattr(decision, 'signal_type'):
            take_profits = "\n".join([
                f"▸ الهدف {i+1}: {tp:,.2f}" 
                for i, tp in enumerate(decision.take_profit)
            ])
            
            pattern_conf = DisplayFormatter.CHECK if decision.pattern_confirmation else DisplayFormatter.CROSS
            volume_conf = DisplayFormatter.CHECK if decision.volume_confirmation else DisplayFormatter.CROSS
            
            return template.format(
                signal_type=decision.signal_type.value,
                confidence=decision.confidence,
                entry=decision.entry_price,
                stop=decision.stop_loss,
                take_profits=take_profits,
                risk_reward=decision.risk_reward_ratio,
                timeframe=decision.timeframe,
                pattern_conf=pattern_conf,
                volume_conf=volume_conf,
                notes="\n".join([f"▸ {note}" for note in decision.additional_notes])
            )
        
        return "لا توجد إشارات تداول حالياً"

