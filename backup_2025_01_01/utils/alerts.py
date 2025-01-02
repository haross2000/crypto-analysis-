import json
import os
from datetime import datetime
from config import DATA_DIR, TRADING_CONFIG

class AlertSystem:
    def __init__(self):
        self.alerts_file = os.path.join(DATA_DIR, 'alerts.json')
        self.active_alerts = self.load_alerts()
    
    def load_alerts(self):
        """تحميل التنبيهات المحفوظة"""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading alerts: {str(e)}")
            return {}
    
    def save_alerts(self):
        """حفظ التنبيهات"""
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump(self.active_alerts, f, indent=4)
        except Exception as e:
            print(f"Error saving alerts: {str(e)}")
    
    def add_price_alert(self, pair, price, alert_type='above'):
        """إضافة تنبيه سعر"""
        if pair not in self.active_alerts:
            self.active_alerts[pair] = {'price_alerts': []}
        
        alert = {
            'price': price,
            'type': alert_type,
            'created_at': datetime.now().isoformat(),
            'triggered': False
        }
        
        self.active_alerts[pair]['price_alerts'].append(alert)
        self.save_alerts()
    
    def add_signal_alert(self, pair, signal_type):
        """إضافة تنبيه إشارة"""
        if pair not in self.active_alerts:
            self.active_alerts[pair] = {'signal_alerts': []}
        elif 'signal_alerts' not in self.active_alerts[pair]:
            self.active_alerts[pair]['signal_alerts'] = []
        
        alert = {
            'type': signal_type,
            'created_at': datetime.now().isoformat(),
            'triggered': False
        }
        
        self.active_alerts[pair]['signal_alerts'].append(alert)
        self.save_alerts()
    
    def check_alerts(self, pair, current_price, signal=None):
        """التحقق من التنبيهات"""
        triggered_alerts = []
        
        if pair in self.active_alerts:
            # التحقق من تنبيهات السعر
            if 'price_alerts' in self.active_alerts[pair]:
                for alert in self.active_alerts[pair]['price_alerts']:
                    if not alert['triggered']:
                        if (alert['type'] == 'above' and current_price >= alert['price']) or \
                           (alert['type'] == 'below' and current_price <= alert['price']):
                            alert['triggered'] = True
                            triggered_alerts.append({
                                'type': 'price',
                                'message': f"{pair} reached {alert['type']} {alert['price']}"
                            })
            
            # التحقق من تنبيهات الإشارة
            if signal and 'signal_alerts' in self.active_alerts[pair]:
                for alert in self.active_alerts[pair]['signal_alerts']:
                    if not alert['triggered'] and alert['type'] == signal:
                        alert['triggered'] = True
                        triggered_alerts.append({
                            'type': 'signal',
                            'message': f"{pair} generated {signal} signal"
                        })
        
        if triggered_alerts:
            self.save_alerts()
        
        return triggered_alerts
    
    def remove_triggered_alerts(self):
        """إزالة التنبيهات المنتهية"""
        for pair in self.active_alerts:
            if 'price_alerts' in self.active_alerts[pair]:
                self.active_alerts[pair]['price_alerts'] = [
                    alert for alert in self.active_alerts[pair]['price_alerts']
                    if not alert['triggered']
                ]
            
            if 'signal_alerts' in self.active_alerts[pair]:
                self.active_alerts[pair]['signal_alerts'] = [
                    alert for alert in self.active_alerts[pair]['signal_alerts']
                    if not alert['triggered']
                ]
        
        self.save_alerts()
    
    def get_active_alerts(self, pair=None):
        """الحصول على التنبيهات النشطة"""
        if pair:
            return self.active_alerts.get(pair, {})
        return self.active_alerts
