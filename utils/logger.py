import logging
import os
from datetime import datetime
from config import LOGS_DIR

class CryptoLogger:
    def __init__(self):
        # إنشاء مسار ملف السجل
        log_file = os.path.join(LOGS_DIR, f'crypto_{datetime.now().strftime("%Y%m%d")}.log')
        
        # إعداد المسجل
        self.logger = logging.getLogger('CryptoAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # إعداد معالج الملف
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # إعداد معالج وحدة التحكم
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # تنسيق السجل
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # إضافة المعالجات
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """تسجيل معلومة"""
        self.logger.info(message)
    
    def warning(self, message):
        """تسجيل تحذير"""
        self.logger.warning(message)
    
    def error(self, message):
        """تسجيل خطأ"""
        self.logger.error(message)
    
    def debug(self, message):
        """تسجيل معلومة تصحيح"""
        self.logger.debug(message)
    
    def log_signal(self, pair, signal_data):
        """تسجيل إشارة تداول"""
        self.logger.info(f"Trading Signal for {pair}: {signal_data}")
    
    def log_error(self, error, context=""):
        """تسجيل خطأ مع السياق"""
        self.logger.error(f"Error in {context}: {str(error)}")
    
    def log_market_data(self, pair, data_summary):
        """تسجيل ملخص بيانات السوق"""
        self.logger.info(f"Market Data for {pair}: {data_summary}")
    
    def log_prediction(self, pair, current_price, predicted_price):
        """تسجيل التنبؤ"""
        self.logger.info(
            f"Price Prediction for {pair}: "
            f"Current={current_price:.8f}, "
            f"Predicted={predicted_price:.8f}, "
            f"Change={((predicted_price-current_price)/current_price*100):.2f}%"
        )
