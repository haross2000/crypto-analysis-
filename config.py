import os
from dotenv import load_dotenv
import ccxt
import time
import json
from datetime import datetime, timedelta
import threading
import pickle

# تحميل المتغيرات البيئية
load_dotenv()

# إعدادات API
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# إعدادات API
API_CONFIG = {
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
        'recvWindow': 60000
    }
}

# الإطار الزمني الافتراضي
TIMEFRAME = '1h'

# أزواج التداول المتاحة
TRADING_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
    'ADA/USDT', 'AVAX/USDT', 'MATIC/USDT', 'DOT/USDT', 'LINK/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT',
    'ETC/USDT', 'XLM/USDT', 'ALGO/USDT', 'NEAR/USDT', 'FTM/USDT'
]

# إعدادات التحليل الفني
TECHNICAL_INDICATORS = {
    'RSI_PERIOD': 14,
    'RSI_OVERBOUGHT': 75,  # زيادة مستوى ذروة الشراء
    'RSI_OVERSOLD': 25,    # خفض مستوى ذروة البيع
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'MA_FAST': 20,
    'MA_SLOW': 50,
    'MA_TREND': 200,      # إضافة المتوسط المتحرك للاتجاه طويل المدى
    'VOLUME_PERIOD': 20,
    'VOLATILITY_PERIOD': 20,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_STD': 2,
    'ATR_PERIOD': 14      # إضافة مؤشر المدى الحقيقي المتوسط
}

# إعدادات التداول
TRADING_CONFIG = {
    'MIN_CONFIDENCE': 30,  # الحد الأدنى للثقة لتوليد إشارة
    'MAX_TRADES_PER_DAY': 3,
    'MIN_VOLUME_MULTIPLIER': 1.5,
    'RISK_PERCENTAGE': 0.02,
    'MIN_RISK_REWARD': 2.0,
    'TRAILING_STOP': 0.02,    # نسبة وقف الخسارة المتحرك
    'PROFIT_TARGETS': [       # مستويات جني الأرباح
        {'target': 1.5, 'size': 0.3},  # أخذ 30% من الربح عند 1.5 ضعف المخاطرة
        {'target': 2.0, 'size': 0.3},  # أخذ 30% إضافية عند 2.0 ضعف المخاطرة
        {'target': 3.0, 'size': 0.4}   # أخذ 40% المتبقية عند 3.0 ضعف المخاطرة
    ],
    'PROFIT_TARGET': 0.02,  # نسبة هدف الربح
    'STOP_LOSS': 0.01,  # نسبة وقف الخسارة
    'MAX_SPREAD': 0.005,  # أقصى فرق بين سعر البيع والشراء
    'MIN_VOLUME': 1000000,  # الحد الأدنى لحجم التداول اليومي بالدولار
    'CACHE_DURATION': 3600,  # مدة صلاحية التخزين المؤقت (ساعة)
    'UPDATE_INTERVAL': 300,  # فترة تحديث الأزواج (5 دقائق)
    'MAX_PAIRS': 20  # الحد الأقصى لعدد الأزواج
}

# المسارات
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# إنشاء المجلدات إذا لم تكن موجودة
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# مسار ملف التخزين المؤقت
CACHE_FILE = os.path.join(DATA_DIR, 'trading_pairs_cache.pkl')

class TradingPairsManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TradingPairsManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = False
            
        if self._initialized:
            return
            
        self._initialized = True
        self.exchange = None
        self.trading_pairs = []
        self.last_update = None
        self.update_thread = None
        self.running = False
        
        # إنشاء مجلد البيانات إذا لم يكن موجوداً
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        
        # تحميل الأزواج المخزنة مؤقتاً
        self._load_cached_pairs()
        
        # بدء خيط التحديث
        self.start_update_thread()

    def _load_cached_pairs(self):
        """تحميل الأزواج المخزنة مؤقتاً"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                    if datetime.now() - cache_data['timestamp'] < timedelta(seconds=TRADING_CONFIG['CACHE_DURATION']):
                        self.trading_pairs = cache_data['pairs']
                        self.last_update = cache_data['timestamp']
                        print(f"Loaded {len(self.trading_pairs)} pairs from cache")
                        return
        except Exception as e:
            print(f"Error loading cached pairs: {str(e)}")
        
        self.trading_pairs = TRADING_PAIRS
        self.last_update = datetime.now()
    
    def _save_cached_pairs(self):
        """حفظ الأزواج في الذاكرة المؤقتة"""
        try:
            cache_data = {
                'pairs': self.trading_pairs,
                'timestamp': datetime.now()
            }
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(self.trading_pairs)} pairs to cache")
        except Exception as e:
            print(f"Error saving pairs to cache: {str(e)}")
    
    def update_trading_pairs(self):
        """تحديث قائمة أزواج التداول"""
        try:
            if not self.exchange:
                self.exchange = ccxt.binance()
            
            # الحصول على جميع الأزواج المتاحة
            markets = self.exchange.load_markets()
            
            # تصفية الأزواج للحصول على أزواج USDT فقط
            usdt_pairs = [
                symbol for symbol in markets.keys()
                if symbol.endswith('/USDT') and 
                not any(x in symbol for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/'])
            ]
            
            # ترتيب الأزواج أبجدياً
            usdt_pairs.sort()
            
            self.trading_pairs = usdt_pairs
            self.last_update = time.time()
            
            # حفظ الأزواج في الذاكرة المؤقتة
            self._save_cached_pairs()
            
            return True
            
        except Exception as e:
            print(f"Error updating trading pairs: {str(e)}")
            return False
    
    def start_update_thread(self):
        """بدء خيط تحديث الأزواج"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
    
    def _update_loop(self):
        """حلقة تحديث الأزواج"""
        while self.running:
            try:
                # تحديث الأزواج إذا انتهت صلاحية التخزين المؤقت
                if (self.last_update is None or 
                    (datetime.now() - self.last_update).total_seconds() >= TRADING_CONFIG['UPDATE_INTERVAL']):
                    self.update_trading_pairs()
                
                # انتظار قبل التحديث التالي
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in update loop: {str(e)}")
                time.sleep(10)
    
    def stop_update_thread(self):
        """إيقاف خيط التحديث"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def get_trading_pairs(self):
        """الحصول على قائمة أزواج التداول"""
        return self.trading_pairs

# إنشاء مدير أزواج التداول
pairs_manager = TradingPairsManager()
TRADING_PAIRS = pairs_manager.get_trading_pairs()

print(f"Loaded {len(TRADING_PAIRS)} trading pairs: {TRADING_PAIRS}")
print(f"Technical indicators configuration loaded")
print(f"Trading configuration loaded")
print(f"Data directory: {DATA_DIR}")
print(f"Logs directory: {LOGS_DIR}")
print(f"Models directory: {MODELS_DIR}")
