import os
from binance.client import Client
from dotenv import load_dotenv
import logging

# تهيئة التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_binance_api():
    """اختبار الاتصال بـ Binance API"""
    try:
        # تحميل مفاتيح API
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("يجب تحديد BINANCE_API_KEY و BINANCE_API_SECRET في ملف .env")
        
        logger.info(f"API Key: {api_key[:5]}...{api_key[-5:]}")
        
        # إنشاء العميل
        client = Client(api_key, api_secret)
        
        # التحقق من حالة النظام
        status = client.get_system_status()
        logger.info(f"حالة النظام: {status}")
        
        # جلب معلومات السوق
        exchange_info = client.get_exchange_info()
        logger.info(f"عدد الأزواج: {len(exchange_info['symbols'])}")
        
        # جلب بيانات BTCUSDT
        klines = client.get_klines(
            symbol='BTCUSDT',
            interval=Client.KLINE_INTERVAL_1HOUR,
            limit=10
        )
        logger.info(f"عدد الشموع: {len(klines)}")
        
        logger.info("تم الاتصال بنجاح!")
        return True
        
    except Exception as e:
        logger.error(f"خطأ في الاتصال: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    test_binance_api()
