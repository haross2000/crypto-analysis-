"""
جامع البيانات من Binance
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, api_key: str, api_secret: str):
        """تهيئة جامع البيانات"""
        try:
            logging.info("تهيئة اتصال Binance")
            self.client = Client(api_key, api_secret)
            
            # التحقق من صحة الاتصال
            try:
                self.client.get_system_status()
                logging.info("تم الاتصال بـ Binance بنجاح")
            except Exception as e:
                logging.error(f"فشل الاتصال بـ Binance: {str(e)}")
                self.client = None
                
        except Exception as e:
            logging.error(f"خطأ في تهيئة جامع البيانات: {str(e)}")
            self.client = None
            
    def get_candlestick_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """جلب بيانات الشموع اليابانية"""
        try:
            # تنظيف الرمز
            symbol = symbol.upper().replace('/', '')
            
            # جلب البيانات
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=300  # آخر 300 شمعة
            )
            
            if not klines:
                logging.error(f"لا توجد بيانات متاحة للزوج {symbol}")
                return pd.DataFrame()
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # تحويل التوقيت
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # تحويل الأعمدة الرقمية مع الحفاظ على الدقة
            numeric_columns = {
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float,
                'quote_asset_volume': float,
                'taker_buy_base_asset_volume': float,
                'taker_buy_quote_asset_volume': float
            }
            
            for col, dtype in numeric_columns.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            
            logging.info(f"تم جلب {len(df)} شمعة للزوج {symbol}")
            return df
            
        except Exception as e:
            logging.error(f"خطأ في جلب البيانات: {str(e)}")
            return pd.DataFrame()

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """جمع بيانات السوق"""
        try:
            logging.info(f"جاري جلب بيانات السوق للزوج {symbol}")
            
            # التأكد من صحة الرمز
            symbol = symbol.upper().strip()  # تحويل الرمز إلى أحرف كبيرة وإزالة المسافات
            logging.info(f"الرمز بعد التنسيق: {symbol}")
            
            # التحقق من وجود الرمز
            exchange_info = self.client.get_exchange_info()
            valid_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            logging.info(f"عدد الأزواج المتاحة: {len(valid_symbols)}")
            
            if symbol not in valid_symbols:
                logging.error(f"الرمز {symbol} غير صالح أو غير متداول")
                logging.error(f"الرموز المشابهة: {[s for s in valid_symbols if symbol in s]}")
                return {}
            
            # الحصول على معلومات التداول الحالية
            ticker = self.client.get_ticker(symbol=symbol)
            if not ticker:
                logging.error(f"لا توجد معلومات تداول متاحة للرمز {symbol}")
                return {}
            
            logging.info(f"تم جلب معلومات التداول: {ticker['lastPrice']}")
            
            # الحصول على معلومات السوق
            market_info = self.client.get_symbol_info(symbol)
            if not market_info:
                logging.error(f"لا توجد معلومات سوق متاحة للرمز {symbol}")
                return {}
            
            logging.info(f"تم جلب معلومات السوق: {market_info['status']}")
            
            # الحصول على عمق السوق
            depth = self.client.get_order_book(symbol=symbol)
            if not depth:
                logging.error(f"لا توجد معلومات عمق السوق متاحة للرمز {symbol}")
                return {}
            
            logging.info(f"تم جلب عمق السوق: {len(depth['bids'])} أوامر بيع، {len(depth['asks'])} أوامر شراء")
            
            # حساب مستويات الدعم والمقاومة
            support_resistance = self._calculate_support_resistance(depth)
            
            # حساب المؤشرات الإحصائية
            stats = self._calculate_market_stats(ticker, depth)
            
            market_data = {
                # معلومات السعر الأساسية
                'current_price': float(ticker['lastPrice']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'price_change_24h': float(ticker['priceChangePercent']),
                'volume_24h': float(ticker['volume']),
                
                # مستويات الدعم والمقاومة
                'support_levels': support_resistance['support'],
                'resistance_levels': support_resistance['resistance'],
                
                # معلومات السوق
                'market_cap': self._calculate_market_cap(ticker),
                'global_rank': self._get_global_rank(symbol),
                'traded_pairs': len(market_info.get('filters', [])),
                'liquidity': stats['liquidity'],
                
                # إحصائيات إضافية
                'vwap': stats['vwap'],
                'volatility': stats['volatility'],
                'turnover_rate': stats['turnover_rate'],
                'market_depth': stats['market_depth'],
                'cash_flow_indicator': stats['cash_flow_indicator'],
                
                # تحليلات متقدمة
                'short_term_forecast': self._calculate_forecast(ticker),
                'volatility_level': stats['volatility_level'],
                'buyer_seller_ratio': stats['buyer_seller_ratio'],
                'risk_level': self._calculate_risk_level(stats),
                'investment_rating': self._calculate_investment_rating(stats)
            }
            
            logging.info(f"تم جمع بيانات السوق بنجاح: {market_data['current_price']}")
            
            # إضافة تسجيل لقيم البيانات
            logging.info(f"نطاق السعر: {market_data['low_24h']:.2f} - {market_data['high_24h']:.2f}")
            logging.info(f"متوسط الحجم: {market_data['volume_24h']:.2f}")
            logging.info(f"معدل التذبذب: {market_data['volatility']:.2f}%")
            
            return market_data
            
        except Exception as e:
            logging.error(f"خطأ في جمع بيانات السوق للرمز {symbol}: {str(e)}")
            import traceback
            logging.error(f"تفاصيل الخطأ: {traceback.format_exc()}")
            return {}

    def get_historical_klines(self, symbol: str, interval: str, start_str: str, end_str: str) -> pd.DataFrame:
        """الحصول على البيانات التاريخية"""
        try:
            logging.info(f"جاري جلب البيانات التاريخية للزوج {symbol}")
            
            # التأكد من صحة الرمز
            symbol = symbol.upper().strip()  # تحويل الرمز إلى أحرف كبيرة وإزالة المسافات
            logging.info(f"الرمز بعد التنسيق: {symbol}")
            
            # التحقق من وجود الرمز
            exchange_info = self.client.get_exchange_info()
            valid_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            logging.info(f"عدد الأزواج المتاحة: {len(valid_symbols)}")
            logging.info(f"الأزواج المتاحة: {valid_symbols[:10]}...")  # عرض أول 10 أزواج
            
            if symbol not in valid_symbols:
                logging.error(f"الرمز {symbol} غير صالح أو غير متداول")
                logging.error(f"الرموز المشابهة: {[s for s in valid_symbols if symbol in s]}")
                return pd.DataFrame()
            
            logging.info(f"جاري جلب البيانات من {start_str} إلى {end_str}")
            
            # جلب البيانات التاريخية من Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=1000  # الحد الأقصى للبيانات
            )
            
            logging.info(f"تم جلب {len(klines)} شمعة")
            
            if not klines:
                logging.error(f"لا توجد بيانات متاحة للرمز {symbol}")
                return pd.DataFrame()
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # تحويل أنواع البيانات
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # إضافة تسجيل لقيم البيانات
            logging.info(f"نطاق السعر: {df['close'].min():.2f} - {df['close'].max():.2f}")
            logging.info(f"متوسط الحجم: {df['volume'].mean():.2f}")
            logging.info(f"عدد الصفقات: {df['number_of_trades'].sum()}")
            
            return df
            
        except Exception as e:
            logging.error(f"خطأ في جلب البيانات التاريخية للرمز {symbol}: {str(e)}")
            import traceback
            logging.error(f"تفاصيل الخطأ: {traceback.format_exc()}")
            return pd.DataFrame()

    def _calculate_support_resistance(self, depth: Dict) -> Dict[str, List[float]]:
        """حساب مستويات الدعم والمقاومة"""
        try:
            bids = pd.DataFrame(depth['bids'], columns=['price', 'quantity'], dtype=float)
            asks = pd.DataFrame(depth['asks'], columns=['price', 'quantity'], dtype=float)
            
            # تحديد مستويات الدعم (أعلى 3 مستويات للطلبات)
            support = bids.groupby('price')['quantity'].sum().nlargest(3).index.tolist()
            
            # تحديد مستويات المقاومة (أعلى 3 مستويات للعروض)
            resistance = asks.groupby('price')['quantity'].sum().nlargest(3).index.tolist()
            
            return {
                'support': support,
                'resistance': resistance
            }
        except Exception as e:
            logging.error(f"خطأ في حساب مستويات الدعم والمقاومة: {str(e)}")
            return {'support': [], 'resistance': []}

    def _calculate_market_stats(self, ticker: Dict, depth: Dict) -> Dict[str, float]:
        """حساب المؤشرات الإحصائية"""
        try:
            # حساب VWAP
            vwap = float(ticker['weightedAvgPrice'])
            
            # حساب التذبذب
            volatility = (float(ticker['highPrice']) - float(ticker['lowPrice'])) / vwap * 100
            
            # تحديد مستوى التذبذب
            volatility_level = self._classify_volatility(volatility)
            
            # حساب نسبة المشترين والبائعين
            bids_volume = sum(float(bid[1]) for bid in depth['bids'])
            asks_volume = sum(float(ask[1]) for ask in depth['asks'])
            buyer_seller_ratio = bids_volume / asks_volume if asks_volume > 0 else 1
            
            # حساب السيولة
            liquidity = (bids_volume + asks_volume) * float(ticker['lastPrice'])
            
            # حساب معدل الدوران
            turnover_rate = float(ticker['volume']) * float(ticker['lastPrice']) / liquidity if liquidity > 0 else 0
            
            # حساب عمق السوق
            market_depth = len(depth['bids']) + len(depth['asks'])
            
            # حساب مؤشر التدفق النقدي
            cash_flow = float(ticker['quoteVolume']) * (float(ticker['priceChangePercent']) / 100)
            
            return {
                'vwap': vwap,
                'volatility': volatility,
                'volatility_level': volatility_level,
                'buyer_seller_ratio': buyer_seller_ratio,
                'liquidity': liquidity,
                'turnover_rate': turnover_rate,
                'market_depth': market_depth,
                'cash_flow_indicator': cash_flow
            }
        except Exception as e:
            logging.error(f"خطأ في حساب المؤشرات الإحصائية: {str(e)}")
            return {
                'vwap': 0,
                'volatility': 0,
                'volatility_level': 'متوسط',
                'buyer_seller_ratio': 1,
                'liquidity': 0,
                'turnover_rate': 0,
                'market_depth': 0,
                'cash_flow_indicator': 0
            }

    def _calculate_market_cap(self, ticker: Dict) -> float:
        """حساب القيمة السوقية"""
        try:
            return float(ticker['lastPrice']) * float(ticker['volume'])
        except Exception as e:
            logging.error(f"خطأ في حساب القيمة السوقية: {str(e)}")
            return 0

    def _get_global_rank(self, symbol: str) -> int:
        """الحصول على التصنيف العالمي"""
        try:
            # هذه مجرد محاكاة للتصنيف - في التطبيق الفعلي يمكن استخدام API خارجي
            return np.random.randint(1, 100)
        except Exception as e:
            logging.error(f"خطأ في الحصول على التصنيف العالمي: {str(e)}")
            return 0

    def _calculate_forecast(self, ticker: Dict) -> Dict[str, float]:
        """حساب توقعات السعر القصيرة المدى"""
        try:
            current_price = float(ticker['lastPrice'])
            price_change = float(ticker['priceChangePercent'])
            
            # حساب التوقعات بناءً على الاتجاه الحالي والتذبذب
            forecast = {
                'optimistic': current_price * (1 + abs(price_change) / 100),
                'pessimistic': current_price * (1 - abs(price_change) / 100),
                'neutral': current_price
            }
            
            return forecast
        except Exception as e:
            logging.error(f"خطأ في حساب التوقعات: {str(e)}")
            return {'optimistic': 0, 'pessimistic': 0, 'neutral': 0}

    def _classify_volatility(self, volatility: float) -> str:
        """تصنيف مستوى التذبذب"""
        if volatility < 1:
            return 'منخفض'
        elif volatility < 3:
            return 'متوسط'
        else:
            return 'مرتفع'

    def _calculate_risk_level(self, stats: Dict) -> str:
        """حساب مستوى المخاطرة"""
        try:
            # حساب المخاطرة بناءً على التذبذب والسيولة
            risk_score = (
                float(stats['volatility']) * 0.4 +
                (1 / float(stats['liquidity']) if float(stats['liquidity']) > 0 else 1) * 0.3 +
                (1 / float(stats['market_depth']) if float(stats['market_depth']) > 0 else 1) * 0.3
            )
            
            if risk_score < 0.3:
                return 'منخفض'
            elif risk_score < 0.7:
                return 'متوسط'
            else:
                return 'مرتفع'
        except Exception as e:
            logging.error(f"خطأ في حساب مستوى المخاطرة: {str(e)}")
            return 'متوسط'

    def _calculate_investment_rating(self, stats: Dict) -> str:
        """حساب تصنيف الفرصة الاستثمارية"""
        try:
            # حساب التصنيف بناءً على عدة عوامل
            rating_score = (
                float(stats['buyer_seller_ratio']) * 0.3 +
                float(stats['turnover_rate']) * 0.3 +
                (float(stats['cash_flow_indicator']) if float(stats['cash_flow_indicator']) > 0 else 0) * 0.4
            )
            
            if rating_score < 0.3:
                return 'ضعيف'
            elif rating_score < 0.7:
                return 'جيد'
            else:
                return 'ممتاز'
        except Exception as e:
            logging.error(f"خطأ في حساب تصنيف الفرصة الاستثمارية: {str(e)}")
            return 'متوسط'

    def get_trading_pairs(self) -> List[str]:
        """الحصول على قائمة الأزواج المتاحة"""
        try:
            # التحقق من وجود العميل
            if not self.client:
                logging.error("لم يتم تهيئة اتصال Binance")
                return []
                
            # التحقق من حالة النظام
            try:
                status = self.client.get_system_status()
                if status['status'] != 0:  # 0 = normal
                    logging.error(f"نظام Binance غير متاح: {status['msg']}")
                    return []
            except Exception as e:
                logging.error(f"خطأ في التحقق من حالة النظام: {str(e)}")
                return []
            
            # جلب معلومات السوق
            try:
                exchange_info = self.client.get_exchange_info()
            except Exception as e:
                logging.error(f"خطأ في جلب معلومات السوق: {str(e)}")
                return []
            
            if not exchange_info or 'symbols' not in exchange_info:
                logging.error("تنسيق معلومات السوق غير صحيح")
                return []
            
            # تصفية الأزواج المتاحة للتداول
            pairs = [
                symbol['symbol'] for symbol in exchange_info['symbols']
                if symbol['status'] == 'TRADING' and symbol['quoteAsset'] == 'USDT'
            ]
            
            if not pairs:
                logging.warning("لم يتم العثور على أزواج تداول")
                return []
            
            logging.info(f"تم جلب {len(pairs)} زوج تداول")
            return sorted(pairs)
            
        except Exception as e:
            logging.error(f"خطأ في جلب قائمة الأزواج: {str(e)}", exc_info=True)
            return []

    def get_available_intervals(self) -> List[str]:
        """الحصول على الفترات الزمنية المتاحة"""
        return [
            '1m', '3m', '5m', '15m', '30m',  # دقائق
            '1h', '2h', '4h', '6h', '8h', '12h',  # ساعات
            '1d', '3d',  # أيام
            '1w',  # أسبوع
            '1M'   # شهر
        ]

    def test_connection(self) -> bool:
        """اختبار الاتصال بـ Binance"""
        try:
            self.client.ping()
            logging.info("تم الاتصال بـ Binance بنجاح")
            return True
        except Exception as e:
            logging.error(f"فشل الاتصال بـ Binance: {str(e)}")
            raise

    def get_latest_data(self) -> pd.DataFrame:
        """
        الحصول على أحدث البيانات من الذاكرة المؤقتة
        
        العائد:
            pd.DataFrame: إطار البيانات مع أحدث البيانات
        """
        try:
            if hasattr(self, '_cached_data') and not self._cached_data.empty:
                return self._cached_data
            
            # إذا لم تكن هناك بيانات مخزنة، نقوم بجلب بيانات جديدة
            symbol = self.default_symbol if hasattr(self, 'default_symbol') else 'BTCUSDT'
            interval = self.default_interval if hasattr(self, 'default_interval') else '1m'
            limit = self.default_limit if hasattr(self, 'default_limit') else 1000
            
            df = self.get_candlestick_data(symbol, interval)
            if not df.empty:
                self._cached_data = df
            
            return df
            
        except Exception as e:
            logging.error(f"خطأ في الحصول على أحدث البيانات: {str(e)}")
            return pd.DataFrame()

    def get_all_usdt_pairs(self) -> List[Dict[str, str]]:
        """جلب جميع أزواج USDT المتاحة"""
        try:
            logging.info("بدء جلب أزواج التداول...")

            if not self.client:
                self.client = Client(self.api_key, self.api_secret)
                logging.info("تم إنشاء اتصال جديد مع Binance")

            # جلب معلومات السوق مباشرة
            try:
                logging.info("جلب معلومات الأزواج من Binance...")
                tickers = self.client.get_all_tickers()
                logging.info(f"تم جلب {len(tickers)} زوج من Binance")
                
                # تصفية أزواج USDT فقط
                usdt_pairs = []
                for ticker in tickers:
                    symbol = ticker['symbol']
                    if symbol.endswith('USDT'):
                        try:
                            # جلب معلومات تفصيلية عن الزوج
                            price = float(ticker['price'])
                            base_asset = symbol[:-4]  # حذف 'USDT' من نهاية الرمز
                            
                            # تنسيق السعر
                            if price >= 1:
                                price_str = f"${price:,.2f}"
                            elif price >= 0.01:
                                price_str = f"${price:.4f}"
                            else:
                                price_str = f"${price:.8f}"
                            
                            # تنسيق الاسم للعرض
                            display_name = f"{base_asset}/USDT"
                            
                            pair_info = {
                                'label': f"{base_asset} ({display_name}) - {price_str}",
                                'value': symbol
                            }
                            usdt_pairs.append(pair_info)
                            logging.debug(f"تمت إضافة الزوج: {symbol} بسعر {price_str}")
                        except Exception as e:
                            logging.warning(f"خطأ في معالجة الزوج {symbol}: {str(e)}")
                            continue

                # ترتيب الأزواج
                top_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE']
                
                def sort_key(pair):
                    base_asset = pair['value'][:-4]  # حذف 'USDT'
                    if base_asset in top_coins:
                        return (0, top_coins.index(base_asset))
                    return (1, base_asset)
                
                usdt_pairs.sort(key=sort_key)
                
                pair_count = len(usdt_pairs)
                logging.info(f"تم العثور على {pair_count} زوج USDT")
                
                if pair_count == 0:
                    raise Exception("لم يتم العثور على أي أزواج USDT")
                
                return usdt_pairs

            except BinanceAPIException as e:
                logging.error(f"خطأ في Binance API: {str(e)}")
                raise Exception(f"خطأ في الاتصال مع Binance: {str(e)}")
                
        except Exception as e:
            logging.error(f"خطأ في جلب أزواج USDT: {str(e)}")
            raise
