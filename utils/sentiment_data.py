from datetime import datetime, timedelta
import logging
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

class CryptoDataCollector:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.binance_base = "https://api.binance.com/api/v3"
        self.blockchain_base = "https://blockchain.info"
        self.glassnode_base = "https://api.glassnode.com/v1"
        self.coinmarketcap_base = "https://pro-api.coinmarketcap.com/v1"
        
    def get_binance_data(self, symbol="BTCUSDT"):
        """الحصول على بيانات من Binance"""
        try:
            # بيانات السعر الحالي
            ticker_url = f"{self.binance_base}/ticker/24hr"
            params = {'symbol': symbol}
            response = requests.get(ticker_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # بيانات الأوردرات
                depth_url = f"{self.binance_base}/depth"
                depth_params = {'symbol': symbol, 'limit': 100}
                depth_response = requests.get(depth_url, params=depth_params)
                depth_data = depth_response.json() if depth_response.status_code == 200 else {}
                
                return {
                    'price': float(data.get('lastPrice', 0)),
                    'volume': float(data.get('volume', 0)),
                    'price_change': float(data.get('priceChange', 0)),
                    'price_change_percent': float(data.get('priceChangePercent', 0)),
                    'weighted_avg_price': float(data.get('weightedAvgPrice', 0)),
                    'trades_count': int(data.get('count', 0)),
                    'order_book': {
                        'bids_count': len(depth_data.get('bids', [])),
                        'asks_count': len(depth_data.get('asks', [])),
                        'bid_volume': sum(float(bid[1]) for bid in depth_data.get('bids', [])[:10]),
                        'ask_volume': sum(float(ask[1]) for ask in depth_data.get('asks', [])[:10])
                    }
                }
            return None
        except Exception as e:
            logging.error(f"خطأ في الحصول على بيانات Binance: {e}")
            return None

    def get_blockchain_info(self):
        """الحصول على بيانات من Blockchain.info"""
        try:
            # إحصائيات الشبكة
            stats_url = f"{self.blockchain_base}/stats?format=json"
            response = requests.get(stats_url)
            
            # بيانات المعاملات غير المؤكدة
            mempool_url = f"{self.blockchain_base}/unconfirmed-transactions?format=json"
            mempool_response = requests.get(mempool_url)
            
            # بيانات الصعوبة
            difficulty_url = f"{self.blockchain_base}/q/getdifficulty"
            difficulty_response = requests.get(difficulty_url)
            
            data = {}
            if response.status_code == 200:
                stats = response.json()
                data.update({
                    'market_price_usd': stats.get('market_price_usd', 0),
                    'hash_rate': stats.get('hash_rate', 0),
                    'total_fees_btc': stats.get('total_fees_btc', 0),
                    'n_btc_mined': stats.get('n_btc_mined', 0),
                    'n_tx': stats.get('n_tx', 0),
                    'n_blocks_mined': stats.get('n_blocks_mined', 0),
                    'minutes_between_blocks': stats.get('minutes_between_blocks', 0),
                    'totalbc': stats.get('totalbc', 0)
                })
            
            if mempool_response.status_code == 200:
                mempool = mempool_response.json()
                data['mempool'] = {
                    'pending_tx_count': len(mempool.get('txs', [])),
                    'pending_tx_value': sum(tx.get('value', 0) for tx in mempool.get('txs', []))
                }
            
            if difficulty_response.status_code == 200:
                data['difficulty'] = float(difficulty_response.text)
                
            return data
        except Exception as e:
            logging.error(f"خطأ في الحصول على بيانات Blockchain.info: {e}")
            return None

    def get_crypto_news(self):
        """الحصول على أخبار العملات المشفرة من مصادر متعددة"""
        try:
            news_data = []
            
            # CryptoCompare News
            cryptocompare_url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = requests.get(cryptocompare_url)
            if response.status_code == 200:
                data = response.json()
                news = data.get('Data', [])
                for item in news[:5]:
                    news_data.append({
                        'source': 'CryptoCompare',
                        'title': item.get('title'),
                        'url': item.get('url'),
                        'sentiment': self._analyze_sentiment(item.get('title', ''))
                    })
            
            # CoinDesk RSS
            coindesk_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
            response = requests.get(coindesk_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')
                items = soup.find_all('item')[:5]
                for item in items:
                    news_data.append({
                        'source': 'CoinDesk',
                        'title': item.find('title').text if item.find('title') else '',
                        'url': item.find('link').text if item.find('link') else '',
                        'sentiment': self._analyze_sentiment(item.find('title').text if item.find('title') else '')
                    })
            
            return news_data
        except Exception as e:
            logging.error(f"خطأ في الحصول على الأخبار: {e}")
            return None

    def _analyze_sentiment(self, text):
        """تحليل مشاعر النص"""
        text = text.lower()
        positive_words = ['bullish', 'surge', 'jump', 'gain', 'rise', 'high', 'up', 'growth', 'profit']
        negative_words = ['bearish', 'drop', 'fall', 'decline', 'low', 'down', 'crash', 'loss']
        
        positive_score = sum(1 for word in positive_words if word in text)
        negative_score = sum(1 for word in negative_words if word in text)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        return 'neutral'

    def get_market_dominance(self):
        """الحصول على هيمنة البيتكوين في السوق"""
        try:
            url = f"{self.coingecko_base}/global"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('data', {})
                return {
                    'btc_dominance': market_data.get('market_cap_percentage', {}).get('btc', 0),
                    'total_market_cap': market_data.get('total_market_cap', {}).get('usd', 0),
                    'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                    'markets_count': market_data.get('markets', 0),
                    'active_cryptocurrencies': market_data.get('active_cryptocurrencies', 0)
                }
            return None
        except Exception as e:
            logging.error(f"خطأ في الحصول على بيانات هيمنة السوق: {e}")
            return None

    def get_fear_greed_index(self):
        """الحصول على مؤشر الخوف والطمع"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url)
            data = response.json()
            
            if data['data']:
                return {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification'],
                    'timestamp': data['data'][0]['timestamp']
                }
            return None
        except Exception as e:
            logging.error(f"خطأ في الحصول على مؤشر الخوف والطمع: {e}")
            return None

    def collect_all_data(self):
        """تجميع كل البيانات المتاحة"""
        all_data = {}
        
        # بيانات Binance
        binance_data = self.get_binance_data()
        if binance_data:
            all_data['binance'] = binance_data
            
        # بيانات Blockchain.info
        blockchain_data = self.get_blockchain_info()
        if blockchain_data:
            all_data['blockchain'] = blockchain_data
            
        # الأخبار
        news_data = self.get_crypto_news()
        if news_data:
            all_data['news'] = news_data
            
        # هيمنة السوق
        dominance_data = self.get_market_dominance()
        if dominance_data:
            all_data['market_dominance'] = dominance_data
            
        # مؤشر الخوف والطمع
        fgi = self.get_fear_greed_index()
        if fgi:
            all_data['fear_greed_index'] = fgi
            
        return all_data

# مثال على الاستخدام
if __name__ == "__main__":
    collector = CryptoDataCollector()
    all_data = collector.collect_all_data()
    
    if all_data:
        # عرض بيانات Binance
        if 'binance' in all_data:
            binance = all_data['binance']
            print("\nبيانات Binance:")
            print(f"السعر الحالي: ${binance['price']:,.2f}")
            print(f"حجم التداول: {binance['volume']:,.0f} BTC")
            print(f"نسبة التغير (24 ساعة): {binance['price_change_percent']}%")
            print(f"عدد الصفقات: {binance['trades_count']:,}")
            
            if 'order_book' in binance:
                ob = binance['order_book']
                print("\nدفتر الأوامر:")
                print(f"عدد أوامر الشراء: {ob['bids_count']:,}")
                print(f"عدد أوامر البيع: {ob['asks_count']:,}")
                print(f"حجم أوامر الشراء (Top 10): {ob['bid_volume']:,.2f} BTC")
                print(f"حجم أوامر البيع (Top 10): {ob['ask_volume']:,.2f} BTC")
        
        # عرض بيانات Blockchain.info
        if 'blockchain' in all_data:
            chain = all_data['blockchain']
            print("\nبيانات Blockchain:")
            print(f"معدل التجزئة: {chain['hash_rate']:,.0f}")
            print(f"عدد المعاملات: {chain['n_tx']:,}")
            print(f"الوقت بين الكتل: {chain['minutes_between_blocks']:.1f} دقيقة")
            print(f"إجمالي البيتكوين المعدن: {chain['totalbc'] / 100000000:,.0f} BTC")
            
            if 'mempool' in chain:
                print(f"\nالمعاملات غير المؤكدة:")
                print(f"العدد: {chain['mempool']['pending_tx_count']:,}")
                print(f"القيمة: {chain['mempool']['pending_tx_value'] / 100000000:,.2f} BTC")
            
            if 'difficulty' in chain:
                print(f"الصعوبة: {chain['difficulty']:,.0f}")
        
        # عرض هيمنة السوق
        if 'market_dominance' in all_data:
            dom = all_data['market_dominance']
            print("\nإحصائيات السوق:")
            print(f"هيمنة البيتكوين: {dom['btc_dominance']:.2f}%")
            print(f"إجمالي القيمة السوقية: ${dom['total_market_cap']:,.0f}")
            print(f"عدد العملات النشطة: {dom['active_cryptocurrencies']:,}")
            print(f"عدد الأسواق: {dom['markets_count']:,}")
        
        # عرض الأخبار
        if 'news' in all_data:
            print("\nآخر الأخبار:")
            for i, news in enumerate(all_data['news'][:5], 1):
                print(f"{i}. [{news['source']}] {news['title']} ({news['sentiment']})")
        
        # عرض مؤشر الخوف والطمع
        if 'fear_greed_index' in all_data:
            fgi = all_data['fear_greed_index']
            print(f"\nمؤشر الخوف والطمع:")
            print(f"القيمة: {fgi['value']}")
            print(f"التصنيف: {fgi['classification']}")
            print(f"آخر تحديث: {datetime.fromtimestamp(int(fgi['timestamp']))}")
