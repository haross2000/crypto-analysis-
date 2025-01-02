"""
نموذج التعلم العميق المتقدم للتنبؤ بأسعار العملات المشفرة
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Concatenate, Bidirectional, Conv1D, MaxPooling1D, Flatten, Add, SpatialDropout1D, LeakyReLU, MultiHeadAttention
import talib as ta
import logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.regularizers import l1_l2
from utils.sentiment_data import CryptoDataCollector

class DeepLearningPredictor:
    def __init__(self, sequence_length=60, learning_rate=0.001):
        """تهيئة المتغيرات الأساسية"""
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
        self.learning_rate = learning_rate
        self.training_history = None
        
        # تعريف المؤشرات الفنية
        self.technical_indicators = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50',  # 4
            'ema_5', 'ema_10', 'ema_20', 'ema_50',  # 8
            'rsi',  # 9
            'macd', 'macd_signal', 'macd_hist',  # 12
            'mom', 'roc',  # 14
            'bbands_upper', 'bbands_middle', 'bbands_lower',  # 17
            'atr',  # 18
            'obv', 'ad', 'adosc',  # 21
            'adx', 'cci', 'dx',  # 24
            'willr', 'ultosc', 'trix',  # 27
            'stoch', 'stoch_signal'  # 29
        ]
        
        # تهيئة المقاييس
        self.price_scalers = [StandardScaler() for _ in range(4)]  # close, high, low, open
        self.volume_scalers = [StandardScaler() for _ in range(1)]  # volume
        self.technical_scalers = [StandardScaler() for _ in range(len(self.technical_indicators))]
        self.market_scalers = None  # سيتم تهيئته عند معرفة عدد الأعمدة
        
        # تهيئة المقاييس القديمة للتوافق مع الكود القديم
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.technical_scaler = StandardScaler()
        self.market_scaler = StandardScaler()
        
    def prepare_market_data(self):
        """جمع وتجهيز بيانات السوق والمشاعر"""
        try:
            collector = CryptoDataCollector()
            market_data = collector.collect_all_data()
            
            if market_data:
                # تجهيز بيانات Binance
                binance_features = {}
                if 'binance' in market_data:
                    binance = market_data['binance']
                    binance_features.update({
                        'price': binance['price'],
                        'volume': binance['volume'],
                        'price_change': binance['price_change'],
                        'price_change_percent': binance['price_change_percent'],
                        'weighted_avg_price': binance['weighted_avg_price'],
                        'trades_count': binance['trades_count']
                    })
                    
                    if 'order_book' in binance:
                        ob = binance['order_book']
                        binance_features.update({
                            'bids_count': ob['bids_count'],
                            'asks_count': ob['asks_count'],
                            'bid_volume': ob['bid_volume'],
                            'ask_volume': ob['ask_volume']
                        })
                
                # تجهيز بيانات Blockchain
                blockchain_features = {}
                if 'blockchain' in market_data:
                    chain = market_data['blockchain']
                    blockchain_features.update({
                        'hash_rate': chain['hash_rate'],
                        'total_fees_btc': chain['total_fees_btc'],
                        'n_tx': chain['n_tx'],
                        'minutes_between_blocks': chain['minutes_between_blocks']
                    })
                    
                    if 'mempool' in chain:
                        blockchain_features.update({
                            'pending_tx_count': chain['mempool']['pending_tx_count'],
                            'pending_tx_value': chain['mempool']['pending_tx_value']
                        })
                    
                    if 'difficulty' in chain:
                        blockchain_features.update({
                            'difficulty': chain['difficulty']
                        })
                
                # تجهيز بيانات هيمنة السوق
                market_dominance_features = {}
                if 'market_dominance' in market_data:
                    dom = market_data['market_dominance']
                    market_dominance_features.update({
                        'btc_dominance': dom['btc_dominance'],
                        'total_market_cap': dom['total_market_cap'],
                        'total_volume': dom['total_volume'],
                        'markets_count': dom['markets_count'],
                        'active_cryptocurrencies': dom['active_cryptocurrencies']
                    })
                
                # تجهيز بيانات المشاعر
                sentiment_features = {}
                if 'fear_greed_index' in market_data:
                    fgi = market_data['fear_greed_index']
                    sentiment_features['fear_greed_value'] = fgi['value']
                
                if 'news' in market_data:
                    news = market_data['news']
                    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
                    for item in news:
                        sentiment_counts[item['sentiment']] += 1
                    
                    total_news = len(news)
                    sentiment_features.update({
                        'positive_news_ratio': sentiment_counts['positive'] / total_news if total_news > 0 else 0,
                        'negative_news_ratio': sentiment_counts['negative'] / total_news if total_news > 0 else 0,
                        'neutral_news_ratio': sentiment_counts['neutral'] / total_news if total_news > 0 else 0
                    })
                
                # دمج كل الميزات
                all_features = {
                    **binance_features,
                    **blockchain_features,
                    **market_dominance_features,
                    **sentiment_features
                }
                
                return pd.DataFrame([all_features])
            
            return None
        except Exception as e:
            logging.error(f"خطأ في تجهيز بيانات السوق: {str(e)}")
            return None
            
    def prepare_data(self, data, is_training=False):
        """تجهيز البيانات للتدريب أو التنبؤ"""
        try:
            df = data.copy()
            
            # تجهيز البيانات
            market_data = self.prepare_market_data()
            if market_data is not None:
                for col in market_data.columns:
                    df[col] = market_data[col].iloc[0]
            
            # معالجة القيم المفقودة
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # حساب المؤشرات الفنية
            df = self.calculate_technical_indicators(df)
            if df is None:
                raise ValueError("فشل في حساب المؤشرات الفنية")
            
            # فصل البيانات حسب النوع
            price_cols = ['close', 'high', 'low', 'open']
            volume_cols = ['volume']
            technical_cols = self.technical_indicators
            market_cols = [col for col in df.columns if col not in price_cols + volume_cols + technical_cols]
            
            # تطبيع البيانات
            price_data = df[price_cols].values
            volume_data = df[volume_cols].values
            technical_data = df[technical_cols].values
            market_data = df[market_cols].values
            
            # تطبيع كل عمود على حدة
            normalized_price = []
            for i in range(price_data.shape[1]):
                norm_col, _ = self.normalize_data(price_data[:, i], self.price_scalers[i])
                if norm_col is None:
                    return None, None, None, None, None
                normalized_price.append(norm_col)
            price_data = np.column_stack(normalized_price)
            
            normalized_volume = []
            for i in range(volume_data.shape[1]):
                norm_col, _ = self.normalize_data(volume_data[:, i], self.volume_scalers[i])
                if norm_col is None:
                    return None, None, None, None, None
                normalized_volume.append(norm_col)
            volume_data = np.column_stack(normalized_volume)
            
            normalized_technical = []
            for i in range(technical_data.shape[1]):
                norm_col, _ = self.normalize_data(technical_data[:, i], self.technical_scalers[i])
                if norm_col is None:
                    return None, None, None, None, None
                normalized_technical.append(norm_col)
            technical_data = np.column_stack(normalized_technical)
            
            if self.market_scalers is None:
                self.market_scalers = [StandardScaler() for _ in range(market_data.shape[1])]
            
            normalized_market = []
            for i in range(market_data.shape[1]):
                norm_col, _ = self.normalize_data(market_data[:, i], self.market_scalers[i])
                if norm_col is None:
                    return None, None, None, None, None
                normalized_market.append(norm_col)
            market_data = np.column_stack(normalized_market)
            
            # التحقق من وجود قيم NaN
            if any(np.isnan(x).any() for x in [price_data, volume_data, technical_data, market_data]):
                logging.error("تم اكتشاف قيم NaN بعد التطبيع")
                return None, None, None, None, None
            
            # طباعة أشكال البيانات
            logging.info("\nأشكال البيانات بعد التطبيع:")
            logging.info(f"price_data: {price_data.shape}")
            logging.info(f"volume_data: {volume_data.shape}")
            logging.info(f"technical_data: {technical_data.shape}")
            logging.info(f"market_data: {market_data.shape}")
            
            # إعداد البيانات للتدريب
            if is_training:
                y = df['close'].values[1:]  # القيم المستهدفة (أسعار الإغلاق)
                
                # إزالة السطر الأخير من X لمطابقة حجم y
                price_data = price_data[:-1]
                volume_data = volume_data[:-1]
                technical_data = technical_data[:-1]
                market_data = market_data[:-1]
                
                return price_data, volume_data, technical_data, market_data, y
            
            return price_data, volume_data, technical_data, market_data, None
            
        except Exception as e:
            logging.error(f"خطأ في تجهيز البيانات: {str(e)}")
            return None, None, None, None, None

    def fit_scalers(self, data):
        """تهيئة المقاييس على البيانات الكاملة"""
        try:
            df = data.copy()
            
            # تجهيز البيانات
            market_data = self.prepare_market_data()
            if market_data is not None:
                for col in market_data.columns:
                    df[col] = market_data[col].iloc[0]
            
            # معالجة القيم المفقودة
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # حساب المؤشرات الفنية
            df = self.calculate_technical_indicators(df)
            if df is None:
                raise ValueError("فشل في حساب المؤشرات الفنية")
            
            # فصل البيانات حسب النوع
            price_cols = ['close', 'high', 'low', 'open']
            volume_cols = ['volume']
            technical_cols = self.technical_indicators
            market_cols = [col for col in df.columns if col not in price_cols + volume_cols + technical_cols]
            
            # تهيئة المقاييس
            for i in range(len(price_cols)):
                self.price_scalers[i].fit(df[price_cols[i]].values)
            
            for i in range(len(volume_cols)):
                self.volume_scalers[i].fit(df[volume_cols[i]].values)
            
            for i in range(len(technical_cols)):
                self.technical_scalers[i].fit(df[technical_cols[i]].values)
            
            if self.market_scalers is None:
                self.market_scalers = [StandardScaler() for _ in range(len(market_cols))]
            
            for i in range(len(market_cols)):
                self.market_scalers[i].fit(df[market_cols[i]].values)
            
            logging.info("تم تهيئة المقاييس بنجاح")
            
        except Exception as e:
            logging.error(f"خطأ في تهيئة المقاييس: {str(e)}")

    def build_model(self):
        """بناء نموذج التعلم العميق"""
        try:
            # تحديد أبعاد المدخلات
            price_input = Input(shape=(self.sequence_length, 4), name='price_input')
            volume_input = Input(shape=(self.sequence_length, 1), name='volume_input')
            technical_input = Input(shape=(self.sequence_length, 29), name='technical_input')
            market_input = Input(shape=(self.sequence_length, len(self.market_scalers)), name='market_input')
            
            # معالجة بيانات السعر
            price_lstm = LSTM(64, return_sequences=True)(price_input)
            price_lstm = LSTM(32)(price_lstm)
            price_dense = Dense(16, activation='relu')(price_lstm)
            
            # معالجة بيانات الحجم
            volume_lstm = LSTM(32, return_sequences=True)(volume_input)
            volume_lstm = LSTM(16)(volume_lstm)
            volume_dense = Dense(8, activation='relu')(volume_lstm)
            
            # معالجة المؤشرات الفنية
            technical_lstm = LSTM(64, return_sequences=True)(technical_input)
            technical_lstm = LSTM(32)(technical_lstm)
            technical_dense = Dense(16, activation='relu')(technical_lstm)
            
            # معالجة بيانات السوق
            market_lstm = LSTM(64, return_sequences=True)(market_input)
            market_lstm = LSTM(32)(market_lstm)
            market_dense = Dense(16, activation='relu')(market_lstm)
            
            # دمج كل المخرجات
            merged = tf.keras.layers.Concatenate()([price_dense, volume_dense, technical_dense, market_dense])
            
            # طبقات كثيفة للتنبؤ
            dense = Dense(32, activation='relu')(merged)
            dense = Dropout(0.2)(dense)
            dense = Dense(16, activation='relu')(dense)
            dense = Dropout(0.1)(dense)
            output = Dense(1, activation='linear')(dense)
            
            # تجميع النموذج
            model = Model(
                inputs=[price_input, volume_input, technical_input, market_input],
                outputs=output
            )
            
            # تجميع النموذج
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # حفظ النموذج في المتغير العضوي
            self.model = model
            
            return model
            
        except Exception as e:
            logging.error(f"خطأ في بناء النموذج: {str(e)}")
            return None

    def custom_loss(self, y_true, y_pred):
        """دالة خسارة مخصصة تجمع بين MSE وMAE مع تركيز على الاتجاه"""
        # حساب MSE و MAE
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        
        # حساب خطأ الاتجاه
        diff_true = y_true[1:] - y_true[:-1]
        diff_pred = y_pred[1:] - y_pred[:-1]
        direction_error = tf.keras.losses.mean_squared_error(
            tf.sign(diff_true),
            tf.sign(diff_pred)
        )
        
        # الجمع بين المقاييس مع أوزان مختلفة
        combined_loss = 0.4 * mse + 0.4 * mae + 0.2 * direction_error
        return combined_loss

    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """تدريب النموذج"""
        try:
            logging.info("\nتجهيز البيانات للتدريب...")
            self.fit_scalers(train_data)
            X_train_price, X_train_volume, X_train_technical, X_train_market, y_train = self.prepare_data(train_data, is_training=True)
            if any(x is None for x in [X_train_price, X_train_volume, X_train_technical, X_train_market, y_train]):
                raise ValueError("فشل في تجهيز بيانات التدريب")
                
            logging.info("\nتجهيز بيانات التحقق...")
            X_val_price, X_val_volume, X_val_technical, X_val_market, y_val = self.prepare_data(val_data, is_training=False)
            if any(x is None for x in [X_val_price, X_val_volume, X_val_technical, X_val_market, y_val]):
                raise ValueError("فشل في تجهيز بيانات التحقق")
            
            # طباعة أشكال البيانات للتأكد
            logging.info("\nأشكال بيانات التدريب:")
            logging.info(f"X_train[0] (price): {X_train_price.shape}")
            logging.info(f"X_train[1] (volume): {X_train_volume.shape}")
            logging.info(f"X_train[2] (technical): {X_train_technical.shape}")
            logging.info(f"X_train[3] (market): {X_train_market.shape}")
            logging.info(f"y_train: {y_train.shape}")
            
            logging.info("\nأشكال بيانات التحقق:")
            logging.info(f"X_val[0] (price): {X_val_price.shape}")
            logging.info(f"X_val[1] (volume): {X_val_volume.shape}")
            logging.info(f"X_val[2] (technical): {X_val_technical.shape}")
            logging.info(f"X_val[3] (market): {X_val_market.shape}")
            logging.info(f"y_val: {y_val.shape}")
            
            if self.model is None:
                self.build_model()
                
            # تكوين callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                CSVLogger('training_log.csv')
            ]
            
            # تدريب النموذج
            history = self.model.fit(
                x=[X_train_price, X_train_volume, X_train_technical, X_train_market],
                y=y_train,
                validation_data=([X_val_price, X_val_volume, X_val_technical, X_val_market], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.history = history
            return history
            
        except Exception as e:
            logging.error(f"خطأ في تدريب النموذج: {str(e)}")
            return None

    def evaluate_model(self, data):
        """تقييم أداء النموذج"""
        try:
            # تحضير البيانات للتقييم
            df = data.copy()
            df = self.calculate_technical_indicators(df)
            if df is None:
                raise ValueError("فشل في حساب المؤشرات الفنية")
            
            # تجهيز البيانات للتقييم
            X_price, X_volume, X_technical, X_market, y_true = self.prepare_data(df)
            if X_price is None:
                raise ValueError("فشل في تجهيز البيانات للتقييم")
            
            # حفظ القيم الأصلية للمقارنة
            y_true_original = df['close'].values[self.sequence_length:]
            if len(y_true_original) > len(y_true):
                y_true_original = y_true_original[:len(y_true)]
            
            # التنبؤ باستخدام النموذج
            y_pred = self.predict(X_price, X_volume, X_technical, X_market)
            if y_pred is None:
                raise ValueError("فشل في التنبؤ")
            
            # التأكد من تطابق الأبعاد
            min_len = min(len(y_true_original), len(y_pred))
            y_true_original = y_true_original[:min_len]
            y_pred = y_pred[:min_len]
            
            # حساب المقاييس
            mae = mean_absolute_error(y_true_original, y_pred)
            mse = mean_squared_error(y_true_original, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_original, y_pred)
            
            # حساب MAPE مع تجنب القسمة على صفر
            epsilon = 1e-7  # قيمة صغيرة لتجنب القسمة على صفر
            mape = np.mean(np.abs((y_true_original - y_pred) / (y_true_original + epsilon))) * 100
            
            # حساب دقة اتجاه الحركة
            direction_true = np.sign(np.diff(y_true_original))
            direction_pred = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(direction_true == direction_pred) * 100
            
            # طباعة النتائج
            logging.info("\nنتائج تقييم النموذج:")
            logging.info(f"متوسط الخطأ المطلق (MAE): ${mae:,.2f}")
            logging.info(f"الجذر التربيعي لمتوسط مربع الخطأ (RMSE): ${rmse:,.2f}")
            logging.info(f"متوسط النسبة المئوية للخطأ المطلق (MAPE): {mape:.2f}%")
            logging.info(f"معامل التحديد (R²): {r2:.4f}")
            logging.info(f"دقة اتجاه الحركة: {direction_accuracy:.2f}%")
            
            # رسم النتائج
            plt.figure(figsize=(15, 10))
            
            # رسم المقارنة بين القيم الحقيقية والتنبؤات
            plt.subplot(2, 1, 1)
            plt.plot(y_true_original, label='القيم الفعلية', color='blue', alpha=0.7)
            plt.plot(y_pred, label='التنبؤات', color='red', alpha=0.7)
            plt.title('مقارنة بين القيم الفعلية والتنبؤات')
            plt.xlabel('الفترة الزمنية')
            plt.ylabel('السعر (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # رسم الخطأ
            plt.subplot(2, 1, 2)
            error = y_pred - y_true_original
            plt.plot(error, color='green', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('خطأ التنبؤ')
            plt.xlabel('الفترة الزمنية')
            plt.ylabel('الخطأ (USD)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('evaluation_results.png')
            plt.close()
            
            return mae, rmse, mape, r2, direction_accuracy
            
        except Exception as e:
            logging.error(f"خطأ في تقييم النموذج: {str(e)}")
            return None, None, None, None, None

    def predict(self, X_price, X_volume, X_technical, X_market):
        """التنبؤ باستخدام النموذج"""
        try:
            # التأكد من أن البيانات لها نفس الطول
            min_length = min(len(X_price), len(X_volume), len(X_technical), len(X_market))
            X_price = X_price[:min_length]
            X_volume = X_volume[:min_length]
            X_technical = X_technical[:min_length]
            X_market = X_market[:min_length]
            
            # التنبؤ باستخدام النموذج
            predictions = self.model.predict(
                [X_price, X_volume, X_technical, X_market],
                verbose=0
            )
            
            # إعادة تشكيل التنبؤات إذا لزم الأمر
            if len(predictions.shape) > 1:
                predictions = predictions.reshape(-1)
                
            # إلغاء التطبيع للتنبؤات
            predictions = self.price_scalers[0].inverse_transform(
                np.column_stack([predictions, np.zeros_like(predictions), np.zeros_like(predictions), np.zeros_like(predictions)])
            )[:, 0]
            
            return predictions
            
        except Exception as e:
            logging.error(f"خطأ في التنبؤ: {str(e)}")
            return None

    def predict_next_day(self, data):
        """التنبؤ بسعر اليوم التالي"""
        try:
            # تحضير نسخة من البيانات
            df = data.copy()
            
            # تأكد من وجود بيانات كافية
            if len(df) < self.sequence_length:
                raise ValueError(f"عدد الصفوف {len(df)} أقل من الحد الأدنى المطلوب {self.sequence_length}")
            
            # حساب المؤشرات الفنية
            df = self.calculate_technical_indicators(df)
            if df is None:
                raise ValueError("فشل في حساب المؤشرات الفنية")
            
            # أخذ آخر sequence_length صف
            df = df.tail(self.sequence_length)
            
            # تطبيع البيانات
            price_data = np.array([self.price_scalers[i].transform(df[['close', 'high', 'low', 'open']].values[:, i].reshape(-1, 1)) for i in range(4)]).T
            volume_data = np.array([self.volume_scalers[i].transform(df[['volume']].values[:, i].reshape(-1, 1)) for i in range(1)]).T
            technical_data = np.array([self.technical_scalers[i].transform(df[self.technical_indicators].values[:, i].reshape(-1, 1)) for i in range(len(self.technical_indicators))]).T
            market_data = np.array([self.market_scalers[i].transform(df[[col for col in df.columns if col not in ['close', 'high', 'low', 'open', 'volume'] + self.technical_indicators]].values[:, i].reshape(-1, 1)) for i in range(len(self.market_scalers))]).T
            
            # تشكيل البيانات للتنبؤ
            X_price = np.array([price_data])
            X_volume = np.array([volume_data])
            X_technical = np.array([technical_data])
            X_market = np.array([market_data])
            
            # التنبؤ
            prediction = self.model.predict(
                [X_price, X_volume, X_technical, X_market], 
                verbose=0,
                batch_size=1
            )
            
            if prediction is None or len(prediction) == 0:
                raise ValueError("فشل في الحصول على تنبؤ")
                
            # إعادة تحويل التنبؤ إلى القيمة الأصلية
            prediction_reshaped = prediction[0].reshape(-1, 1)
            prediction_original = self.price_scalers[0].inverse_transform(
                np.hstack([prediction_reshaped, np.zeros((1, 3))])
            )[0, 0]
            
            # التأكد من أن القيمة المتنبأ بها موجبة
            prediction_original = max(0, float(prediction_original))
            
            return prediction_original
            
        except Exception as e:
            logging.error(f"خطأ في التنبؤ باليوم التالي: {str(e)}")
            return None

    def predict_sequence(self, initial_sequence, n_steps):
        """التنبؤ بسلسلة من الأسعار المستقبلية"""
        try:
            predictions = []
            current_sequence = initial_sequence.copy()
        
            for _ in range(n_steps):
                # التنبؤ بالسعر التالي
                next_price = self.predict_next_day(current_sequence)
                if next_price is None:
                    break
                
                predictions.append(next_price)
            
                # تحديث التسلسل للتنبؤ التالي
                new_row = pd.DataFrame({
                    'close': [next_price],
                    'high': [next_price * 1.01],  # تقدير تقريبي
                    'low': [next_price * 0.99],   # تقدير تقريبي
                    'open': [next_price],
                    'volume': [current_sequence['volume'].mean()]
                }, index=[current_sequence.index[-1] + pd.Timedelta(days=1)])
            
                current_sequence = pd.concat([current_sequence[1:], new_row])
        
            return predictions
        
        except Exception as e:
            logging.error(f"خطأ في التنبؤ بالتسلسل: {str(e)}")
            return None

    def save_model(self, filepath):
        """حفظ النموذج"""
        try:
            if self.model is not None:
                self.model.save(filepath)
                logging.info(f"تم حفظ النموذج في {filepath}")
            else:
                logging.error("لا يوجد نموذج لحفظه")
        
        except Exception as e:
            logging.error(f"خطأ في حفظ النموذج: {str(e)}")
        
    def load_model(self, filepath):
        """تحميل نموذج محفوظ"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            logging.info(f"تم تحميل النموذج من {filepath}")
        
        except Exception as e:
            logging.error(f"خطأ في تحميل النموذج: {str(e)}")
        
    def plot_training_history(self):
        """رسم منحنيات التدريب"""
        try:
            if not self.training_history:
                logging.warning("لا يوجد تاريخ تدريب للرسم")
                return
        
            # إنشاء شكل بحجم مناسب
            plt.figure(figsize=(15, 10))
        
            # رسم منحنى الخسارة
            plt.subplot(2, 1, 1)
            plt.plot(self.training_history['loss'], label='Training Loss')
            if 'val_loss' in self.training_history:
                plt.plot(self.training_history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            # رسم منحنى MAE
            plt.subplot(2, 1, 2)
            plt.plot(self.training_history['mae'], label='Training MAE')
            if 'val_mae' in self.training_history:
                plt.plot(self.training_history['val_mae'], label='Validation MAE')
            plt.title('Model MAE During Training')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            # تحسين التخطيط
            plt.tight_layout()
        
            # حفظ الرسم
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            logging.error(f"خطأ في رسم تاريخ التدريب: {str(e)}")
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            # تحسين التخطيط
            plt.tight_layout()
        
            # حفظ الرسم
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            logging.error(f"خطأ في رسم تاريخ التدريب: {str(e)}")

    def normalize_data(self, data, scaler=None):
        """تطبيع البيانات"""
        try:
            if data is None or len(data) == 0:
                logging.error("البيانات فارغة أو غير موجودة")
                return None, None
                
            # التأكد من أن البيانات رقمية
            data = np.array(data, dtype=np.float64)
            
            # إعادة تشكيل البيانات إذا كانت أحادية البعد
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # معالجة القيم المفقودة
            if np.isnan(data).any():
                logging.warning("تم اكتشاف قيم NaN قبل التطبيع")
                data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
            
            # تطبيع البيانات
            if scaler is None:
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(data)
            else:
                normalized_data = scaler.transform(data)
            
            # التحقق من وجود قيم NaN بعد التطبيع
            if np.isnan(normalized_data).any():
                logging.error("تم اكتشاف قيم NaN بعد التطبيع")
                return None, None
            
            return normalized_data, scaler
            
        except Exception as e:
            logging.error(f"خطأ في تطبيع البيانات: {str(e)}")
            return None, None

    def calculate_technical_indicators(self, df):
        """حساب المؤشرات الفنية"""
        try:
            # تأكد من أن أسماء الأعمدة بأحرف صغيرة
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            
            # طباعة أسماء الأعمدة للتأكد
            logging.info(f"أسماء الأعمدة: {df.columns.tolist()}")
            
            # تحويل البيانات إلى numpy arrays
            close = np.array(df['close'].astype(float))
            high = np.array(df['high'].astype(float))
            low = np.array(df['low'].astype(float))
            
            # التعامل مع عمود الحجم
            if isinstance(df['volume'], pd.DataFrame):
                volume = np.array(df['volume'].iloc[:, 0].astype(float))  # أخذ العمود الأول فقط
            else:
                volume = np.array(df['volume'].astype(float))
            
            # طباعة شكل البيانات للتأكد
            logging.info(f"شكل البيانات:")
            logging.info(f"close: {close.shape}")
            logging.info(f"high: {high.shape}")
            logging.info(f"low: {low.shape}")
            logging.info(f"volume: {volume.shape}")
            
            # إنشاء نسخة من البيانات
            df_tech = df.copy()
            
            # حساب المؤشرات الفنية مع معالجة القيم المفقودة
            def safe_indicator(func, *args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, tuple):
                    return tuple(pd.Series(r).fillna(method='ffill').fillna(method='bfill').values for r in result)
                return pd.Series(result).fillna(method='ffill').fillna(method='bfill').values
            
            # المتوسطات المتحركة البسيطة
            df_tech['sma_5'] = safe_indicator(ta.SMA, close, timeperiod=5)
            df_tech['sma_10'] = safe_indicator(ta.SMA, close, timeperiod=10)
            df_tech['sma_20'] = safe_indicator(ta.SMA, close, timeperiod=20)
            df_tech['sma_50'] = safe_indicator(ta.SMA, close, timeperiod=50)
            
            # المتوسطات المتحركة الأسية
            df_tech['ema_5'] = safe_indicator(ta.EMA, close, timeperiod=5)
            df_tech['ema_10'] = safe_indicator(ta.EMA, close, timeperiod=10)
            df_tech['ema_20'] = safe_indicator(ta.EMA, close, timeperiod=20)
            df_tech['ema_50'] = safe_indicator(ta.EMA, close, timeperiod=50)
            
            # مؤشر القوة النسبية
            df_tech['rsi'] = safe_indicator(ta.RSI, close, timeperiod=14)
            
            # مؤشر MACD
            macd, macd_signal, macd_hist = safe_indicator(ta.MACD, close)
            df_tech['macd'] = macd
            df_tech['macd_signal'] = macd_signal
            df_tech['macd_hist'] = macd_hist
            
            # مؤشرات الزخم
            df_tech['mom'] = safe_indicator(ta.MOM, close, timeperiod=10)
            df_tech['roc'] = safe_indicator(ta.ROC, close, timeperiod=10)
            
            # مؤشر Bollinger Bands
            upper, middle, lower = safe_indicator(ta.BBANDS, close)
            df_tech['bbands_upper'] = upper
            df_tech['bbands_middle'] = middle
            df_tech['bbands_lower'] = lower
            
            # مؤشر ATR
            df_tech['atr'] = safe_indicator(ta.ATR, high, low, close)
            
            # مؤشرات الحجم
            df_tech['obv'] = safe_indicator(ta.OBV, close, volume)
            df_tech['ad'] = safe_indicator(ta.AD, high, low, close, volume)
            df_tech['adosc'] = safe_indicator(ta.ADOSC, high, low, close, volume)
            
            # مؤشرات التذبذب
            df_tech['adx'] = safe_indicator(ta.ADX, high, low, close)
            df_tech['cci'] = safe_indicator(ta.CCI, high, low, close)
            df_tech['dx'] = safe_indicator(ta.DX, high, low, close)
            df_tech['willr'] = safe_indicator(ta.WILLR, high, low, close)
            df_tech['ultosc'] = safe_indicator(ta.ULTOSC, high, low, close)
            df_tech['trix'] = safe_indicator(ta.TRIX, close)
            
            # مؤشر Stochastic
            stoch, stoch_signal = safe_indicator(ta.STOCH, high, low, close)
            df_tech['stoch'] = stoch
            df_tech['stoch_signal'] = stoch_signal
            
            # نسخ الأعمدة الأصلية
            for col in df.columns:
                if col not in df_tech.columns:
                    df_tech[col] = df[col]
            
            # التحقق من وجود قيم NaN
            if df_tech.isnull().any().any():
                logging.warning("تم اكتشاف قيم NaN في المؤشرات الفنية")
                df_tech = df_tech.fillna(method='ffill').fillna(method='bfill')
            
            return df_tech
            
        except Exception as e:
            logging.error(f"خطأ في حساب المؤشرات الفنية: {str(e)}")
            return None
