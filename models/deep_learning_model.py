import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('..')
from config import MODEL_PARAMS

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(MODEL_PARAMS['dropout_rate'])
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class CryptoPredictor:
    def __init__(self):
        self.sequence_length = MODEL_PARAMS['sequence_length']
        self.n_features = MODEL_PARAMS['n_features']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # إنشاء نموذج LSTM
        self.model = LSTM(
            input_size=self.n_features,
            hidden_size=MODEL_PARAMS['lstm_units']
        ).to(self.device)
        
        # إنشاء مقياس التطبيع للأسعار فقط
        self.price_scaler = MinMaxScaler()
        # إنشاء مقياس التطبيع للميزات
        self.feature_scaler = MinMaxScaler()
    
    def prepare_data(self, df):
        """تحضير البيانات للتدريب أو التنبؤ"""
        try:
            # تحديد الميزات المطلوبة
            features = ['open', 'high', 'low', 'close', 'volume', 'MA20']
            
            if df is None or df.empty or any(col not in df.columns for col in features):
                print("البيانات غير مكتملة أو غير صحيحة")
                return None, None
            
            # تطبيع بيانات الأسعار
            prices = df['close'].values.reshape(-1, 1)
            self.price_scaler.fit(prices)
            
            # تطبيع جميع الميزات
            feature_data = df[features].values
            self.feature_scaler.fit(feature_data)
            scaled_features = self.feature_scaler.transform(feature_data)
            
            # إنشاء متسلسلات البيانات
            X = []
            y = []
            
            for i in range(len(scaled_features) - self.sequence_length):
                X.append(scaled_features[i:(i + self.sequence_length)])
                y.append(prices[i + self.sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"خطأ في تحضير البيانات: {str(e)}")
            return None, None
    
    def train(self, X, y):
        """تدريب النموذج"""
        try:
            if X is None or y is None:
                return
                
            # تحويل البيانات إلى تنسيق PyTorch
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)
            
            # تحديد دالة الخسارة ومحسن النموذج
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters())
            
            # تدريب النموذج
            self.model.train()
            for epoch in range(100):  # يمكن تعديل عدد الحلقات
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
                    
        except Exception as e:
            print(f"خطأ في تدريب النموذج: {str(e)}")
    
    def predict(self, X):
        """التنبؤ باستخدام النموذج"""
        try:
            if X is None:
                return None
                
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(X).to(self.device)
                predictions = self.model(X).cpu().numpy()
                
                # عكس عملية التطبيع للأسعار فقط
                return self.price_scaler.inverse_transform(predictions)
                
        except Exception as e:
            print(f"خطأ في التنبؤ: {str(e)}")
            return None
