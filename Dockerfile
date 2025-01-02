# استخدام صورة Python الرسمية
FROM python:3.11-slim

# تثبيت المتطلبات الأساسية
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# تحميل وتثبيت TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملفات المشروع
COPY . /app/

# تثبيت المتطلبات Python
RUN pip install --upgrade pip && \
    pip install numpy==1.26.2 && \
    pip install TA-Lib==0.4.28 && \
    pip install -r requirements.txt

# تعريف المنفذ
EXPOSE 8000

# أمر التشغيل
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:app.server"]
