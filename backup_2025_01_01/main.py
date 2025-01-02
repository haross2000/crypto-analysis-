"""
التطبيق الرئيسي للتحليل الفني
"""

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
from utils.data_collector import DataCollector
from utils.technical_analyzer import TechnicalAnalyzer
from utils.pattern_analyzer import PatternAnalyzer
from utils.market_analyzer import MarketAnalyzer
from utils.decision_maker import DecisionMaker
from utils.risk_management import RiskManager
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import ta.momentum
import ta.trend
import ta.volatility
from typing import Dict, Any, List, Tuple

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# تحميل متغيرات البيئة
load_dotenv()

# الحصول على مفاتيح API
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise ValueError("الرجاء إضافة مفاتيح API في ملف .env")

# تهيئة المكونات
data_collector = DataCollector(BINANCE_API_KEY, BINANCE_API_SECRET)
technical_analyzer = TechnicalAnalyzer()
pattern_analyzer = PatternAnalyzer()
market_analyzer = MarketAnalyzer()
decision_maker = DecisionMaker()
risk_manager = RiskManager()

# تهيئة المتغيرات العالمية
available_pairs = []

# تحديث قائمة الأزواج والفترات الزمنية
def get_trading_pairs():
    """جلب قائمة أزواج التداول"""
    try:
        logging.info("محاولة جلب أزواج التداول...")
        pairs = data_collector.get_all_usdt_pairs()
        logging.info(f"تم جلب {len(pairs)} زوج بنجاح")
        return pairs
    except Exception as e:
        logging.error(f"خطأ في جلب أزواج التداول: {str(e)}")
        # إعادة القائمة الافتراضية في حالة الخطأ
        default_pairs = [
            {'label': 'Bitcoin (BTC/USDT)', 'value': 'BTCUSDT'},
            {'label': 'Ethereum (ETH/USDT)', 'value': 'ETHUSDT'},
            {'label': 'Binance Coin (BNB/USDT)', 'value': 'BNBUSDT'},
            {'label': 'Cardano (ADA/USDT)', 'value': 'ADAUSDT'},
            {'label': 'Solana (SOL/USDT)', 'value': 'SOLUSDT'},
            {'label': 'Ripple (XRP/USDT)', 'value': 'XRPUSDT'}
        ]
        logging.warning("استخدام القائمة الافتراضية للأزواج")
        return default_pairs

TRADING_PAIRS = get_trading_pairs()

TIME_FRAMES = [
    {'label': '1 دقيقة', 'value': '1m'},
    {'label': '5 دقائق', 'value': '5m'},
    {'label': '15 دقيقة', 'value': '15m'},
    {'label': '1 ساعة', 'value': '1h'},
    {'label': '4 ساعات', 'value': '4h'},
    {'label': '1 يوم', 'value': '1d'}
]

def initialize_components():
    """تهيئة المكونات"""
    try:
        # تحقق من وجود مفاتيح API
        if os.path.exists('.env'):
            load_dotenv()
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            logging.info("تم تحميل مفاتيح API")
            logging.info(f"API Key: {api_key[:5]}...{api_key[-5:]}")
            
            if not api_key or not api_secret:
                raise ValueError("يجب تحديد BINANCE_API_KEY و BINANCE_API_SECRET في ملف .env")
                
            analyzer = TechnicalAnalyzer()  
            pattern_analyzer = PatternAnalyzer()
            market_analyzer = MarketAnalyzer()
            decision_maker = DecisionMaker()
            risk_manager = RiskManager()
            
            # تحديث قائمة الأزواج
            global available_pairs
            available_pairs = data_collector.get_trading_pairs()
            if not available_pairs:
                raise ValueError("لم يتم العثور على أزواج تداول")
            
            logging.info("تم تهيئة جميع المكونات بنجاح")
            return analyzer, pattern_analyzer, market_analyzer, decision_maker, risk_manager
            
        else:
            raise FileNotFoundError("ملف .env غير موجود")
            
    except Exception as e:
        logging.error(f"خطأ في التهيئة: {str(e)}", exc_info=True)
        raise

def get_available_pairs():
    """الحصول على قائمة الأزواج المتاحة"""
    global available_pairs
    try:
        pairs = data_collector.get_trading_pairs()
        if pairs:
            available_pairs = pairs
            logging.info(f"تم تحديث قائمة الأزواج المتاحة: {len(available_pairs)} زوج")
            logging.info(f"الأزواج المتاحة: {available_pairs[:5]}...")
            return available_pairs
        else:
            raise ValueError("لم يتم العثور على أزواج تداول")
    except Exception as e:
        logging.error(f"خطأ في جلب قائمة الأزواج: {str(e)}")
        # استخدام قائمة افتراضية في حالة الخطأ
        available_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
            'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LINKUSDT', 'SOLUSDT'
        ]
        return available_pairs

try:
    # تهيئة المكونات
    analyzer, pattern_analyzer, market_analyzer, decision_maker, risk_manager = initialize_components()
    get_available_pairs()
    logging.info("تم تهيئة جميع المكونات بنجاح")
except Exception as e:
    logger.error(f"خطأ في تهيئة المكونات: {str(e)}")
    raise

# تهيئة تطبيق Dash
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="نظام تداول العملات الرقمية"
)

# تعريف تخطيط التطبيق
app.layout = html.Div([
    # شريط التنقل
    html.Nav([
        html.Div([
            html.H3("تحليل العملات الرقمية", className="text-light mb-0"),
            html.Span("Crypto Analysis", className="text-muted")
        ], className="container-fluid")
    ], className="navbar navbar-dark bg-dark mb-4 shadow-sm"),

    # المحتوى الرئيسي
    html.Div([
        # صف أعلى للتحكم
        html.Div([
            html.Div([
                html.Label("الزوج:", className="form-label"),
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=get_trading_pairs(),
                    value='BTCUSDT',
                    clearable=False,
                    className="form-select mb-3"
                )
            ], className="col-md-4"),
            
            html.Div([
                html.Label("الفترة الزمنية:", className="form-label"),
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=TIME_FRAMES,
                    value='1h',
                    clearable=False,
                    className="form-select mb-3"
                )
            ], className="col-md-4"),
            
            html.Div([
                html.Label("\u00A0", className="form-label d-block"),
                html.Button(
                    [
                        html.I(className="fas fa-sync-alt me-2"),
                        "تحديث"
                    ],
                    id='update-button',
                    n_clicks=0,
                    className="btn btn-primary w-100"
                )
            ], className="col-md-4")
        ], className="row g-3 mb-4"),

        # مؤشر التحميل
        dcc.Loading(
            id="loading-1",
            type="circle",
            children=[
                html.Div(id="loading-output-1"),
                # الرسم البياني
                html.Div([
                    dcc.Graph(
                        id='candlestick-chart',
                        config={'displayModeBar': True, 'scrollZoom': True},
                        className="shadow-sm"
                    )
                ], className="card mb-4"),

                # صف المؤشرات والإشارات
                html.Div([
                    # المؤشرات الفنية
                    html.Div([
                        html.Div(id='technical-indicators', className="h-100")
                    ], className="col-md-4"),
                    
                    # إشارات التداول
                    html.Div([
                        html.Div(id='trading-signals', className="h-100")
                    ], className="col-md-4"),
                    
                    # القرارات
                    html.Div([
                        html.Div(id='trading-decisions', className="h-100")
                    ], className="col-md-4")
                ], className="row g-4")
            ]
        ),

        # تذييل الصفحة
        html.Footer([
            html.Hr(className="my-4"),
            html.P([
                "تم التطوير بواسطة ",
                html.A("Codeium", href="https://codeium.com", target="_blank", className="text-decoration-none"),
                " © 2024"
            ], className="text-center text-muted")
        ]),

        # تحديث تلقائي
        dcc.Interval(
            id='auto-refresh',
            interval=60*1000,  # تحديث كل دقيقة
            n_intervals=0
        ),
        
        # تحديث قائمة الأزواج
        dcc.Interval(
            id='pairs-refresh',
            interval=5*60*1000,  # تحديث كل 5 دقائق
            n_intervals=0
        )
    ], className="container-fluid px-4")
], className="min-vh-100 bg-light")

def create_help_modal():
    """إنشاء نافذة المساعدة"""
    return html.Div([
        html.Button(
            "❓ مساعدة",
            id="help-button",
            className="btn btn-outline-info btn-sm position-fixed",
            style={"top": "10px", "right": "10px"}
        ),
        dbc.Modal([
            dbc.ModalHeader("كيفية قراءة المؤشرات الفنية"),
            dbc.ModalBody([
                html.H5("المتوسطات المتحركة (EMA)", className="mt-3"),
                html.Ul([
                    html.Li("تظهر كنسبة مئوية للتغير عن السعر السابق"),
                    html.Li("القيم الموجبة (خضراء) تشير إلى اتجاه صاعد"),
                    html.Li("القيم السالبة (حمراء) تشير إلى اتجاه هابط"),
                ]),
                
                html.H5("مؤشر القوة النسبية (RSI)", className="mt-3"),
                html.Ul([
                    html.Li([
                        html.Strong("القيمة: "),
                        "من 0 إلى 100"
                    ]),
                    html.Li([
                        html.Strong("المناطق: "),
                        "فوق 70 (ذروة شراء) - تحت 30 (ذروة بيع)"
                    ]),
                    html.Li([
                        html.Strong("الاتجاه: "),
                        "↗️ صاعد - ↘️ هابط - ↔️ متعادل"
                    ]),
                    html.Li([
                        html.Strong("المدة: "),
                        "عدد الفترات في المنطقة الحالية"
                    ])
                ]),
                
                html.H5("مؤشر MACD", className="mt-3"),
                html.Ul([
                    html.Li([
                        html.Strong("القيم: "),
                        "نسبة مئوية من السعر"
                    ]),
                    html.Li([
                        html.Strong("التقاطعات: "),
                        "MACD فوق الإشارة (صعود) - تحت الإشارة (هبوط)"
                    ]),
                    html.Li([
                        html.Strong("الهستوجرام: "),
                        "الأعمدة الخضراء (إيجابي) - الحمراء (سلبي)"
                    ])
                ])
            ])
        ], id="help-modal", size="lg")
    ])

# إضافة النافذة للتطبيق
app.layout = html.Div([
    create_help_modal(),
    # باقي مكونات التطبيق
    # شريط التنقل
    html.Nav([
        html.Div([
            html.H3("تحليل العملات الرقمية", className="text-light mb-0"),
            html.Span("Crypto Analysis", className="text-muted")
        ], className="container-fluid")
    ], className="navbar navbar-dark bg-dark mb-4 shadow-sm"),

    # المحتوى الرئيسي
    html.Div([
        # صف أعلى للتحكم
        html.Div([
            html.Div([
                html.Label("الزوج:", className="form-label"),
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=get_trading_pairs(),
                    value='BTCUSDT',
                    clearable=False,
                    className="form-select mb-3"
                )
            ], className="col-md-4"),
            
            html.Div([
                html.Label("الفترة الزمنية:", className="form-label"),
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=TIME_FRAMES,
                    value='1h',
                    clearable=False,
                    className="form-select mb-3"
                )
            ], className="col-md-4"),
            
            html.Div([
                html.Label("\u00A0", className="form-label d-block"),
                html.Button(
                    [
                        html.I(className="fas fa-sync-alt me-2"),
                        "تحديث"
                    ],
                    id='update-button',
                    n_clicks=0,
                    className="btn btn-primary w-100"
                )
            ], className="col-md-4")
        ], className="row g-3 mb-4"),

        # مؤشر التحميل
        dcc.Loading(
            id="loading-1",
            type="circle",
            children=[
                html.Div(id="loading-output-1"),
                # الرسم البياني
                html.Div([
                    dcc.Graph(
                        id='candlestick-chart',
                        config={'displayModeBar': True, 'scrollZoom': True},
                        className="shadow-sm"
                    )
                ], className="card mb-4"),

                # صف المؤشرات والإشارات
                html.Div([
                    # المؤشرات الفنية
                    html.Div([
                        html.Div(id='technical-indicators', className="h-100")
                    ], className="col-md-4"),
                    
                    # إشارات التداول
                    html.Div([
                        html.Div(id='trading-signals', className="h-100")
                    ], className="col-md-4"),
                    
                    # القرارات
                    html.Div([
                        html.Div(id='trading-decisions', className="h-100")
                    ], className="col-md-4")
                ], className="row g-4")
            ]
        ),

        # تذييل الصفحة
        html.Footer([
            html.Hr(className="my-4"),
            html.P([
                "تم التطوير بواسطة ",
                html.A("Codeium", href="https://codeium.com", target="_blank", className="text-decoration-none"),
                " © 2024"
            ], className="text-center text-muted")
        ]),

        # تحديث تلقائي
        dcc.Interval(
            id='auto-refresh',
            interval=60*1000,  # تحديث كل دقيقة
            n_intervals=0
        ),
        
        # تحديث قائمة الأزواج
        dcc.Interval(
            id='pairs-refresh',
            interval=5*60*1000,  # تحديث كل 5 دقائق
            n_intervals=0
        )
    ], className="container-fluid px-4")
], className="min-vh-100 bg-light")

@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-button", "n_clicks")],
    [State("help-modal", "is_open")],
)
def toggle_help_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('loading-output-1', 'children'),
    [Input('pairs-refresh', 'n_intervals')]
)
def update_pairs_status(n_intervals):
    """تحديث حالة تحديث الأزواج"""
    return ""

@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('technical-indicators', 'children'),
     Output('trading-signals', 'children'),
     Output('trading-decisions', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('interval-dropdown', 'value'),
     Input('update-button', 'n_clicks'),
     Input('auto-refresh', 'n_intervals')]
)
def update_chart(symbol, interval, n_clicks, n_intervals):
    """تحديث الرسم البياني والمؤشرات"""
    if not symbol or not interval:
        empty_chart = go.Figure()
        empty_chart.update_layout(
            title="اختر زوج وفترة زمنية",
            template='plotly_dark'
        )
        return empty_chart, "", "", ""

    try:
        # جلب البيانات
        df = data_collector.get_candlestick_data(symbol, interval)
        if df.empty:
            empty_chart = go.Figure()
            empty_chart.update_layout(
                title="لا توجد بيانات متاحة لهذا الزوج في هذه الفترة",
                template='plotly_dark'
            )
            return empty_chart, "", "", html.Div(
                "لا توجد بيانات متاحة لهذا الزوج في هذه الفترة",
                className="alert alert-warning"
            )

        # تحليل المؤشرات الفنية
        df_with_indicators = technical_analyzer.add_indicators(df)
        
        # تحليل الأنماط
        patterns = pattern_analyzer.analyze(df)
        
        # تحليل السوق
        market_analysis = market_analyzer.analyze(df)
        
        # اتخاذ القرارات
        decisions = decision_maker.make_decision(
            df_with_indicators.to_dict('records'),
            patterns,
            {'symbol': symbol, 'interval': interval, 'market_analysis': market_analysis}
        )

        # تحديث المخطط
        fig = create_candlestick_chart(df, df_with_indicators, patterns)
        
        # تحديث المؤشرات والإشارات والقرارات
        technical_indicators_div = update_technical_indicators(df_with_indicators)
        trading_signals_div = update_trading_signals(patterns)
        trading_decisions_div = update_trading_decisions(decisions)

        return fig, technical_indicators_div, trading_signals_div, trading_decisions_div

    except Exception as e:
        logging.error(f"خطأ في تحديث البيانات: {str(e)}")
        error_chart = go.Figure()
        error_chart.update_layout(
            title=f"خطأ: {str(e)}",
            template='plotly_dark'
        )
        error_div = html.Div(
            f"حدث خطأ: {str(e)}",
            className="alert alert-danger"
        )
        return error_chart, error_div, error_div, error_div

def update_technical_indicators(df_with_indicators: pd.DataFrame) -> html.Div:
    """تحديث عرض المؤشرات الفنية"""
    if df_with_indicators.empty:
        return html.Div("لا توجد مؤشرات متاحة", className="alert alert-warning")

    indicators = []
    last_row = df_with_indicators.iloc[-1]

    # المتوسطات المتحركة
    ma_div = html.Div([
        html.H6("المتوسطات المتحركة", className="mb-3"),
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("SMA 20:", className="text-muted"),
                    html.Td(f"{last_row.get('SMA_20', 0):.1f}%", 
                           className=f"text-end {'text-success' if last_row.get('SMA_20', 0) < 0 else 'text-danger'}")
                ]),
                html.Tr([
                    html.Td("SMA 50:", className="text-muted"),
                    html.Td(f"{last_row.get('SMA_50', 0):.1f}%", 
                           className=f"text-end {'text-success' if last_row.get('SMA_50', 0) < 0 else 'text-danger'}")
                ]),
                html.Tr([
                    html.Td("EMA 20:", className="text-muted"),
                    html.Td(f"{last_row.get('EMA_20', 0):.1f}%", 
                           className=f"text-end {'text-success' if last_row.get('EMA_20', 0) < 0 else 'text-danger'}")
                ]),
                html.Tr([
                    html.Td("EMA 50:", className="text-muted"),
                    html.Td(f"{last_row.get('EMA_50', 0):.1f}%", 
                           className=f"text-end {'text-success' if last_row.get('EMA_50', 0) < 0 else 'text-danger'}")
                ])
            ])
        ], className="table table-sm")
    ], className="mb-4")
    indicators.append(ma_div)

    # مؤشر RSI
    rsi = last_row.get('RSI', 0)
    rsi_trend = last_row.get('RSI_Trend', 'متعادل')
    rsi_zone_duration = last_row.get('RSI_Zone_Duration', 0)
    
    rsi_status = (
        'ذروة شراء' if rsi > 70 else
        'ذروة بيع' if rsi < 30 else
        'متعادل'
    )
    
    rsi_color = (
        'danger' if rsi > 70 else
        'success' if rsi < 30 else
        'warning'
    )
    
    trend_icon = (
        '↗️' if 'صاعد' in rsi_trend else
        '↘️' if 'هابط' in rsi_trend else
        '↔️'
    )
    
    zone_text = (
        f" (منذ {rsi_zone_duration} فترة)" if rsi_zone_duration > 0 else ""
    )
    
    rsi_div = html.Div([
        html.H6("مؤشر القوة النسبية (RSI)", className="mb-3"),
        html.Div([
            html.Div(
                className="d-flex justify-content-between align-items-center mb-2",
                children=[
                    html.Div([
                        html.Span(f"{rsi:.1f}", className=f"badge bg-{rsi_color} me-2"),
                        html.Span(trend_icon, className="me-2"),
                        html.Small(rsi_trend, className="text-muted")
                    ]),
                    html.Small(f"{rsi_status}{zone_text}", className="text-muted")
                ]
            ),
            html.Div([
                html.Div(
                    style={
                        "width": f"{min(rsi, 100)}%",
                        "backgroundColor": f"var(--bs-{rsi_color})",
                        "transition": "width 0.3s ease-in-out"
                    },
                    className="progress-bar",
                )
            ], className="progress")
        ])
    ], className="mb-4")
    indicators.append(rsi_div)

    # مؤشر MACD
    if all(col in df_with_indicators.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        try:
            macd_value = last_row['MACD']
            signal_value = last_row['MACD_Signal']
            hist_value = last_row['MACD_Hist']
            
            macd_div = html.Div([
                html.H6("مؤشر MACD", className="mb-3"),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("MACD:", className="text-muted"),
                            html.Td(f"{macd_value:.2f}%", 
                                   className=f"text-end {'text-success' if macd_value > signal_value else 'text-danger'}")
                        ]),
                        html.Tr([
                            html.Td("Signal:", className="text-muted"),
                            html.Td(f"{signal_value:.2f}%", className="text-end")
                        ]),
                        html.Tr([
                            html.Td("Histogram:", className="text-muted"),
                            html.Td(f"{hist_value:.2f}%", 
                                   className=f"text-end {'text-success' if hist_value > 0 else 'text-danger'}")
                        ])
                    ])
                ], className="table table-sm")
            ], className="mb-4")
        except Exception as e:
            logging.error(f"خطأ في عرض MACD: {str(e)}")
            macd_div = html.Div("خطأ في عرض MACD", className="alert alert-danger")
    
    indicators.append(macd_div)

    # حزم بولينجر
    bb_div = html.Div([
        html.H6("حزم بولينجر", className="mb-3"),
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("الحد العلوي:", className="text-muted"),
                    html.Td(f"{last_row.get('BB_Upper', 0):.2f}", className="text-end")
                ]),
                html.Tr([
                    html.Td("المتوسط:", className="text-muted"),
                    html.Td(f"{last_row.get('BB_Middle', 0):.2f}", className="text-end")
                ]),
                html.Tr([
                    html.Td("الحد السفلي:", className="text-muted"),
                    html.Td(f"{last_row.get('BB_Lower', 0):.2f}", className="text-end")
                ])
            ])
        ], className="table table-sm")
    ], className="mb-4")
    indicators.append(bb_div)

    return html.Div([
        html.H4("المؤشرات الفنية", className="card-header bg-dark text-light"),
        html.Div(indicators, className="card-body")
    ], className="card shadow-sm mb-4")

def update_trading_signals(patterns: List[Dict[str, Any]]) -> html.Div:
    """تحديث عرض إشارات التداول"""
    if not patterns:
        return html.Div("لا توجد إشارات متاحة", className="alert alert-warning")

    pattern_items = []
    for pattern in patterns:
        # تحديد لون الإشارة
        pattern_type = pattern.get('type', 'محايد')
        color = (
            'success' if pattern_type == 'bullish' else
            'danger' if pattern_type == 'bearish' else
            'warning'
        )
        
        # تحديد اتجاه النمط بالعربية للعرض
        direction = (
            'صاعد' if pattern_type == 'bullish' else
            'هابط' if pattern_type == 'bearish' else
            'محايد'
        )
        
        # تحديد قوة النمط
        strength = pattern.get('strength', 0) * 100
        strength_color = (
            'danger' if strength < 40 else
            'warning' if strength < 70 else
            'success'
        )

        pattern_items.append(html.Div([
            html.H6([
                pattern.get('name', 'نمط غير معروف'),
                html.Span(
                    direction,
                    className=f"badge bg-{color} float-end"
                )
            ], className="mb-2"),
            html.Div([
                "قوة النمط: ",
                html.Div([
                    html.Div(
                        style={
                            "width": f"{strength}%",
                            "backgroundColor": f"var(--bs-{strength_color})"
                        },
                        className="progress-bar",
                        children=f"{strength:.1f}%"
                    )
                ], className="progress")
            ], className="mb-2"),
            html.P(
                pattern.get('description', ''),
                className="small text-muted mb-0"
            )
        ], className="mb-4"))

    return html.Div([
        html.H4("إشارات التداول", className="card-header bg-dark text-light"),
        html.Div(pattern_items, className="card-body")
    ], className="card shadow-sm mb-4")

def update_trading_decisions(decisions):
    """تحديث قرارات التداول"""
    if not decisions:
        return html.Div("لا توجد قرارات متاحة", className="alert alert-warning")

    smart_decision = decisions.get('smart_decision', {})
    
    # تنسيق الأرقام
    def format_price(price):
        return f"{price:,.2f}" if price >= 1 else f"{price:.8f}"
    
    # تحديد لون القرار
    action_colors = {
        'شراء': 'success',
        'بيع': 'danger',
        'انتظار': 'warning'
    }
    action_color = action_colors.get(smart_decision.get('action', 'انتظار'), 'warning')
    
    # تحديد مستوى الثقة
    confidence = smart_decision.get('confidence', 0) * 100
    confidence_color = (
        'danger' if confidence < 40 else
        'warning' if confidence < 70 else
        'success'
    )

    return html.Div([
        # القرار الرئيسي
        html.Div([
            html.H4("قرار التداول الذكي", className="card-header bg-dark text-light"),
            html.Div([
                # القرار والثقة
                html.Div([
                    html.H5([
                        "القرار: ",
                        html.Span(
                            smart_decision.get('action', 'انتظار'),
                            className=f"badge bg-{action_color}"
                        )
                    ], className="mb-3"),
                    html.Div([
                        "مستوى الثقة: ",
                        html.Div([
                            html.Div(
                                style={
                                    "width": f"{confidence}%",
                                    "backgroundColor": f"var(--bs-{confidence_color})"
                                },
                                className="progress-bar",
                                children=f"{confidence:.1f}%"
                            )
                        ], className="progress")
                    ], className="mb-3"),
                ], className="mb-4"),

                # تفاصيل الأسعار
                html.Div([
                    html.H6("مستويات الأسعار:", className="mb-3"),
                    html.Table([
                        html.Tbody([
                            html.Tr([
                                html.Td("سعر الدخول:", className="text-muted"),
                                html.Td(format_price(smart_decision.get('entry_price', 0)), className="text-end")
                            ]),
                            html.Tr([
                                html.Td("وقف الخسارة:", className="text-muted"),
                                html.Td(format_price(smart_decision.get('stop_loss', 0)), className="text-end text-danger")
                            ]),
                            html.Tr([
                                html.Td("جني الأرباح:", className="text-muted"),
                                html.Td(format_price(smart_decision.get('take_profit', 0)), className="text-end text-success")
                            ]),
                            html.Tr([
                                html.Td("نسبة المخاطرة/المكافأة:", className="text-muted"),
                                html.Td(f"{smart_decision.get('risk_reward_ratio', 0):.2f}", className="text-end")
                            ])
                        ])
                    ], className="table table-sm")
                ], className="mb-4"),

                # أسباب القرار
                html.Div([
                    html.H6("أسباب القرار:", className="mb-2"),
                    html.Ul([
                        html.Li(reason) for reason in smart_decision.get('reasoning', [])
                    ], className="list-unstyled small")
                ])
            ], className="card-body")
        ], className="card shadow-sm mb-4"),

        # معلومات إضافية
        html.Div([
            html.H4("معلومات إضافية", className="card-header bg-dark text-light"),
            html.Div([
                html.Div([
                    html.Strong("المؤشرات المستخدمة:"),
                    html.Ul([
                        html.Li([
                            f"{indicator}: ",
                            html.Span(f"{weight * 100:.0f}%", className="badge bg-secondary")
                        ]) for indicator, weight in decisions.get('weights', {}).items()
                    ], className="list-unstyled small")
                ], className="mb-3"),

                html.Div([
                    html.Strong("آخر تحديث: "),
                    html.Span(
                        decisions.get('timestamp', '').replace('T', ' ').split('.')[0],
                        className="text-muted small"
                    )
                ])
            ], className="card-body")
        ], className="card shadow-sm")
    ])

def create_candlestick_chart(df: pd.DataFrame, df_with_indicators=None, patterns=None) -> go.Figure:
    """إنشاء مخطط الشموع اليابانية مع المؤشرات والأنماط"""
    try:
        if df.empty:
            return go.Figure()

        # إنشاء مخطط الشموع الأساسي
        fig = go.Figure()

        # إضافة الشموع
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))

        # إضافة المتوسطات المتحركة إذا كانت متوفرة
        ma_colors = {
            'SMA_20': '#2196F3',  # أزرق
            'SMA_50': '#9C27B0',  # بنفسجي
            'EMA_20': '#FF9800',  # برتقالي
            'EMA_50': '#795548'   # بني
        }

        for ma_name, color in ma_colors.items():
            if ma_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[ma_name],
                    name=ma_name,
                    line=dict(color=color, width=1),
                    opacity=0.7
                ))

        # إضافة حزم بولينجر إذا كانت متوفرة
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(200, 200, 200, 0.1)',
                showlegend=True
            ))

        # إضافة مؤشر MACD في نافذة منفصلة
        if all(col in df_with_indicators.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            try:
                # التأكد من أن القيم منطقية
                macd_data = df_with_indicators['MACD'].copy()
                signal_data = df_with_indicators['MACD_Signal'].copy()
                hist_data = df_with_indicators['MACD_Hist'].copy()
                
                # تنظيف القيم غير المنطقية
                max_value = 100
                macd_data[abs(macd_data) > max_value] = 0
                signal_data[abs(signal_data) > max_value] = 0
                hist_data[abs(hist_data) > max_value] = 0
                
                # إضافة نافذة منفصلة لـ MACD
                fig.add_trace(go.Scatter(
                    x=df_with_indicators.index,
                    y=macd_data,
                    name='MACD',
                    line=dict(color='blue', width=2),
                    yaxis="y2"
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_with_indicators.index,
                    y=signal_data,
                    name='Signal',
                    line=dict(color='orange', width=2),
                    yaxis="y2"
                ))
                
                fig.add_trace(go.Bar(
                    x=df_with_indicators.index,
                    y=hist_data,
                    name='Histogram',
                    marker_color=np.where(hist_data >= 0, 'green', 'red'),
                    yaxis="y2"
                ))
                
                # تحديث تخطيط الشارت لإضافة النافذة الثانية
                fig.update_layout(
                    yaxis2=dict(
                        title="MACD",
                        overlaying="y",
                        side="right",
                        showgrid=True,
                        domain=[0, 0.2]  # يأخذ 20% من ارتفاع الشارت
                    ),
                    yaxis=dict(domain=[0.3, 1])  # يأخذ 70% من ارتفاع الشارت
                )
                
            except Exception as e:
                logging.error(f"خطأ في رسم MACD: {str(e)}")

        # إضافة الأنماط المكتشفة على الشارت
        if patterns:
            for pattern in patterns:
                pattern_type = pattern.get('type', '')
                pattern_name = pattern.get('name', '')
                pattern_price = pattern.get('price', 0)
                
                if pattern_price > 0:
                    # تحديد لون ورمز النمط
                    color = '#26a69a' if pattern_type == 'bullish' else '#ef5350'
                    symbol = '▲' if pattern_type == 'bullish' else '▼'
                    
                    # إضافة علامة النمط
                    fig.add_annotation(
                        x=df.index[-1],
                        y=pattern_price,
                        text=f"{symbol} {pattern_name}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        font=dict(size=10, color=color),
                        align='right',
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor=color,
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8
                    )

        # تحسين مظهر الشارت
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text='تحليل الشموع اليابانية',
                x=0.5,
                xanchor='center',
                font=dict(size=24, family='Arial')
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            ),
            xaxis=dict(
                title='الوقت',
                gridcolor='rgba(255,255,255,0.1)',
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(
                title='السعر',
                gridcolor='rgba(255,255,255,0.1)',
                side='right'
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0.8)',
                font_size=12,
                font_family='Arial'
            )
        )

        return fig

    except Exception as e:
        logging.error(f"خطأ في إنشاء الشارت: {str(e)}", exc_info=True)
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            title=dict(text=f'خطأ: {str(e)}', x=0.5, xanchor='center')
        )
        return empty_fig

def create_technical_view(indicators: Dict[str, Any]) -> html.Div:
    """إنشاء عرض المؤشرات الفنية"""
    try:
        if not indicators:
            return html.Div([
                html.H5("لا توجد مؤشرات فنية متاحة", className="text-muted text-center p-3")
            ])

        # تصنيف المؤشرات حسب الإشارة
        bullish_indicators = []
        bearish_indicators = []
        neutral_indicators = []

        for name, data in indicators.items():
            signal = data.get('signal', '').lower()
            if any(term in signal for term in ['شراء', 'صعود', 'buy', 'bullish']):
                bullish_indicators.append((name, data))
            elif any(term in signal for term in ['بيع', 'هبوط', 'sell', 'bearish']):
                bearish_indicators.append((name, data))
            else:
                neutral_indicators.append((name, data))

        def create_indicator_card(name: str, data: Dict[str, Any]) -> html.Div:
            """إنشاء بطاقة مؤشر"""
            signal = data.get('signal', '').lower()
            color_class = 'success' if any(term in signal for term in ['شراء', 'صعود', 'buy', 'bullish']) else \
                         'danger' if any(term in signal for term in ['بيع', 'هبوط', 'sell', 'bearish']) else \
                         'warning'

            return html.Div([
                html.Div([
                    html.H6(name, className=f"text-{color_class} mb-2"),
                    html.P([
                        html.Strong("الإشارة: "),
                        html.Span(data.get('signal', ''), className=f"text-{color_class}")
                    ], className="small mb-1"),
                    html.P([
                        html.Strong("القيمة: "),
                        f"{data.get('value', 0):.2f}"
                    ], className="small mb-1"),
                    html.P([
                        html.Strong("القوة: "),
                        f"{data.get('strength', 0)*100:.0f}%"
                    ], className="small mb-1"),
                    html.P(data.get('description', ''), className="small text-muted")
                ], className="card-body")
            ], className=f"card border-{color_class} mb-2 indicator-card")

        # إنشاء أقسام المؤشرات
        sections = []
        
        if bullish_indicators:
            sections.extend([
                html.H5("إشارات شراء", className="text-success mb-3"),
                html.Div([create_indicator_card(name, data) for name, data in bullish_indicators], className="mb-4")
            ])
            
        if bearish_indicators:
            sections.extend([
                html.H5("إشارات بيع", className="text-danger mb-3"),
                html.Div([create_indicator_card(name, data) for name, data in bearish_indicators], className="mb-4")
            ])
            
        if neutral_indicators:
            sections.extend([
                html.H5("إشارات محايدة", className="text-warning mb-3"),
                html.Div([create_indicator_card(name, data) for name, data in neutral_indicators], className="mb-4")
            ])

        return html.Div([
            html.H4("المؤشرات الفنية", className="text-primary mb-4"),
            *sections
        ], className="p-3")

    except Exception as e:
        logging.error(f"خطأ في إنشاء عرض المؤشرات: {str(e)}", exc_info=True)
        return html.Div([
            html.H5("خطأ في عرض المؤشرات", className="text-danger"),
            html.P(str(e), className="text-danger-emphasis")
        ], className="p-3")

def create_trading_signals_view(patterns: List[Dict[str, Any]]) -> html.Div:
    """إنشاء عرض إشارات التداول"""
    try:
        if not patterns:
            return html.Div([
                html.H5("لا توجد أنماط مكتشفة", className="text-muted text-center p-3")
            ])

        # تصنيف الأنماط حسب النوع
        bullish_patterns = [p for p in patterns if p.get('type') == 'bullish']
        bearish_patterns = [p for p in patterns if p.get('type') == 'bearish']
        neutral_patterns = [p for p in patterns if p.get('type') not in ['bullish', 'bearish']]

        def create_pattern_card(pattern: Dict[str, Any]) -> html.Div:
            """إنشاء بطاقة نمط"""
            pattern_type = pattern.get('type', '')
            color_class = {
                'bullish': 'success',
                'bearish': 'danger',
                'neutral': 'warning'
            }.get(pattern_type, 'info')

            return html.Div([
                html.Div([
                    html.H6(pattern.get('name', ''), className=f"text-{color_class} mb-2"),
                    html.P([
                        html.Strong("القوة: "),
                        f"{pattern.get('strength', 0)*100:.0f}%"
                    ], className="small mb-1"),
                    html.P([
                        html.Strong("الهدف السعري: "),
                        f"{pattern.get('price', 0):.2f}"
                    ], className="small mb-1"),
                    html.P(pattern.get('description', ''), className="small text-muted")
                ], className="card-body")
            ], className=f"card border-{color_class} mb-2")

        # إنشاء أقسام الأنماط
        sections = []
        
        if bullish_patterns:
            sections.extend([
                html.H5("الأنماط الصعودية", className="text-success mb-3"),
                html.Div([create_pattern_card(p) for p in bullish_patterns], className="mb-4")
            ])
            
        if bearish_patterns:
            sections.extend([
                html.H5("الأنماط الهبوطية", className="text-danger mb-3"),
                html.Div([create_pattern_card(p) for p in bearish_patterns], className="mb-4")
            ])
            
        if neutral_patterns:
            sections.extend([
                html.H5("الأنماط المحايدة", className="text-warning mb-3"),
                html.Div([create_pattern_card(p) for p in neutral_patterns], className="mb-4")
            ])

        return html.Div([
            html.H4("تحليل الأنماط", className="text-primary mb-4"),
            *sections
        ], className="p-3")

    except Exception as e:
        logging.error(f"خطأ في إنشاء عرض الأنماط: {str(e)}", exc_info=True)
        return html.Div([
            html.H5("خطأ في عرض الأنماط", className="text-danger"),
            html.P(str(e), className="text-danger-emphasis")
        ], className="p-3")

def create_decision_view(signals: Dict[str, Any]) -> html.Div:
    """إنشاء عرض متخذ القرارات"""
    try:
        if not signals or not isinstance(signals, dict):
            return html.Div("لا توجد بيانات كافية لاتخاذ قرار", className="text-danger p-3")

        # تحليل الإشارات واتخاذ القرار
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # تحليل الأنماط
        patterns = signals.get('patterns', [])
        for pattern in patterns:
            if isinstance(pattern, dict):
                if pattern.get('type') == 'bullish':
                    bullish_signals += 1
                elif pattern.get('type') == 'bearish':
                    bearish_signals += 1
                total_signals += 1

        # تحليل المؤشرات الفنية
        technical = signals.get('technical', {})
        for indicator, value in technical.items():
            try:
                if isinstance(value, dict):
                    signal = value.get('signal', 'محايد')
                else:
                    signal = str(value)
                
                if any(term in signal for term in ['شراء', 'صعود', 'buy', 'bullish']):
                    bullish_signals += 1
                elif any(term in signal for term in ['بيع', 'هبوط', 'sell', 'bearish']):
                    bearish_signals += 1
                total_signals += 1
            except (AttributeError, TypeError):
                continue

        # تحليل إشارات السوق
        market = signals.get('market', {})
        for indicator, value in market.items():
            try:
                if isinstance(value, dict):
                    signal = value.get('signal', 'محايد')
                else:
                    signal = str(value)
                
                if any(term in signal for term in ['شراء', 'صعود', 'buy', 'bullish']):
                    bullish_signals += 1
                elif any(term in signal for term in ['بيع', 'هبوط', 'sell', 'bearish']):
                    bearish_signals += 1
                total_signals += 1
            except (AttributeError, TypeError):
                continue

        # حساب القوة النسبية للإشارات
        if total_signals > 0:
            bullish_strength = (bullish_signals / total_signals) * 100
            bearish_strength = (bearish_signals / total_signals) * 100
        else:
            bullish_strength = bearish_strength = 0

        # اتخاذ القرار
        if bullish_strength > 60:
            decision = 'شراء'
            decision_color = 'success'
            icon = 'fa-arrow-trend-up'
        elif bearish_strength > 60:
            decision = 'بيع'
            decision_color = 'danger'
            icon = 'fa-arrow-trend-down'
        else:
            decision = 'انتظار'
            decision_color = 'warning'
            icon = 'fa-arrows-left-right'

        return html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("متخذ القرارات", className="text-primary mb-0")
                ], className="border-0 bg-transparent"),
                dbc.CardBody([
                    # القرار الرئيسي
                    html.Div([
                        html.I(className=f"fas {icon} fa-2x text-{decision_color} mb-3"),
                        html.H3(decision, className=f"text-{decision_color} mb-4")
                    ], className="text-center"),

                    # قوة الإشارات
                    html.Div([
                        html.H6("قوة الإشارات", className="mb-3"),
                        html.Div([
                            html.Div([
                                html.Span("إشارات صعودية", className="text-success"),
                                dbc.Progress(
                                    value=bullish_strength,
                                    color="success",
                                    className="mb-2",
                                    style={"height": "8px"}
                                )
                            ], className="mb-3"),
                            html.Div([
                                html.Span("إشارات هبوطية", className="text-danger"),
                                dbc.Progress(
                                    value=bearish_strength,
                                    color="danger",
                                    className="mb-2",
                                    style={"height": "8px"}
                                )
                            ])
                        ], className="mb-4")
                    ]),

                    # تفاصيل الإشارات
                    dbc.Alert([
                        html.H6("تفاصيل التحليل:", className="alert-heading mb-3"),
                        html.Div([
                            html.Div([
                                html.Strong("عدد الإشارات الصعودية: "),
                                html.Span(f"{bullish_signals}")
                            ], className="mb-2"),
                            html.Div([
                                html.Strong("عدد الإشارات الهبوطية: "),
                                html.Span(f"{bearish_signals}")
                            ], className="mb-2"),
                            html.Div([
                                html.Strong("إجمالي الإشارات: "),
                                html.Span(f"{total_signals}")
                            ])
                        ])
                    ], color="info", className="mb-0")
                ])
            ], className="shadow-sm")
        ], className="mb-4")

    except Exception as e:
        logging.error(f"خطأ في إنشاء عرض متخذ القرارات: {str(e)}", exc_info=True)
        return html.Div([
            html.H5("خطأ في متخذ القرارات", className="text-danger mb-2"),
            html.P(str(e), className="text-danger-emphasis")
        ], className="p-3")

class TechnicalAnalyzer:
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحليل البيانات وإرجاع النتائج"""
        try:
            # حساب المؤشرات الفنية
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.volatility.BollingerBands(
                df['close'],
                window=20,
                window_dev=2
            )
            
            logging.info("تم اكتمال التحليل الفني")
            logging.debug(f"نتائج التحليل: {df}")
            
            return df
            
        except Exception as e:
            logging.error(f"خطأ في التحليل الفني: {str(e)}")
            logging.error(traceback.format_exc())
            return df

if __name__ == '__main__':
    try:
        logging.info("بدء تشغيل التطبيق...")
        app.run_server(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        logging.error(f"خطأ في تشغيل التطبيق: {str(e)}")
