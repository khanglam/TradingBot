import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, IntegerField, FileField, SubmitField
from wtforms.validators import DataRequired
import yaml
import pyotp
from app.trading.engine import PaperTradingEngine
from app.strategies.sample_strategy import SampleMovingAverageStrategy
import pandas as pd
import sqlite3

# --- Config Loading ---
with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')
login_manager = LoginManager()
login_manager.init_app(app)

# --- User Model (for demo) ---
class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.otp_secret = pyotp.random_base32()

users = {'admin': User('admin')}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# --- Forms ---
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    otp = StringField('2FA Code')
    submit = SubmitField('Login')

class StrategyForm(FlaskForm):
    fast = IntegerField('Fast MA', default=10)
    slow = IntegerField('Slow MA', default=30)
    submit = SubmitField('Update Strategy')

# --- Portfolio, Logging, Alerts ---
from app.portfolio.manager import PortfolioManager
from app.logging.logger import log_trade, log_error, log_event
from app.alerts.notifier import send_email, send_telegram

portfolio = PortfolioManager(base_currency=config['trading']['base_currency'], initial_balance=config['trading']['initial_balance'])
engine = PaperTradingEngine(initial_balance=config['trading']['initial_balance'])
strategy = SampleMovingAverageStrategy()

# --- SQLite Setup ---
def get_db():
    conn = sqlite3.connect(config['database']['path'])
    conn.row_factory = sqlite3.Row
    return conn

def save_trade(trade):
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, symbol TEXT, price REAL, qty REAL, profit REAL, time TEXT
    )''')
    c.execute('''INSERT INTO trades (type, symbol, price, qty, profit, time) VALUES (?, ?, ?, ?, ?, ?)''',
        (trade.get('type'), trade.get('symbol'), trade.get('price'), trade.get('qty'), trade.get('profit', 0), str(trade.get('time'))))
    conn.commit()
    conn.close()

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
@login_required
def dashboard():
    balance = portfolio.get_balance()
    open_positions = portfolio.get_open_positions()
    closed_positions = portfolio.get_closed_positions()
    # For demo: simulate latest prices
    latest_prices = {pos['symbol']: pos['entry_price'] for pos in open_positions}
    portfolio_value = portfolio.get_portfolio_value(latest_prices)
    return render_template('dashboard.html', balance=balance, open_positions=open_positions,
                           closed_positions=closed_positions, portfolio_value=portfolio_value, yaml_config=config)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        otp = form.otp.data
        if username == 'admin' and password == 'admin':
            if config['security'].get('enable_2fa'):
                # For demo, OTP is always valid
                if not otp or len(otp) != 6:
                    flash('Enter any 6-digit 2FA code for demo.')
                    return render_template('login.html', form=form, yaml_config=config)
            login_user(users['admin'])
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('login.html', form=form, yaml_config=config)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/strategy', methods=['GET', 'POST'])
@login_required
def strategy_settings():
    form = StrategyForm()
    if form.validate_on_submit():
        strategy.fast = form.fast.data
        strategy.slow = form.slow.data
        flash('Strategy parameters updated!')
    return render_template('strategy.html', form=form, fast=strategy.fast, slow=strategy.slow, yaml_config=config)

@app.route('/run_backtest')
@login_required
def run_backtest():
    # For demo: use random data
    df = pd.DataFrame({'close': pd.Series([100 + i for i in range(100)])})
    signals = strategy.generate_signals(df)
    # Simulate trades (stub)
    flash('Backtest completed (demo)!')
    return redirect(url_for('dashboard'))

# --- Backtesting & Optimization ---
from app.backtesting.engine import run_backtest
from app.optimization.optimizer import grid_search
from app.strategies.sample_strategy import SampleMovingAverageStrategy
from app.strategies.advanced_ta.LorentzianClassification.Classifier import LorentzianClassification

from app.strategies.advanced_ta.LorentzianClassification.Types import Settings

from app.strategies.advanced_ta.LorentzianClassification.Types import FilterSettings, KernelFilter
import numpy as np

class LorentzianClassificationStrategy:
    def __init__(self, **kwargs):
        self.settings_kwargs = kwargs

    def generate_signals(self, df):
        print('DEBUG: Input DataFrame columns:', df.columns)
        print('DEBUG: First few rows:\n', df.head(10))
        print('DEBUG: DataFrame null counts:')
        print(df.isnull().sum())
        print('DEBUG: DataFrame describe:')
        print(df.describe())
        settings = Settings(
            source=df['close'],
            neighborsCount=self.settings_kwargs.get('neighbors_count', 8),
            maxBarsBack=self.settings_kwargs.get('max_bars_back', 2000),
        )
        # Disable all filters for diagnosis
        filterSettings = FilterSettings(
            useVolatilityFilter=False,
            useRegimeFilter=False,
            useAdxFilter=False,
            regimeThreshold=-0.1,
            adxThreshold=20,
            kernelFilter=KernelFilter()
        )
        model = LorentzianClassification(
            df,
            neighbors_count=self.settings_kwargs.get('neighbors_count', 8),
            max_bars_back=self.settings_kwargs.get('max_bars_back', 2000),
            feature_count=5,  # or make this configurable
            feature_defs=[
                ("RSI", 14, 2),
                ("WT", 10, 11),
                ("CCI", 20, 2),
                ("ADX", 20, 2),
                ("RSI", 9, 2)
            ],
            use_volatility_filter=False,
            use_regime_filter=False,
            use_adx_filter=False,
            regime_threshold=-0.1,
            adx_threshold=20,
            kernel_filter=KernelFilter()
        )
        # Print feature stats
        for i, f in enumerate(model.features):
            print(f"Feature {i} min: {np.nanmin(f)}, max: {np.nanmax(f)}, nan count: {np.isnan(f).sum()}")
        pred = model.df['prediction']
        print('Signal value counts:', pred.value_counts(dropna=False))
        print(model.df[['prediction', 'signal', 'isNewBuySignal', 'isNewSellSignal']].tail(20))
        # Map all negative predictions to -1 (SHORT), all positive to 1 (LONG), zero to 0 (NEUTRAL)
        mapped_pred = pred.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        print('Mapped signal value counts:', mapped_pred.value_counts(dropna=False))
        return mapped_pred


@app.route('/backtest', methods=['GET', 'POST'])
@login_required
def backtest():
    results = None
    fast = 10
    slow = 30
    mode = 'backtest'
    strategy_choice = 'ma'
    neighbors_count = 8
    max_bars_back = 200
    symbol = 'BTCUSD'
    interval = '1d'
    if request.method == 'POST':
        fast = int(request.form.get('fast', 10))
        slow = int(request.form.get('slow', 30))
        mode = request.form.get('mode', 'backtest')
        strategy_choice = request.form.get('strategy_choice', 'ma')
        neighbors_count = int(request.form.get('neighbors_count', 8))
        max_bars_back = int(request.form.get('max_bars_back', 200))
        symbol = request.form.get('symbol', 'BTCUSD')
        interval = request.form.get('interval', '1d')
        # Handle CSV upload
        if 'csv' in request.files and request.files['csv'].filename:
            f = request.files['csv']
            df = pd.read_csv(f)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime').reset_index(drop=True)
        else:
            # Try TradingView via tvdatafeed, fallback to yfinance
            try:
                from tvdatafeed import TvDatafeed, Interval
                tv = TvDatafeed()
                tv_interval = getattr(Interval, interval, Interval.in_daily)
                df = tv.get_hist(symbol=symbol, exchange='BINANCE', interval=tv_interval)
            except Exception:
                try:
                    import yfinance as yf
                    yf_symbol = symbol if symbol.endswith('=X') else symbol+'-USD'
                    df = yf.download(yf_symbol, interval=interval)
                    df = df.rename(columns={c:c.lower() for c in df.columns})
                    df = df.reset_index()
                except Exception:
                    import numpy as np
                    x = np.arange(100)
                    prices = 100 + 10 * np.sin(x / 5)
                    df = pd.DataFrame({'close': prices})
                    df['open'] = df['close']
                    df['high'] = df['close'] + 1
                    df['low'] = df['close'] - 1
        # --- Ensure required OHLC columns exist and are numeric ---
        import numpy as np
        if df is not None:
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df.ffill()
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'close' not in df:
            df['close'] = df.iloc[:, -1]
        if strategy_choice == 'ma':
            strat = SampleMovingAverageStrategy(fast=fast, slow=slow)
        else:
            strat = LorentzianClassificationStrategy(neighbors_count=neighbors_count, max_bars_back=max_bars_back)
        if mode == 'backtest':
            if strategy_choice != 'ma':
                signals = strat.generate_signals(df)
                print('Signal value counts:', signals.value_counts())
                print('Signals preview:', signals.head(20))
                # DEBUG: Print the last rows of the model DataFrame if possible
                if hasattr(strat, 'get_full_output'):
                    model_df = strat.get_full_output(df)
                    print('Model prediction counts:', model_df['prediction'].value_counts(dropna=False))
                    print(model_df[['prediction', 'isNewBuySignal', 'isNewSellSignal']].tail(20))
                results = run_backtest(df, lambda x: signals)
            else:
                results = run_backtest(df, strat.generate_signals)
        elif mode == 'optimize':
            if strategy_choice == 'ma':
                param_grid = {'fast': list(range(5, 16, 2)), 'slow': list(range(20, 41, 5))}
                def bt_func(fast, slow, data):
                    strat = SampleMovingAverageStrategy(fast=fast, slow=slow)
                    return run_backtest(data, strat.generate_signals)
                best_params, best_metrics = grid_search(param_grid, bt_func, df)
                results = best_metrics
                fast, slow = best_params['fast'], best_params['slow']
            else:
                results = run_backtest(df, LorentzianClassificationStrategy(neighbors_count=neighbors_count, max_bars_back=max_bars_back).generate_signals)
    return render_template('backtest.html', results=results, fast=fast, slow=slow, mode=mode, strategy_choice=strategy_choice, neighbors_count=neighbors_count, max_bars_back=max_bars_back, symbol=symbol, interval=interval, yaml_config=config)

# --- PineScript Upload & Conversion ---
from app.indicators.pine_converter import PineScriptConverter

@app.route('/pine_upload', methods=['GET', 'POST'])
@login_required
def pine_upload():
    python_code = None
    pinescript_code = ''
    if request.method == 'POST':
        pinescript_code = request.form.get('pinescript', '')
        if pinescript_code:
            converter = PineScriptConverter(pinescript_code)
            python_code = converter.to_python_func()
    return render_template('pine_upload.html', pinescript_code=pinescript_code, python_code=python_code, yaml_config=config)

# --- Community Sharing ---
import os
from flask import send_from_directory

COMMUNITY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'community_uploads'))
os.makedirs(COMMUNITY_DIR, exist_ok=True)

@app.route('/community', methods=['GET', 'POST'])
@login_required
def community():
    items = []
    if request.method == 'POST':
        f = request.files.get('file')
        desc = request.form.get('desc', '')
        if f and desc:
            fname = f.filename
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ['.py', '.txt', '.pine']:
                flash('Only Python or PineScript files allowed!')
            else:
                save_path = os.path.join(COMMUNITY_DIR, fname)
                f.save(save_path)
                meta_path = save_path + '.meta'
                with open(meta_path, 'w', encoding='utf-8') as meta:
                    meta.write(desc)
                flash('Upload successful!')
    # List all uploads
    for fname in os.listdir(COMMUNITY_DIR):
        if fname.endswith('.py') or fname.endswith('.txt') or fname.endswith('.pine'):
            meta_path = os.path.join(COMMUNITY_DIR, fname + '.meta')
            desc = ''
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as meta:
                    desc = meta.read()
            items.append({
                'name': os.path.splitext(fname)[0],
                'desc': desc,
                'type': 'Python' if fname.endswith('.py') else 'PineScript',
                'filename': fname
            })
    return render_template('community.html', items=items, yaml_config=config)

@app.route('/community/download/<filename>')
@login_required
def download_community(filename):
    return send_from_directory(COMMUNITY_DIR, filename, as_attachment=True)

# --- Logs Viewer ---

@app.route('/logs')
@login_required
def logs():
    lines = int(request.args.get('lines', 100))
    log_file = os.path.join(os.path.dirname(__file__), '..', 'tradingbot.log')
    log_file = os.path.abspath(log_file)
    log_content = ''
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            log_content = ''.join(all_lines[-lines:])
    return render_template('logs.html', log_content=log_content, lines=lines, yaml_config=config)

# --- Error Handlers ---
@app.errorhandler(401)
def unauthorized(e):
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host=config['ui']['host'], port=config['ui']['port'], debug=True)
