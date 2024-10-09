import os
import time
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
import numpy as np
import pyupbit
import requests
from dotenv import load_dotenv
from notion_client import Client

# =====================
# 환경 변수 로드
# =====================
load_dotenv()

# 텔레그램 설정
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Notion 설정
NOTION_TOKEN = os.getenv('NOTION_TOKEN')
NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')
notion = Client(auth=NOTION_TOKEN)

# Upbit API 키 설정
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

# 상태 파일 경로
STATE_FILE = 'bot_state.json'

# 로깅 설정
logging.basicConfig(
    filename='autotrader.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# =====================
# 유틸리티 함수
# =====================

def save_state(state, filename=STATE_FILE):
    """현재 봇의 상태를 JSON 파일에 저장합니다."""
    try:
        with open(filename, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"Exception in save_state: {e}")

def load_state(filename=STATE_FILE):
    """JSON 파일에서 봇의 상태를 불러옵니다."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Exception in load_state: {e}")
    # 초기 상태 반환
    return {
        "position": 0,                # 포지션 여부 (0: 없음, 1: 보유 중)
        "current_ticker": None,       # 현재 보유 중인 코인의 티커
        "qty": 0.0,                   # 보유 중인 코인의 수량
        "last_processed_time": None,  # 마지막으로 처리한 캔들의 시간
        "initial_balance": 0.0,       # 초기 KRW 잔고
        "entry_time": None,           # 포지션 진입 시간
        "last_trade_id": 0            # 마지막 거래 ID
    }

def send_telegram_message(message: str):
    """텔레그램 봇을 통해 메시지를 전송합니다."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        response = requests.post(url, data=data)
        if response.status_code != 200:
            logging.error(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        logging.error(f"Exception in send_telegram_message: {e}")

def log_to_notion(trade: dict):
    """Notion 데이터베이스에 거래 내역을 기록합니다."""
    try:
        properties = {
            "Trade ID": {"title": [{"text": {"content": str(trade['Trade ID'])}}]},
            "Ticker": {"rich_text": [{"text": {"content": trade['Ticker']}}]},  # 티커 추가
            "Entry Time": {"date": {"start": trade['Entry Time'].isoformat()}},
            "Exit Time": {"date": {"start": trade['Exit Time'].isoformat()} if trade['Exit Time'] else None},
            "Signal Type": {"rich_text": {"name": trade['Signal Type']}},
            "Entry Price": {"number": trade['Entry Price']},
            "Exit Price": {"number": trade['Exit Price'] if trade['Exit Price'] is not None else None},
            "Log Close Price": {"number": trade['Log Close Price']},
            "Resistance Level": {"number": trade['Resistance Level']},
            "Support Level": {"number": trade['Support Level']},
            "Volume": {"number": trade['Volume']},
            "ATR": {"number": trade['ATR']},
            "Reason for Entry": {"rich_text": [{"text": {"content": trade['Reason for Entry']}}]},
            "Reason for Exit": {"rich_text": [{"text": {"content": trade['Reason for Exit']}}] if trade['Reason for Exit'] else None},
            "Profit/Loss": {"number": trade['Profit/Loss'] if trade['Profit/Loss'] is not None else None},
            "Return (%)": {"number": trade['Return (%)'] if trade['Return (%)'] is not None else None},
        }
        notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties=properties
        )
    except Exception as e:
        logging.error(f"Exception in log_to_notion: {e}")

# =====================
# 트렌드라인 분석 함수
# =====================

def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0
    err = (diffs ** 2.0).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    slope_unit = (y.max() - y.min()) / len(y)
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0)
    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                raise Exception("Derivative failed. Check your data.")
            get_derivative = False
        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step
        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True
    return (best_slope, -best_slope * pivot + y[pivot])

def fit_trendlines_single(data: np.array):
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
    return (support_coefs, resist_coefs)

# =====================
# 실시간 거래 신호 생성 함수
# =====================

def generate_trade_signal(ohlcv: pd.DataFrame, lookback: int, tp_mult: float, sl_mult: float, atr_mult: float):
    """
    최신 캔들에 대한 거래 신호를 생성합니다.
    """
    if len(ohlcv) < lookback + 1:
        return None  # 데이터 부족

    window = ohlcv.iloc[-lookback -1:-1]  # 최신 캔들 제외
    latest_candle = ohlcv.iloc[-1]

    close = window['close'].to_numpy()
    try:
        s_coefs, r_coefs = fit_trendlines_single(close)
    except Exception as e:
        logging.error(f"Exception in fit_trendlines_single: {e}")
        return None

    r_val = r_coefs[1] + len(close) * r_coefs[0]
    s_val = s_coefs[1] + len(close) * s_coefs[0]

    # ATR 계산
    atr = ta.atr(window['high'], window['low'], window['close'], length=14)
    latest_atr = atr.iloc[-1]

    signal = None
    if latest_candle['close'] > r_val:
        signal = 'Buy'
    elif latest_candle['close'] < s_val:
        signal = 'Sell'

    if signal == 'Buy':
        tp_price = latest_candle['close'] + latest_atr * tp_mult
        sl_price = latest_candle['close'] - latest_atr * sl_mult
        return {'Signal': 'Buy', 'Entry Price': latest_candle['close'], 'TP': tp_price, 'SL': sl_price, 'Resistance': r_val, 'Support': s_val}
    elif signal == 'Sell':
        tp_price = latest_candle['close'] - latest_atr * tp_mult
        sl_price = latest_candle['close'] + latest_atr * sl_mult
        return {'Signal': 'Sell', 'Entry Price': latest_candle['close'], 'TP': tp_price, 'SL': sl_price, 'Resistance': r_val, 'Support': s_val}
    else:
        return None

# =====================
# 매수/매도 주문 실행 함수
# =====================

def execute_trade(signal: dict, state: dict, ohlcv: pd.DataFrame, ticker: str = None, reason_override: str = None):
    """
    거래 신호에 따라 매수 또는 매도 주문을 실행하고 상태를 업데이트합니다.
    `ticker`는 거래할 코인의 티커입니다.
    `reason_override`는 자동 매도 시 'Hold Period Exceeded'와 같은 이유를 지정할 때 사용됩니다.
    """
    trade_id = state.get('last_trade_id', 0) + 1
    signal_type = signal['Signal']
    entry_price = signal['Entry Price']
    tp_price = signal.get('TP', None)
    sl_price = signal.get('SL', None)
    resistance = signal.get('Resistance', None)
    support = signal.get('Support', None)
    current_time = ohlcv.index[-1]
    volume = ohlcv['volume'].iloc[-1]
    atr = ta.atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=14).iloc[-1]

    # Ticker 기본값 설정
    if ticker is None:
        ticker = "KRW-BTC"

    if signal_type == 'Buy' and state['position'] == 0:
        krw_balance = upbit.get_balance("KRW")
        if krw_balance is None:
            krw_balance = 0.0
        trade_amount = krw_balance * 0.3  # KRW 잔고의 30% 사용
        if trade_amount < 5000:
            logging.info(f"Trade ID {trade_id} skipped due to insufficient KRW balance.")
            print(f"[{datetime.now()}] Trade ID {trade_id} skipped due to insufficient KRW balance.")
            return state
        try:
            # 시장가 매수
            order = upbit.buy_market_order(ticker, trade_amount)
            logging.info(f"Trade ID {trade_id} Buy Order: {order}")
            print(f"[{datetime.now()}] Trade ID {trade_id}: Buy order placed for {trade_amount:,.0f} KRW")
            time.sleep(1)  # API 호출 간 대기

            # 실제 매수 가격 확인
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                raise Exception("Failed to get current price after buy order.")
            qty_bought = trade_amount / current_price if current_price else 0.0
            state['position'] = 1
            state['current_ticker'] = ticker
            state['qty'] = qty_bought
            state['last_trade_id'] = trade_id
            state['last_processed_time'] = current_time.isoformat()
            state['entry_price'] = current_price  # 진입 가격 저장
            state['entry_time'] = current_time.isoformat()  # 진입 시간 저장
            save_state(state)

            # Notion 로그
            trade_info = {
                'Trade ID': trade_id,
                'Ticker': ticker,  # 티커 포함
                'Entry Time': current_time,
                'Exit Time': None,
                'Signal Type': 'Buy',
                'Entry Price': current_price,
                'Exit Price': None,
                'Log Close Price': np.log(current_price),
                'Resistance Level': resistance,
                'Support Level': support,
                'Volume': volume,
                'ATR': atr,
                'Reason for Entry': 'Price crossed above Resistance',
                'Reason for Exit': None,
                'Profit/Loss': None,
                'Return (%)': None,
            }
            log_to_notion(trade_info)

            # 텔레그램 메시지
            message = (
                f"Buy Order Executed\n"
                f"Trade ID: {trade_id}\n"
                f"Ticker: {ticker}\n"
                f"Buy Price: {current_price:,.0f} KRW\n"
                f"Amount: {trade_amount:,.0f} KRW\n"
                f"Qty: {qty_bought:.6f} {ticker.split('-')[1]}"
            )
            send_telegram_message(message)
        except Exception as e:
            logging.error(f"Exception during buy order for Trade ID {trade_id}: {e}")
            print(f"[{datetime.now()}] Exception during buy order for Trade ID {trade_id}: {e}")

    elif (signal_type == 'Sell' and state['position'] == 1 and state['current_ticker'] == ticker) or \
         (signal_type == 'AutoSell' and state['position'] == 1 and state['current_ticker'] == ticker):
        qty = state['qty']
        entry_price = state.get('entry_price', 0.0)
        entry_time = state.get('entry_time', None)
        try:
            # 시장가 매도
            order = upbit.sell_market_order(ticker, qty)
            logging.info(f"Trade ID {trade_id} Sell Order: {order}")
            print(f"[{datetime.now()}] Trade ID {trade_id}: Sell order placed for {qty:.6f} {ticker}")
            time.sleep(1)  # API 호출 간 대기

            # 실제 매도 가격 확인
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                raise Exception("Failed to get current price after sell order.")
            trade_amount = qty * current_price
            trade_cost = qty * entry_price
            profit_loss = trade_amount - trade_cost
            return_pct = (profit_loss / trade_cost) * 100 if trade_cost != 0 else 0.0

            # 상태 업데이트
            state['position'] = 0
            state['current_ticker'] = None
            state['qty'] = 0.0
            state['last_trade_id'] = trade_id
            state['last_processed_time'] = current_time.isoformat()
            state['entry_price'] = 0.0  # 진입 가격 초기화
            state['entry_time'] = None  # 진입 시간 초기화
            save_state(state)

            # Notion 로그
            trade_info = {
                'Trade ID': trade_id,
                'Ticker': ticker,  # 티커 포함
                'Entry Time': datetime.fromisoformat(entry_time) if entry_time else None,
                'Exit Time': current_time,
                'Signal Type': 'Sell' if signal_type == 'Sell' else 'AutoSell',
                'Entry Price': entry_price,
                'Exit Price': current_price,
                'Log Close Price': np.log(current_price),
                'Resistance Level': resistance,
                'Support Level': support,
                'Volume': volume,
                'ATR': atr,
                'Reason for Entry': 'Price crossed above Resistance',
                'Reason for Exit': 'Price crossed below Support' if signal_type == 'Sell' else 'Hold Period Exceeded',
                'Profit/Loss': profit_loss,
                'Return (%)': return_pct,
            }
            log_to_notion(trade_info)

            # 텔레그램 메시지
            reason = 'Hold Period Exceeded' if signal_type == 'AutoSell' else 'Price crossed below Support'
            message = (
                f"Sell Order Executed\n"
                f"Trade ID: {trade_id}\n"
                f"Ticker: {ticker}\n"
                f"Sell Price: {current_price:,.0f} KRW\n"
                f"Amount: {qty:.6f} {ticker.split('-')[1]}\n"
                f"Profit/Loss: {profit_loss:,.0f} KRW\n"
                f"Return: {return_pct:.2f}%\n"
                f"Reason: {reason}"
            )
            send_telegram_message(message)
        except Exception as e:
            logging.error(f"Exception during sell order for Trade ID {trade_id}: {e}")
            print(f"[{datetime.now()}] Exception during sell order for Trade ID {trade_id}: {e}")
    else:
        # No action required
        pass

    return state

# =====================
# 데이터 로드 함수
# =====================

def get_latest_ohlcv(symbol: str, interval: str, count: int = 100):
    """Upbit API를 통해 최신 OHLCV 데이터를 가져옵니다."""
    try:
        data = pyupbit.get_ohlcv(symbol, interval=interval, count=count)
        return data.dropna()
    except Exception as e:
        logging.error(f"Exception in get_latest_ohlcv: {e}")
        return pd.DataFrame()

def get_top_coins_by_volume(limit=10):
    """Upbit에 상장된 코인 중 거래대금이 가장 큰 상위 `limit`개 코인을 반환합니다."""
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        volumes = []
        for ticker in tickers:
            ohlcv_day = pyupbit.get_ohlcv(ticker, interval="day", count=1)
            if not ohlcv_day.empty:
                # 거래대금 = 종가 * 거래량
                trade_volume = ohlcv_day['close'].iloc[-1] * ohlcv_day['volume'].iloc[-1]
            else:
                trade_volume = 0
            volumes.append(trade_volume)
        df = pd.DataFrame({
            'ticker': tickers,
            'trade_volume': volumes
        })
        top_coins = df.sort_values(by='trade_volume', ascending=False).head(limit)['ticker'].tolist()
        return top_coins
    except Exception as e:
        logging.error(f"Exception in get_top_coins_by_volume: {e}")
        return []

# =====================
# 메인 함수
# =====================

def main():
    # 상태 로드
    state = load_state()

    # 초기 설정
    interval = "minute60"      # Upbit API에서 지원하는 인터벌
    lookback = 72               # 추세선을 계산할 기간
    tp_mult = 3.0               # 테이크 프로핏 배수
    sl_mult = 3.0               # 스톱 로스 배수
    atr_mult = 1.0              # ATR 배수 (현재 사용되지 않음, 필요 시 조정)
    hold_candles = 12           # 포지션을 유지할 최대 캔들 수

    # 봇 시작 메시지 전송
    initial_balance = upbit.get_balance("KRW")
    if initial_balance is None:
        initial_balance = 0.0
    state['initial_balance'] = state.get('initial_balance', initial_balance)
    save_state(state)
    print(f"[{datetime.now()}] Autotrading Bot started. Current KRW Balance: {initial_balance:,.0f} KRW")
    send_telegram_message(f"Autotrading Bot started successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\nCurrent KRW Balance: {initial_balance:,.0f} KRW")

    # Initialize last_processed_time if not set
    if state.get('last_processed_time') is None:
        # Fetch the latest candle's timestamp from top coins
        top_coins = get_top_coins_by_volume()
        if top_coins:
            data = get_latest_ohlcv(top_coins[0], interval=interval, count=1)
            if not data.empty:
                state['last_processed_time'] = data.index[-1].isoformat()
                save_state(state)
                print(f"[{datetime.now()}] Initial last_processed_time set to {state['last_processed_time']}")
            else:
                print("No data fetched during initialization.")
        else:
            print("No top coins fetched during initialization.")

    while True:
        try:
            current_time = datetime.utcnow() + timedelta(hours=9)  # KST 시간으로 변환
            # 다음 60분 마크까지 대기
            minute = current_time.minute
            second = current_time.second
            sleep_seconds = (60 - minute) * 60 - second
            if sleep_seconds <= 0:
                sleep_seconds += 3600  # 60분
            print(f"[{datetime.now()}] Sleeping for {sleep_seconds} seconds until next 60-minute mark.")
            time.sleep(sleep_seconds)

            # 포지션이 없을 때: 매수 신호 탐색
            if state['position'] == 0:
                top_coins = get_top_coins_by_volume(limit=10)
                buy_signals = []
                for ticker in top_coins:
                    data = get_latest_ohlcv(ticker, interval=interval, count=lookback + 2)
                    if data.empty:
                        continue
                    trade_signal = generate_trade_signal(data, lookback=lookback, tp_mult=tp_mult, sl_mult=sl_mult, atr_mult=atr_mult)
                    if trade_signal and trade_signal['Signal'] == 'Buy':
                        # 해당 코인의 거래대금 가져오기
                        ohlcv_day = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                        if not ohlcv_day.empty:
                            trade_volume = ohlcv_day['close'].iloc[-1] * ohlcv_day['volume'].iloc[-1]
                        else:
                            trade_volume = 0
                        buy_signals.append((ticker, trade_volume, trade_signal))
                if buy_signals:
                    # 거래대금 기준으로 내림차순 정렬 후 상위 코인 선택
                    buy_signals.sort(key=lambda x: x[1], reverse=True)
                    selected_ticker, _, selected_signal = buy_signals[0]
                    print(f"[{datetime.now()}] Buy signal detected for {selected_ticker}. Executing buy...")
                    state = execute_trade(
                        selected_signal,
                        state,
                        get_latest_ohlcv(selected_ticker, interval=interval, count=lookback + 2),
                        ticker=selected_ticker
                    )
                else:
                    print(f"[{datetime.now()}] No buy signals detected among top 10 coins.")
            else:
                # 포지션이 있을 때: 매도 신호 확인
                current_ticker = state['current_ticker']
                data = get_latest_ohlcv(current_ticker, interval=interval, count=lookback + 2)
                if data.empty:
                    print(f"[{datetime.now()}] No data fetched for {current_ticker}.")
                else:
                    trade_signal = generate_trade_signal(data, lookback=lookback, tp_mult=tp_mult, sl_mult=sl_mult, atr_mult=atr_mult)
                    if trade_signal and trade_signal['Signal'] == 'Sell':
                        print(f"[{datetime.now()}] Sell signal detected for {current_ticker}. Executing sell...")
                        state = execute_trade(
                            trade_signal,
                            state,
                            data,
                            ticker=current_ticker
                        )
                    else:
                        # 자동 매도 조건 확인 (hold period)
                        if state.get('entry_time'):
                            entry_time = datetime.fromisoformat(state['entry_time'])
                            candles_since_entry = (current_time - entry_time) // timedelta(minutes=60)
                            if candles_since_entry >= hold_candles:
                                print(f"[{datetime.now()}] Hold period exceeded for {current_ticker}. Executing automatic sell.")
                                # AutoSell 신호 생성
                                auto_sell_signal = {
                                    'Signal': 'AutoSell'
                                }
                                state = execute_trade(
                                    auto_sell_signal,
                                    state,
                                    data,
                                    ticker=current_ticker
                                )
            # 현재 잔고 로그
            krw_balance = upbit.get_balance("KRW")
            if krw_balance is None:
                krw_balance = 0.0
            current_ticker = state['current_ticker']
            if current_ticker:
                current_price = pyupbit.get_current_price(current_ticker)
                if current_price is None:
                    current_price = 0.0
                coin_symbol = current_ticker.split('-')[1]
                coin_balance = upbit.get_balance(coin_symbol)
                if coin_balance is None:
                    coin_balance = 0.0
                total_balance = krw_balance + coin_balance * current_price
                balance_message = (
                    f"Current Balances -> KRW: {krw_balance:,.0f} KRW, "
                    f"{coin_symbol}: {coin_balance:.6f} {coin_symbol}, "
                    f"Total: {total_balance:,.0f} KRW"
                )
            else:
                total_balance = krw_balance
                balance_message = f"Current Balances -> KRW: {krw_balance:,.0f} KRW, Total: {total_balance:,.0f} KRW"
            print(f"[{datetime.now()}] {balance_message}")
            send_telegram_message(balance_message)
        except Exception as e:
            logging.error(f"Exception in main loop: {e}")
            print(f"[{datetime.now()}] Exception in main loop: {e}")
            print("Sleeping for 1 minute before retrying...\n")
            time.sleep(60)  # 에러 발생 시 1분 대기 후 재시도

if __name__ == '__main__':
    main()

