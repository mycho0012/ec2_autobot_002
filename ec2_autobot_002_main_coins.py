import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pyupbit
import requests
from dotenv import load_dotenv
from notion_client import Client
import uuid

# =====================
# 환경 변수 로드
# =====================
load_dotenv()

# Slack 설정
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')

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
        "last_trade_id": ""           # 마지막 거래 ID
    }

def send_slack_message(message: str):
    """Slack을 통해 메시지를 전송합니다."""
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'text': message}
        response = requests.post(SLACK_WEBHOOK_URL, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            logging.error(f"Failed to send Slack message: {response.text}")
    except Exception as e:
        logging.error(f"Exception in send_slack_message: {e}")

def log_to_notion(trade: dict):
    """Notion 데이터베이스에 거래 내역을 기록합니다."""
    try:
        properties = {
            "Trade ID": {
                "title": [
                    {
                        "text": {
                            "content": str(trade['Trade ID'])
                        }
                    }
                ]
            },
            "Ticker": {
                "rich_text": [
                    {
                        "text": {
                            "content": trade['Ticker']
                        }
                    }
                ]
            },
            "Type": {
                "select": {
                    "name": trade['Type']
                }
            },
            "Timestamp": {
                "date": {
                    "start": trade['Timestamp']
                }
            },
            "Price": {
                "number": trade['Price']
            },
            "Stop Loss": {
                "number": trade.get('Stop Loss', None)
            },
            "Take Profit": {
                "number": trade.get('Take Profit', None)
            },
            "Quantity": {
                "number": trade.get('Quantity', None)
            },
            "Status": {
                "select": {
                    "name": trade.get('Status', 'Open')
                }
            },
            "Sell Timestamp": {
                "date": {
                    "start": trade.get('Sell Timestamp', None)
                }
            },
            "Sell Price": {
                "number": trade.get('Sell Price', None)
            },
            "Buy Trade ID": {
                "rich_text": [
                    {
                        "text": {
                            "content": str(trade.get('Buy Trade ID', ''))
                        }
                    }
                ]
            }
        }
        notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties=properties
        )
        logging.info(f"Trade logged to Notion: {trade['Trade ID']}")
    except Exception as e:
        logging.error(f"Exception in log_to_notion: {e}")

# =====================
# ATR 계산 함수
# =====================

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """평균 실제 범위(ATR)를 계산합니다."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

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

    # ATR 계산 (compute_atr 함수 사용)
    atr = compute_atr(window['high'], window['low'], window['close'], length=14)
    latest_atr = atr.iloc[-1]

    # 저항선과 지지선의 기울기 비교 조건 추가
    resist_slope = r_coefs[0]
    support_slope = s_coefs[0]
    slope_condition = False

    # 상승 추세일 때
    if resist_slope > 0 and support_slope > 0:
        if support_slope > resist_slope:
            slope_condition = True
    # 하락 추세일 때
    elif resist_slope < 0 and support_slope < 0:
        if support_slope > resist_slope:
            slope_condition = True

    signal = None
    if latest_candle['close'] > r_val and slope_condition:
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
# 매수/매도 거래 조회 함수
# =====================

def get_last_open_buy_trade(ticker: str):
    """
    특정 티커에 대한 가장 최근 열린 매수 거래 정보를 Notion에서 조회합니다.
    """
    try:
        response = notion.databases.query(
            **{
                "database_id": NOTION_DATABASE_ID,
                "filter": {
                    "and": [
                        {
                            "property": "Ticker",
                            "rich_text": {
                                "equals": ticker
                            }
                        },
                        {
                            "property": "Type",
                            "select": {
                                "equals": "Buy"
                            }
                        },
                        {
                            "property": "Status",
                            "select": {
                                "equals": "Open"
                            }
                        }
                    ]
                },
                "sorts": [
                    {
                        "property": "Timestamp",
                        "direction": "descending"
                    }
                ],
                "page_size": 1
            }
        )
        logging.info(f"get_last_open_buy_trade response for {ticker}: {response}")
        if response['results']:
            trade = response['results'][0]
            trade_id = trade['properties']['Trade ID']['title'][0]['text']['content']
            page_id = trade['id']
            logging.info(f"Found trade: Trade ID {trade_id}, Page ID {page_id}")
            return {
                'Trade ID': trade_id,
                'Page ID': page_id
            }
        else:
            logging.info(f"No open trade found for {ticker}.")
            return None
    except Exception as e:
        logging.error(f"Exception in get_last_open_buy_trade: {e}")
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
    trade_id = str(uuid.uuid4())  # 고유한 Trade ID 생성
    signal_type = signal['Signal']
    entry_price = signal['Entry Price']
    tp_price = signal.get('TP', None)
    sl_price = signal.get('SL', None)
    resistance = signal.get('Resistance', None)
    support = signal.get('Support', None)
    current_time = ohlcv.index[-1]
    volume = ohlcv['volume'].iloc[-1]
    # ATR 계산 (compute_atr 함수 사용)
    atr = compute_atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=14).iloc[-1]

    # Ticker 기본값 설정
    if ticker is None:
        ticker = "KRW-BTC"

    if signal_type == 'Buy':
        # Notion에서 열린 포지션 확인
        open_trade = get_last_open_buy_trade(ticker)
        if open_trade:
            logging.info(f"Buy signal received but an open position exists for {ticker}: Trade ID {open_trade['Trade ID']}")
            send_slack_message(f"매수 신호가 감지되었으나 이미 열린 포지션이 존재합니다: Trade ID {open_trade['Trade ID']}, Ticker: {ticker}")
            return state

        krw_balance = upbit.get_balance("KRW")
        if krw_balance is None:
            krw_balance = 0.0
        trade_amount = krw_balance * 0.3  # KRW 잔고의 30% 사용
        if trade_amount < 5000:
            logging.info(f"Trade ID {trade_id} skipped due to insufficient KRW balance.")
            send_slack_message(f"[{datetime.now()}] Trade ID {trade_id} skipped due to insufficient KRW balance.")
            return state
        try:
            # 시장가 매수
            order = upbit.buy_market_order(ticker, trade_amount)
            logging.info(f"Trade ID {trade_id} Buy Order: {order}")
            send_slack_message(f"[{datetime.now()}] Trade ID {trade_id}: Buy order placed for {trade_amount:,.0f} KRW of {ticker}")
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

            # Notion 로그 - Buy
            trade_info = {
                'Trade ID': trade_id,
                'Ticker': ticker,
                'Type': 'Buy',
                'Timestamp': state['entry_time'],
                'Price': current_price,
                'Stop Loss': sl_price,
                'Take Profit': tp_price,
                'Quantity': qty_bought,
                'Status': 'Open'
            }
            log_to_notion(trade_info)

            # Slack 메시지
            message = (
                f"Buy Order Executed\n"
                f"Trade ID: {trade_id}\n"
                f"Ticker: {ticker}\n"
                f"Buy Price: {current_price:,.0f} KRW\n"
                f"Amount: {trade_amount:,.0f} KRW\n"
                f"Qty: {qty_bought:.6f} {ticker.split('-')[1]}"
            )
            send_slack_message(message)
        except Exception as e:
            logging.error(f"Exception during buy order for Trade ID {trade_id}: {e}")
            send_slack_message(f"[{datetime.now()}] Exception during buy order for Trade ID {trade_id}: {e}")

    elif signal_type in ['Sell', 'AutoSell']:
        if ticker is None:
            logging.error("Ticker is None during Sell/AutoSell execution.")
            return state

        open_trade = get_last_open_buy_trade(ticker)
        if not open_trade:
            logging.info(f"Sell signal received but no open position exists for {ticker}.")
            send_slack_message(f"매도 신호가 감지되었으나 열린 포지션이 존재하지 않습니다: {ticker}")
            return state

        status = "Open"  # 이미 get_last_open_buy_trade 함수에서 확인했으므로 'Open'으로 고정
        if status != 'Open':
            logging.info(f"Sell signal received but the last trade for {ticker} is not open: Status {status}")
            send_slack_message(f"매도 신호가 감지되었으나 마지막 거래가 열린 상태가 아닙니다: Trade ID {open_trade['Trade ID']}, Ticker: {ticker}")
            return state

        try:
            buy_trade_id = open_trade['Trade ID']
            page_id = open_trade['Page ID']
            qty = state.get('qty', 0.0)
            if qty <= 0.0:
                raise Exception(f"Invalid quantity {qty} for Sell trade.")

            # 시장가 매도
            order = upbit.sell_market_order(ticker, qty)
            logging.info(f"Sell Order: {order}")
            send_slack_message(f"[{datetime.now()}] Trade ID {trade_id}: Sell order placed for {qty:.6f} {ticker}")

            time.sleep(1)  # API 호출 간 대기

            # 실제 매도 가격 확인
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                raise Exception("Failed to get current price after sell order.")
            trade_amount = qty * current_price
            trade_cost = qty * state.get('entry_price', 0.0)
            profit_loss = trade_amount - trade_cost
            return_pct = (profit_loss / trade_cost) * 100 if trade_cost != 0 else 0.0

            # 상태 업데이트
            trade_id_sell = str(uuid.uuid4())
            state['position'] = 0
            state['current_ticker'] = None
            state['qty'] = 0.0
            state['last_trade_id'] = trade_id_sell
            state['last_processed_time'] = current_time.isoformat()
            state['entry_price'] = 0.0  # 진입 가격 초기화
            state['entry_time'] = None  # 진입 시간 초기화
            save_state(state)

            # Notion 로그 - Sell
            trade_info = {
                'Trade ID': trade_id_sell,
                'Ticker': ticker,
                'Type': 'Sell',
                'Timestamp': current_time.isoformat(),
                'Price': current_price,
                'Status': 'Closed',
                'Sell Timestamp': current_time.isoformat(),
                'Sell Price': current_price,
                'Buy Trade ID': buy_trade_id
            }
            log_to_notion(trade_info)

            # 매수 거래의 Status를 'Closed'로 업데이트
            try:
                notion.pages.update(
                    page_id=page_id,
                    properties={
                        "Status": {
                            "select": {
                                "name": "Closed"
                            }
                        }
                    }
                )
                logging.info(f"Updated Buy Trade ID {buy_trade_id} Status to Closed.")
                send_slack_message(f"✅ Buy Trade ID {buy_trade_id} Status updated to Closed.")
                print(f"✅ Buy Trade ID {buy_trade_id} Status updated to Closed.")
            except Exception as e:
                logging.error(f"Exception in updating Buy trade status: {e}")
                send_slack_message(f"❌ 오류 발생: Buy Trade ID {buy_trade_id} 상태 업데이트 실패.")
                print(f"❌ 오류 발생: Buy Trade ID {buy_trade_id} 상태 업데이트 실패.")

            # Slack 메시지
            reason = reason_override if reason_override else ('Hold Period Exceeded' if signal_type == 'AutoSell' else 'Price crossed below Support')
            message = (
                f"Sell Order Executed\n"
                f"Trade ID: {trade_id_sell}\n"
                f"Ticker: {ticker}\n"
                f"Sell Price: {current_price:,.0f} KRW\n"
                f"Qty: {qty:.6f} {ticker.split('-')[1]}\n"
                f"Profit/Loss: {profit_loss:,.0f} KRW\n"
                f"Return: {return_pct:.2f}%\n"
                f"Reason: {reason}"
            )
            send_slack_message(message)
        except Exception as e:
            logging.error(f"Exception during sell order for Trade ID {buy_trade_id}: {e}")
            send_slack_message(f"[{datetime.now()}] Exception during sell order for Trade ID {buy_trade_id}: {e}")
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
    interval = "minute30"       # Upbit API에서 지원하는 인터벌
    lookback = 50                # 추세선을 계산할 기간
    tp_mult = 3.0                # 테이크 프로핏 배수
    sl_mult = 3.0                # 스톱 로스 배수
    atr_mult = 1.0               # ATR 배수 (현재 사용되지 않음, 필요 시 조정)
    hold_candles = 24            # 포지션을 유지할 최대 캔들 수 (30분 기준)

    # 봇 시작 메시지 전송
    initial_balance = upbit.get_balance("KRW")
    if initial_balance is None:
        initial_balance = 0.0
    state['initial_balance'] = state.get('initial_balance', initial_balance)
    save_state(state)
    print(f"[{datetime.now()}] Autotrading Bot started. Current KRW Balance: {initial_balance:,.0f} KRW")
    send_slack_message(f"Autotrading Bot started successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\nCurrent KRW Balance: {initial_balance:,.0f} KRW")

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
            # KST 타임존 인식으로 변경
            current_time = datetime.now(timezone.utc) + timedelta(hours=9)  # KST 시간으로 변환
            # 다음 30분 마크까지 대기
            minute = current_time.minute
            second = current_time.second
            sleep_minutes = 30 - (minute % 30)
            sleep_seconds = sleep_minutes * 60 - second
            if sleep_seconds <= 0:
                sleep_seconds += 1800  # 30분
            print(f"[{datetime.now()}] Sleeping for {sleep_seconds} seconds until next 30-minute mark.")
            time.sleep(sleep_seconds)

            if state.get('position', 0) == 1 and state.get('current_ticker'):
                # 포지션이 열려 있는 경우: 해당 티커에 대한 Sell 시그널만 처리
                ticker = state['current_ticker']
                data = get_latest_ohlcv(ticker, interval=interval, count=lookback + 2)
                if data.empty:
                    continue
                trade_signal = generate_trade_signal(data, lookback=lookback, tp_mult=tp_mult, sl_mult=sl_mult, atr_mult=atr_mult)
                if trade_signal and trade_signal['Signal'] in ['Sell', 'AutoSell']:
                    print(f"[{datetime.now()}] Sell signal detected for {ticker}. Executing sell...")
                    state = execute_trade(
                        trade_signal,
                        state,
                        data,
                        ticker=ticker
                    )
                else:
                    print(f"[{datetime.now()}] No sell signal detected for {ticker}.")
                
                # 자동 매도 조건 확인 (hold period)
                if state.get('position', 0) == 1 and state.get('entry_time'):
                    entry_time = datetime.fromisoformat(state['entry_time']).replace(tzinfo=timezone.utc) + timedelta(hours=9)  # KST
                    candles_since_entry = (current_time - entry_time) // timedelta(minutes=30)
                    if candles_since_entry >= hold_candles:
                        print(f"[{datetime.now()}] Hold period exceeded for {ticker}. Executing automatic sell.")
                        # AutoSell 신호 생성
                        auto_sell_signal = {
                            'Signal': 'AutoSell'
                        }
                        state = execute_trade(
                            auto_sell_signal,
                            state,
                            data,
                            ticker=ticker,
                            reason_override='Hold Period Exceeded'
                        )

            else:
                # 포지션이 없는 경우: 상위 10개 코인 중 매수 시그널 탐색
                top_coins = get_top_coins_by_volume(limit=10)
                for ticker in top_coins:
                    data = get_latest_ohlcv(ticker, interval=interval, count=lookback + 2)
                    if data.empty:
                        continue
                    trade_signal = generate_trade_signal(data, lookback=lookback, tp_mult=tp_mult, sl_mult=sl_mult, atr_mult=atr_mult)
                    if trade_signal and trade_signal['Signal'] == 'Buy':
                        print(f"[{datetime.now()}] Buy signal detected for {ticker}. Executing buy...")
                        state = execute_trade(
                            trade_signal,
                            state,
                            data,
                            ticker=ticker
                        )
                        # 포지션이 생겼으므로 다른 코인에 대한 매수 시도 중지
                        break
                    elif trade_signal and trade_signal['Signal'] == 'Sell':
                        # 포지션이 없는데 Sell 시그널이 발생하면 무시
                        logging.info(f"Sell signal detected for {ticker} but no open position exists.")
                        send_slack_message(f"매도 신호가 감지되었으나 열린 포지션이 존재하지 않습니다: {ticker}")

            # 현재 잔고 로그
            krw_balance = upbit.get_balance("KRW")
            if krw_balance is None:
                krw_balance = 0.0
            current_ticker = state.get('current_ticker')
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
                    f"현재 잔고 -> KRW: {krw_balance:,.0f} KRW, "
                    f"{coin_symbol}: {coin_balance:.6f} {coin_symbol}, "
                    f"총합: {total_balance:,.0f} KRW"
                )
            else:
                total_balance = krw_balance
                balance_message = f"현재 잔고 -> KRW: {krw_balance:,.0f} KRW, 총합: {total_balance:,.0f} KRW"
            print(f"[{datetime.now()}] {balance_message}")
            send_slack_message(balance_message)
        except Exception as e:
            logging.error(f"Exception in main loop: {e}")
            send_slack_message(f"[{datetime.now()}] Exception in main loop: {e}")
            print(f"[{datetime.now()}] Exception in main loop: {e}")
            print("Sleeping for 1 minute before retrying...\n")
            time.sleep(60)  # 에러 발생 시 1분 대기 후 재시도

if __name__ == '__main__':
    main()


