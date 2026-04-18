import os
import asyncio
import time
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(__: FastAPI):
    asyncio.create_task(bg_position_monitor())
    asyncio.create_task(bg_balance_broadcast())
    asyncio.create_task(bg_fng_update())
    yield

load_dotenv()

from data_feed    import DataFeed
from claude_agent import ClaudeAgent
from weex_client  import WeexClient
from risk_manager import RiskManager, RiskSettings
import ccxt

app = FastAPI(title='AI Trade Terminal', version='2.0.0', lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ── Initialize services ────────────────────────────────────────────────────────
PAPER       = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
AUTO_TRADE  = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
ACCOUNT_USDT = float(os.getenv('ACCOUNT_USDT', '10000'))
SYMBOLS     = os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT').split(',')

public_exchange = ccxt.binance({'options': {'defaultType': 'spot'}, 'enableRateLimit': True})

feed   = DataFeed(public_exchange)
claude = ClaudeAgent(api_key=os.getenv('ANTHROPIC_API_KEY', ''))
risk   = RiskManager(RiskSettings(
    account_usdt=ACCOUNT_USDT,
    risk_pct=float(os.getenv('RISK_PCT', '1.5')),
    max_trade_usdt=float(os.getenv('MAX_TRADE_USDT', '150')),
    min_confidence=int(os.getenv('MIN_CONFIDENCE', '68')),
    daily_loss_limit=float(os.getenv('DAILY_LOSS_LIMIT_USDT', '250')),
    max_concurrent=int(os.getenv('MAX_CONCURRENT_TRADES', '3')),
))
weex = WeexClient(
    api_key=os.getenv('WEEX_API_KEY', ''),
    secret=os.getenv('WEEX_SECRET', ''),
    passphrase=os.getenv('WEEX_PASSPHRASE', ''),
    paper=PAPER,
)

# ── WebSocket client registry ──────────────────────────────────────────────────
clients: List[WebSocket] = []

async def broadcast(msg: Dict):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        if d in clients:
            clients.remove(d)

def log_trade(line: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open('trades.log', 'a') as f:
        f.write(f'{ts} | {line}\n')

# ── BACKGROUND TASKS ──────────────────────────────────────────────────────────

async def bg_position_monitor():
    """Every 60s: check paper positions for TP/SL hits, move stops."""
    while True:
        await asyncio.sleep(60)
        try:
            events = weex.check_positions()
            for ev in events:
                await broadcast({'type': 'position_event', **ev})
                log_trade(f"POS_EVENT | {ev.get('event')} | {ev.get('symbol')} | PnL: {ev.get('pnl', 0)}")
        except Exception as e:
            print(f'[Monitor] Error: {e}')

async def bg_balance_broadcast():
    """Every 30s: broadcast live balance to dashboard."""
    while True:
        await asyncio.sleep(30)
        try:
            bal = weex.get_balance()
            stats = risk.get_stats()
            await broadcast({'type': 'balance', 'balance': bal, 'stats': stats})
        except Exception as e:
            print(f'[Balance] Error: {e}')

async def bg_fng_update():
    """Every 5min: refresh Fear & Greed cache."""
    while True:
        await asyncio.sleep(300)
        try:
            fng = feed.get_fear_and_greed()
            await broadcast({'type': 'sentiment', 'fear_greed': fng})
        except Exception as e:
            print(f'[F&G] Error: {e}')

# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get('/')
async def root():
    return {'message': 'AI Trade Terminal v2 is RUNNING', 'docs': '/docs', 'health': '/health'}

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'paper': PAPER,
        'auto_trade': AUTO_TRADE,
        'connected_clients': len(clients),
        'weex': 'paper' if PAPER else 'live',
        'paused': risk.is_paused(),
    }

@app.get('/stats')
async def get_stats():
    return risk.get_stats()

@app.get('/balance')
async def get_balance():
    bal = weex.get_balance()
    stats = risk.get_stats()
    return {'balance': bal, 'stats': stats}

@app.get('/sentiment')
async def get_sentiment():
    fng = feed.get_fear_and_greed()
    mood = 'Cautious (contrarian LONG opportunity)' if fng['value'] < 25 else \
           'Greedy (tighten SL on longs)' if fng['value'] > 75 else 'Neutral'
    return {'fear_greed': fng, 'mood': mood}

@app.get('/positions')
async def get_positions():
    return weex.get_positions()

@app.post('/analyze/{symbol:path}')
async def analyze(symbol: str, background_tasks: BackgroundTasks):
    try:
        ex_symbol = symbol.replace('-', '/').upper()
        data = feed.get_full_analysis(ex_symbol, '1h')
        signal = claude.get_signal(data)

        if signal:
            background_tasks.add_task(broadcast, {'type': 'signal', 'symbol': ex_symbol, 'signal': signal})
            if AUTO_TRADE:
                background_tasks.add_task(maybe_execute, signal)

        return signal or {'action': 'WAIT', 'reason': 'Analysis returned no signal'}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/scan')
async def scan_all(background_tasks: BackgroundTasks):
    for sym in SYMBOLS:
        background_tasks.add_task(_analyze_market, sym.strip())
    return {'status': 'scanning', 'markets': SYMBOLS}

async def _analyze_market(symbol: str):
    try:
        data = feed.get_full_analysis(symbol, '1h')
        signal = claude.get_signal(data)
        if signal:
            await broadcast({'type': 'signal', 'symbol': symbol, 'signal': signal})
            if AUTO_TRADE:
                await maybe_execute(signal)
    except Exception as e:
        print(f'[Scan] Error on {symbol}: {e}')

async def maybe_execute(signal: Dict):
    can, reason = risk.can_trade(signal)
    log_trade(f"ATTEMPT | {signal.get('symbol')} | {signal.get('action')} | conf:{signal.get('confidence')} | {reason}")

    if not can:
        print(f'[Risk] Skip: {reason}')
        await broadcast({'type': 'trade_skipped', 'reason': reason, 'symbol': signal.get('symbol')})
        return

    entry = signal.get('entry', 0)
    sl    = signal.get('sl', 0)
    tp1   = signal.get('tp1', 0)
    tp2   = signal.get('tp2', 0)
    tp3   = signal.get('tp3', 0)
    conf  = signal.get('confidence', 70)

    if not all([entry, sl, tp1]):
        return

    sizing = risk.position_size(entry, sl, confidence=conf)
    qty = sizing.get('qty', 0)
    if qty <= 0:
        return

    try:
        result = weex.place_order(signal['symbol'], signal['action'], qty, entry, sl, tp1, tp2, tp3)
        if result:
            risk.record_open(result.get('id', ''), signal)
            await broadcast({'type': 'trade_opened', 'order': result, 'signal': signal})
            log_trade(f"OPENED | {signal['action']} {qty} {signal['symbol']} @ {entry} | SL:{sl} TP1:{tp1} | id:{result.get('id')}")
    except Exception as e:
        log_trade(f"ERROR | {signal.get('symbol')} | {e}")
        print(f'[Trade] Error placing order: {e}')

@app.post('/trade/execute')
async def manual_trade(body: Dict):
    signal = body.get('signal', {})
    can, reason = risk.can_trade(signal)
    if not can:
        return {'executed': False, 'reason': reason}
    entry  = signal.get('entry')
    sl     = signal.get('sl')
    tp1    = signal.get('tp1')
    tp2    = signal.get('tp2', 0)
    tp3    = signal.get('tp3', 0)
    conf   = signal.get('confidence', 70)
    sizing = risk.position_size(entry, sl, confidence=conf)
    result = weex.place_order(signal.get('symbol'), signal.get('action'),
                              sizing['qty'], entry, sl, tp1, tp2, tp3)
    if result:
        risk.record_open(result.get('id', ''), signal)
    return {'executed': bool(result), 'order': result, 'sizing': sizing}

@app.post('/settings')
async def update_settings(body: Dict):
    global AUTO_TRADE
    if 'auto_trade' in body:
        AUTO_TRADE = bool(body['auto_trade'])
    risk.settings.risk_pct         = body.get('risk_pct', risk.settings.risk_pct)
    risk.settings.min_confidence   = body.get('min_confidence', risk.settings.min_confidence)
    risk.settings.max_concurrent   = body.get('max_trades', risk.settings.max_concurrent)
    risk.settings.daily_loss_limit = body.get('daily_loss_limit', risk.settings.daily_loss_limit)
    return {'ok': True, 'auto_trade': AUTO_TRADE, 'settings': risk.settings.__dict__}

@app.post('/emergency/stop')
async def emergency_stop():
    try:
        closed = weex.close_all()
        risk.open_trades = []
        await broadcast({'type': 'emergency_stop', 'closed': closed})
        log_trade(f'EMERGENCY_STOP | closed {len(closed)} positions')
        return {'status': 'emergency_stop_executed', 'closed_count': len(closed), 'details': closed}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/webhook/tradingview')
async def tradingview_webhook(payload: Dict):
    symbol = payload.get('symbol', '').replace('USDT', '/USDT').upper()
    action = payload.get('action', '').upper()
    print(f'[TradingView] {symbol} {action} @ {payload.get("price")}')
    try:
        data   = feed.get_full_analysis(symbol, '1h')
        signal = claude.get_signal(data)
        if signal:
            if signal['action'] == action or signal.get('confidence', 0) >= 75:
                await broadcast({'type': 'tv_alert', 'payload': payload, 'ai_signal': signal})
                if AUTO_TRADE:
                    await maybe_execute(signal)
    except Exception as e:
        print(f'[Webhook] Error: {e}')
    return {'received': True}

# ── WEBSOCKET ──────────────────────────────────────────────────────────────────

@app.websocket('/ws/signals')
async def ws_signals(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print(f'[WS] Client connected ({len(clients)} total)')

    # Send initial state
    bal  = weex.get_balance()
    fng  = feed.get_fear_and_greed()
    stats = risk.get_stats()
    await websocket.send_json({'type': 'weex_connected', 'paper': PAPER})
    await websocket.send_json({'type': 'balance', 'balance': bal, 'stats': stats})
    await websocket.send_json({'type': 'sentiment', 'fear_greed': fng})

    try:
        while True:
            # Stream live prices every 2s
            for market in SYMBOLS[:4]:
                try:
                    ticker = weex.get_ticker(market.strip())
                    await websocket.send_json({
                        'type':   'price',
                        'symbol': market.strip(),
                        'price':  ticker.get('last', 0),
                        'change': ticker.get('percentage', 0),
                    })
                except Exception:
                    pass
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
        print(f'[WS] Client disconnected ({len(clients)} total)')

if __name__ == '__main__':
    print('=' * 52)
    print('  AI TRADE TERMINAL v2 -- Python Backend')
    print(f'  Paper Mode : {PAPER}')
    print(f'  Auto Trade : {AUTO_TRADE}')
    print(f'  Account    : ${ACCOUNT_USDT:.2f}')
    print('  Listening  : http://127.0.0.1:8000')
    print('  Dashboard  : open trading_dashboard.html')
    print('=' * 52)
    uvicorn.run(app, host=os.getenv('HOST', '127.0.0.1'),
                port=int(os.getenv('PORT', '8000')))
