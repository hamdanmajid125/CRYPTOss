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

load_dotenv()

@asynccontextmanager
async def lifespan(__: FastAPI):
    # Start all background tasks on boot — no manual scan button needed
    asyncio.create_task(bg_auto_scanner())       # 🔥 NEW: auto-scans top coins
    asyncio.create_task(bg_position_monitor())
    asyncio.create_task(bg_balance_broadcast())
    asyncio.create_task(bg_fng_update())
    asyncio.create_task(bg_top_coins_refresh())  # 🔥 NEW: refreshes coin list
    yield

from data_feed import DataFeed
from claude_agent import ClaudeAgent
from weex_client import WeexClient
from risk_manager import RiskManager, RiskSettings
from telegram_notifier import TelegramNotifier
from news_watcher import NewsWatcher
from trade_journal import TradeJournal
from pump_scanner import PumpScanner
import ccxt

app = FastAPI(title='AI Trade Terminal', version='3.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ── Config ─────────────────────────────────────────────────────────────────────
PAPER       = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
AUTO_TRADE  = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
ACCOUNT_USDT = float(os.getenv('ACCOUNT_USDT', '500'))

# How many top coins to scan (set 10 or 20 in .env)
TOP_N_COINS  = int(os.getenv('TOP_N_COINS', '20'))

# How often to run the auto-scan loop (default: every 5 minutes)
SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL_SECONDS', '300'))

public_exchange = ccxt.binance({'options': {'defaultType': 'spot'}, 'enableRateLimit': True})

feed   = DataFeed(public_exchange, cryptopanic_token=os.getenv('CRYPTOPANIC_TOKEN', ''))
claude = ClaudeAgent(api_key=os.getenv('ANTHROPIC_API_KEY', ''))
notifier = TelegramNotifier(
    token   = os.getenv('TELEGRAM_BOT_TOKEN', ''),
    chat_id = os.getenv('TELEGRAM_CHAT_ID', ''),
)
risk   = RiskManager(RiskSettings(
    account_usdt    = ACCOUNT_USDT,
    risk_pct        = float(os.getenv('RISK_PCT', '1.5')),
    max_trade_usdt  = float(os.getenv('MAX_TRADE_USDT', '75')),
    min_confidence  = int(os.getenv('MIN_CONFIDENCE', '75')),
    daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT_USDT', '50')),
    max_concurrent  = int(os.getenv('MAX_CONCURRENT_TRADES', '2')),
))
weex = WeexClient(
    api_key    = os.getenv('WEEX_API_KEY', ''),
    secret     = os.getenv('WEEX_SECRET', ''),
    passphrase = os.getenv('WEEX_PASSPHRASE', ''),
    paper      = PAPER,
)
news_watcher  = NewsWatcher(blackout_minutes=int(os.getenv('NEWS_BLACKOUT_MIN', '30')))
journal       = TradeJournal()
pump_scanner  = PumpScanner(public_exchange)

# ── WebSocket registry ─────────────────────────────────────────────────────────
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

async def bg_auto_scanner():
    """
    🔥 THE MAIN ENGINE — runs automatically every SCAN_INTERVAL seconds.
    No manual button needed. Fetches top coins by volume, analyzes each,
    sends signals to dashboard in real-time, and executes if AUTO_TRADE=true.
    """
    print(f'[AutoScanner] Starting — will scan top {TOP_N_COINS} coins every {SCAN_INTERVAL}s')

    # Small initial delay to let server boot fully
    await asyncio.sleep(15)

    while True:
        try:
            # Step 1: Get top coins — merge volume list with pump scanner hits
            top_coins   = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)
            pump_coins  = pump_scanner.get_pump_symbols()
            # Insert pump coins at front (deduplicated) so they get analysed first
            seen = set()
            merged = []
            for c in (pump_coins + top_coins):
                if c not in seen:
                    seen.add(c)
                    merged.append(c)
            top_coins = merged[:TOP_N_COINS + len(pump_coins)]
            if pump_coins:
                print(f'[PumpScanner] Prioritising {len(pump_coins)} pumping coins: {pump_coins[:5]}')

            print(f'[AutoScanner] Scanning {len(top_coins)} coins: {top_coins}')
            await broadcast({
                'type': 'scan_start',
                'coins': top_coins,
                'count': len(top_coins),
                'message': f'Auto-scanning top {len(top_coins)} coins by volume...'
            })

            signals_found = []

            for symbol in top_coins:
                try:
                    # Step 2: Get full technical analysis
                    data = feed.get_full_analysis(symbol, '1h')

                    # Step 3: Quick pre-filter — skip if TF is MIXED (saves Claude API calls)
                    tf_overall = data.get('timeframe_bias', {}).get('overall', 'MIXED')
                    vol_signal = data.get('volume_signal', 'NORMAL')

                    if tf_overall == 'MIXED':
                        print(f'[AutoScanner] {symbol} — SKIP (mixed TF bias)')
                        await broadcast({
                            'type':   'scan_skip',
                            'symbol': symbol,
                            'reason': 'Mixed timeframe bias — no clear direction'
                        })
                        await asyncio.sleep(0.5)  # small delay between symbols
                        continue

                    if 'VERY LOW' in vol_signal:
                        print(f'[AutoScanner] {symbol} — SKIP (very low volume)')
                        await broadcast({
                            'type':   'scan_skip',
                            'symbol': symbol,
                            'reason': 'Very low volume — not safe to trade'
                        })
                        await asyncio.sleep(0.5)
                        continue

                    # Step 4: Ask Claude AI for trading signal
                    print(f'[AutoScanner] {symbol} — Asking Claude (TF={tf_overall}, vol={vol_signal})')
                    signal = claude.get_signal(data)

                    if signal:
                        action = signal.get('action', 'WAIT')
                        conf   = signal.get('confidence', 0)

                        # Broadcast every signal (LONG, SHORT, or WAIT) to dashboard
                        await broadcast({
                            'type':   'signal',
                            'symbol': symbol,
                            'signal': signal
                        })

                        if action in ('LONG', 'SHORT'):
                            log_trade(f"SIGNAL | {action} | {symbol} | conf:{conf}% | {signal.get('setup_type','')}")
                            signals_found.append(signal)
                            notifier.send_signal(signal)

                            # Step 5: Execute if AUTO_TRADE is enabled
                            if AUTO_TRADE:
                                await maybe_execute(signal)
                        else:
                            print(f'[AutoScanner] {symbol} — WAIT (conf:{conf}%)')

                    # Delay between coins to avoid rate limiting
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f'[AutoScanner] Error on {symbol}: {e}')
                    await asyncio.sleep(1)
                    continue

            # Broadcast scan summary
            await broadcast({
                'type':          'scan_complete',
                'total_scanned': len(top_coins),
                'signals_found': len(signals_found),
                'next_scan_in':  SCAN_INTERVAL,
                'message':       f'Scan done. {len(signals_found)} signal(s) found. Next scan in {SCAN_INTERVAL//60}min.'
            })
            notifier.send_scan_summary(len(top_coins), len(signals_found), SCAN_INTERVAL)

            print(f'[AutoScanner] Done. {len(signals_found)} signals found. Sleeping {SCAN_INTERVAL}s...')

        except Exception as e:
            print(f'[AutoScanner] Critical error: {e}')
            await asyncio.sleep(60)  # Wait 1 min on error

        # Wait before next full scan cycle
        await asyncio.sleep(SCAN_INTERVAL)


async def bg_top_coins_refresh():
    """Refreshes the top coins list every 10 minutes independently."""
    while True:
        await asyncio.sleep(600)
        try:
            coins = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)
            await broadcast({'type': 'top_coins_updated', 'coins': coins})
        except Exception as e:
            print(f'[TopCoins] Error: {e}')


async def bg_position_monitor():
    """Every 60s: check paper positions for TP/SL hits."""
    while True:
        await asyncio.sleep(60)
        try:
            events = weex.check_positions()
            for ev in events:
                await broadcast({'type': 'position_event', **ev})
                log_trade(f"POS_EVENT | {ev.get('event')} | {ev.get('symbol')} | PnL: {ev.get('pnl', 0)}")
                notifier.send_position_event(ev)
        except Exception as e:
            print(f'[Monitor] Error: {e}')


async def bg_balance_broadcast():
    """Every 30s: broadcast live balance."""
    while True:
        await asyncio.sleep(30)
        try:
            bal   = weex.get_balance()
            stats = risk.get_stats()
            await broadcast({'type': 'balance', 'balance': bal, 'stats': stats})
        except Exception as e:
            print(f'[Balance] Error: {e}')


async def bg_fng_update():
    """Every 5min: refresh Fear & Greed."""
    while True:
        await asyncio.sleep(300)
        try:
            fng = feed.get_fear_and_greed()
            await broadcast({'type': 'sentiment', 'fear_greed': fng})
        except Exception as e:
            print(f'[F&G] Error: {e}')


# ── TRADE EXECUTION ────────────────────────────────────────────────────────────
async def maybe_execute(signal: Dict):
    # News blackout check — skip trades ±30 min around high-impact events
    blacked_out, blackout_reason = news_watcher.is_blackout()
    if blacked_out:
        print(f'[NewsWatcher] {blackout_reason}')
        await broadcast({'type': 'trade_skipped', 'reason': blackout_reason,
                         'symbol': signal.get('symbol')})
        return

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
    qty    = sizing.get('qty', 0)
    if qty <= 0:
        return

    sig_id = journal.log_signal(signal)
    try:
        result = weex.place_order(signal['symbol'], signal['action'],
                                  qty, entry, sl, tp1, tp2, tp3)
        if result:
            order_id = result.get('id', '')
            risk.record_open(order_id, signal)
            journal.log_trade_open(sig_id, order_id, signal['symbol'],
                                   signal['action'], qty, entry, sl, tp1, paper=PAPER)
            await broadcast({'type': 'trade_opened', 'order': result, 'signal': signal})
            log_trade(f"OPENED | {signal['action']} {qty} {signal['symbol']} @ {entry} | SL:{sl} TP1:{tp1} | id:{order_id}")
            notifier.send_trade_opened(signal['symbol'], signal['action'], qty, entry, sl, tp1)
    except Exception as e:
        log_trade(f"ERROR | {signal.get('symbol')} | {e}")
        print(f'[Trade] Error placing order: {e}')


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.get('/')
async def root():
    return {
        'message':      'AI Trade Terminal v3 — AUTO SCANNER ACTIVE',
        'auto_scan':    f'Scanning top {TOP_N_COINS} coins every {SCAN_INTERVAL}s',
        'paper':        PAPER,
        'auto_trade':   AUTO_TRADE,
        'docs':         '/docs',
    }

@app.get('/health')
async def health():
    return {
        'status':            'ok',
        'paper':             PAPER,
        'auto_trade':        AUTO_TRADE,
        'auto_scanner':      'RUNNING',
        'scan_interval_sec': SCAN_INTERVAL,
        'top_n_coins':       TOP_N_COINS,
        'connected_clients': len(clients),
        'paused':            risk.is_paused(),
    }

@app.get('/top-coins')
async def get_top_coins():
    """Get current top coins being scanned."""
    coins = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)
    return {'coins': coins, 'count': len(coins)}

@app.get('/stats')
async def get_stats():
    return risk.get_stats()

@app.get('/balance')
async def get_balance():
    bal   = weex.get_balance()
    stats = risk.get_stats()
    return {'balance': bal, 'stats': stats}

@app.get('/sentiment')
async def get_sentiment():
    fng  = feed.get_fear_and_greed()
    mood = ('Cautious — contrarian LONG opportunity' if fng['value'] < 25 else
            'Greedy — tighten SL on longs'          if fng['value'] > 75 else 'Neutral')
    return {'fear_greed': fng, 'mood': mood}

@app.get('/positions')
async def get_positions():
    return weex.get_positions()

@app.post('/scan')
async def scan_now(background_tasks: BackgroundTasks):
    """Trigger an immediate scan cycle (called by dashboard Scan All button)."""
    async def _run():
        top_coins = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)
        await broadcast({'type': 'scan_start', 'coins': top_coins, 'count': len(top_coins),
                         'message': f'Manual scan — {len(top_coins)} coins'})
        signals_found = []
        for symbol in top_coins:
            try:
                data = feed.get_full_analysis(symbol, '1h')
                tf_overall = data.get('timeframe_bias', {}).get('overall', 'MIXED')
                vol_signal = data.get('volume_signal', 'NORMAL')
                if tf_overall == 'MIXED' or 'VERY LOW' in vol_signal:
                    await broadcast({'type': 'scan_skip', 'symbol': symbol,
                                     'reason': 'Mixed TF or very low volume'})
                    await asyncio.sleep(0.3)
                    continue
                signal = claude.get_signal(data)
                if signal:
                    await broadcast({'type': 'signal', 'symbol': symbol, 'signal': signal})
                    if signal.get('action') in ('LONG', 'SHORT'):
                        signals_found.append(signal)
                        notifier.send_signal(signal)
                        if AUTO_TRADE:
                            await maybe_execute(signal)
                await asyncio.sleep(1)
            except Exception as e:
                print(f'[ManualScan] {symbol}: {e}')
        await broadcast({'type': 'scan_complete', 'total_scanned': len(top_coins),
                         'signals_found': len(signals_found), 'next_scan_in': SCAN_INTERVAL,
                         'message': f'Scan done — {len(signals_found)} signal(s) found.'})
    background_tasks.add_task(_run)
    return {'status': 'scan_started', 'coins': TOP_N_COINS}


@app.post('/analyze/{symbol:path}')
async def analyze_manual(symbol: str, background_tasks: BackgroundTasks):
    """Manual single-coin analysis (for testing specific coins)."""
    try:
        ex_symbol = symbol.replace('-', '/').upper()
        data      = feed.get_full_analysis(ex_symbol, '1h')
        signal    = claude.get_signal(data)
        if signal:
            background_tasks.add_task(broadcast, {'type': 'signal', 'symbol': ex_symbol, 'signal': signal})
            if AUTO_TRADE:
                background_tasks.add_task(maybe_execute, signal)
        return signal or {'action': 'WAIT', 'reason': 'Analysis returned no signal'}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/settings')
async def update_settings(body: Dict):
    global AUTO_TRADE, SCAN_INTERVAL, TOP_N_COINS
    if 'auto_trade'   in body: AUTO_TRADE    = bool(body['auto_trade'])
    if 'scan_interval' in body: SCAN_INTERVAL = int(body['scan_interval'])
    if 'top_n_coins'  in body: TOP_N_COINS   = int(body['top_n_coins'])
    risk.settings.risk_pct        = body.get('risk_pct',        risk.settings.risk_pct)
    risk.settings.min_confidence  = body.get('min_confidence',  risk.settings.min_confidence)
    risk.settings.max_concurrent  = body.get('max_trades',      risk.settings.max_concurrent)
    risk.settings.daily_loss_limit = body.get('daily_loss_limit', risk.settings.daily_loss_limit)
    return {'ok': True, 'auto_trade': AUTO_TRADE, 'scan_interval': SCAN_INTERVAL,
            'top_n_coins': TOP_N_COINS, 'settings': risk.settings.__dict__}

@app.get('/pumps')
async def get_pumps():
    """Returns current pump/dump movers — coins with unusual volume+price spikes."""
    return {'pumps': pump_scanner.scan(), 'count': len(pump_scanner._cache['results'])}

@app.get('/journal/signals')
async def journal_signals(limit: int = 50):
    return journal.get_recent_signals(limit=limit)

@app.get('/journal/trades')
async def journal_trades():
    return journal.get_open_trades()

@app.get('/journal/stats')
async def journal_stats():
    return {
        'win_rate':   journal.get_win_rate(),
        'daily':      journal.get_daily_stats(days=7),
        'best_setups': journal.get_best_setups(),
    }

@app.get('/news/events')
async def news_events():
    blacked_out, reason = news_watcher.is_blackout()
    return {
        'blackout':    blacked_out,
        'reason':      reason,
        'next_event':  news_watcher.next_event(),
        'all_events':  news_watcher.get_events(),
    }

@app.post('/emergency/stop')
async def emergency_stop():
    try:
        closed = weex.close_all()
        risk.open_trades = []
        await broadcast({'type': 'emergency_stop', 'closed': closed})
        log_trade(f'EMERGENCY_STOP | closed {len(closed)} positions')
        return {'status': 'emergency_stop_executed', 'closed_count': len(closed)}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/trade/execute')
async def manual_trade(body: Dict):
    signal  = body.get('signal', {})
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

@app.post('/webhook/tradingview')
async def tradingview_webhook(payload: Dict):
    symbol = payload.get('symbol', '').replace('USDT', '/USDT').upper()
    action = payload.get('action', '').upper()
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

    # Send welcome state
    bal   = weex.get_balance()
    fng   = feed.get_fear_and_greed()
    stats = risk.get_stats()
    coins = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)

    await websocket.send_json({'type': 'weex_connected', 'paper': PAPER})
    await websocket.send_json({'type': 'balance', 'balance': bal, 'stats': stats})
    await websocket.send_json({'type': 'sentiment', 'fear_greed': fng})
    await websocket.send_json({'type': 'top_coins', 'coins': coins})
    await websocket.send_json({
        'type':    'scanner_info',
        'message': f'Auto-scanning top {TOP_N_COINS} coins every {SCAN_INTERVAL//60} minutes. No manual action needed.'
    })

    try:
        while True:
            # Stream live prices every 3s for top 4 coins
            top4 = coins[:4]
            for market in top4:
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
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
        print(f'[WS] Client disconnected ({len(clients)} total)')


if __name__ == '__main__':
    print('=' * 60)
    print('  AI TRADE TERMINAL v3 — AUTO SCANNER')
    print(f'  Paper Mode   : {PAPER}')
    print(f'  Auto Trade   : {AUTO_TRADE}')
    print(f'  Account USDT : ${ACCOUNT_USDT:.2f}')
    print(f'  Top N Coins  : {TOP_N_COINS}')
    print(f'  Scan Every   : {SCAN_INTERVAL}s ({SCAN_INTERVAL//60} min)')
    print(f'  Min Confidence: 72%')
    print('  Listening    : http://127.0.0.1:8000')
    print('  Dashboard    : open trading_dashboard.html')
    print('  ✅ NO MANUAL SCAN BUTTON NEEDED — fully automatic')
    print('=' * 60)
    uvicorn.run(app, host=os.getenv('HOST', '127.0.0.1'),
                port=int(os.getenv('PORT', '8000')))