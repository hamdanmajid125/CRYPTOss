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
    await sync_balance_to_risk()   # seed risk manager with live wallet before anything else
    asyncio.create_task(bg_auto_scanner())
    asyncio.create_task(bg_balance_broadcast())
    asyncio.create_task(bg_fng_update())
    asyncio.create_task(bg_top_coins_refresh())
    asyncio.create_task(trade_manager.monitor_loop(broadcast))
    yield

from data_feed import DataFeed
from claude_agent import ClaudeAgent
from signal_logic import generate_signal as _rule_signal
from weex_client import WeexClient
from risk_manager import RiskManager, RiskSettings
from telegram_notifier import TelegramNotifier
from news_watcher import NewsWatcher
from trade_journal import TradeJournal
from pump_scanner import PumpScanner
from trade_manager import TradeManager
import ccxt

app = FastAPI(title='AI Trade Terminal', version='3.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ── Config ─────────────────────────────────────────────────────────────────────
PAPER       = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
AUTO_TRADE  = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
# Account balance is fetched live from WEEX on startup — no env var needed

# How many top coins to scan (set 10 or 20 in .env)
TOP_N_COINS  = int(os.getenv('TOP_N_COINS', '20'))

# How often to run the auto-scan loop (default: every 5 minutes)
SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL_SECONDS', '300'))

# Batch mode settings — max candidates sent to Claude, min pre-score to qualify
BATCH_SIZE          = int(os.getenv('BATCH_SIZE', '5'))
PRE_SCORE_THRESHOLD = int(os.getenv('PRE_SCORE_THRESHOLD', '55'))

public_exchange = ccxt.binance({'options': {'defaultType': 'spot'}, 'enableRateLimit': True})

feed   = DataFeed(public_exchange, cryptopanic_token=os.getenv('CRYPTOPANIC_TOKEN', ''))
claude = ClaudeAgent(api_key=os.getenv('ANTHROPIC_API_KEY', ''))
notifier = TelegramNotifier(
    token   = os.getenv('TELEGRAM_BOT_TOKEN', ''),
    chat_id = os.getenv('TELEGRAM_CHAT_ID', ''),
)
risk   = RiskManager(RiskSettings(
    account_usdt    = 0.0,  # seeded from live WEEX balance in lifespan startup
    risk_pct        = float(os.getenv('RISK_PCT', '1.5')),
    max_trade_usdt  = float(os.getenv('MAX_TRADE_USDT', '75')),
    min_confidence  = int(os.getenv('MIN_CONFIDENCE', '75')),
    daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT_USDT', '50')),
    max_concurrent  = int(os.getenv('MAX_CONCURRENT_TRADES', '2')),
))
weex = WeexClient(
    api_key       = os.getenv('WEEX_API_KEY', ''),
    secret        = os.getenv('WEEX_SECRET', ''),
    passphrase    = os.getenv('WEEX_PASSPHRASE', ''),
    paper         = PAPER,
    paper_balance = float(os.getenv('ACCOUNT_USDT', '10000')),
)
news_watcher   = NewsWatcher(blackout_minutes=int(os.getenv('NEWS_BLACKOUT_MIN', '30')))
journal        = TradeJournal()
pump_scanner   = PumpScanner(public_exchange)
trade_manager  = TradeManager(weex, risk, notifier)

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


def _make_signal(data: dict) -> dict:
    """
    Phase 3 flow: rules generate, LLM vetoes.

    1. generate_signal() applies all hard gates (MTF, R:R, volume, regime, confidence).
    2. If not WAIT, call claude.veto_signal() to APPROVE / REJECT / DOWNGRADE.
    3. Return the final signal (always includes 'action' and 'symbol').
    """
    rule_sig = _rule_signal(data)
    symbol   = data.get('symbol', '')
    rule_sig.setdefault('symbol', symbol)

    if rule_sig['action'] == 'WAIT':
        return rule_sig

    # Enrich with live-context fields the dashboard expects
    rule_sig.update({
        'symbol':         symbol,
        'price':          data.get('price', rule_sig.get('entry', 0)),
        'timestamp':      data.get('timestamp', ''),
        'pre_score':      rule_sig['confidence'],
        'timeframe_bias': data.get('timeframe_bias', {}).get('overall', 'MIXED'),
        'volume_signal':  data.get('volume_signal', 'NORMAL'),
        'candle_pattern': data.get('candle_pattern', ''),
        'sentiment':      data.get('fear_greed', {}).get('label', 'Neutral'),
        'session':        claude._trading_session(),
        'atr':            data.get('atr', rule_sig.get('atr', 0)),
    })

    try:
        veto = claude.veto_signal(rule_sig, data)
        decision = veto.get('decision', 'APPROVE')
        adj      = veto.get('confidence_adjustment', 0)
        rule_sig['veto_decision'] = decision
        rule_sig['veto_reason']   = veto.get('reason', '')

        if decision == 'REJECT':
            rule_sig['action'] = 'WAIT'
            rule_sig['reason'] = 'LLM veto: ' + veto.get('reason', '')
            return rule_sig

        rule_sig['confidence'] = max(0, min(100, rule_sig['confidence'] + adj))
        if rule_sig['confidence'] < risk.settings.min_confidence:
            rule_sig['action'] = 'WAIT'
            rule_sig['reason'] = (
                f'Post-veto confidence {rule_sig["confidence"]} '
                f'< {risk.settings.min_confidence}')
    except Exception as e:
        print(f'[Veto] Error for {symbol}: {e} — proceeding with rule signal')

    return rule_sig


async def sync_balance_to_risk():
    """Fetch live USDT available from Weex and sync it to the risk manager."""
    try:
        usdt = weex.get_usdt_available()
        if usdt > 0:
            risk.update_account_balance(usdt)
            log_trade(f'[Risk] Balance synced: ${usdt:.2f}')
        else:
            log_trade('[Risk] Balance sync returned $0.00 — check exchange connection')
    except Exception as e:
        log_trade(f'[Risk] Balance sync error: {e}')


# ── BACKGROUND TASKS ──────────────────────────────────────────────────────────

async def bg_auto_scanner():
    """
    BATCH MODE ENGINE — 2 phases per scan cycle:
      Phase 1: Collect data for all coins (no Claude), pre-score with _score_confidence().
      Phase 2: ONE Claude batch call for the top BATCH_SIZE candidates.

    Token cost: was N calls/scan → now always 1 call/scan regardless of coin count.
    """
    print(f'[AutoScanner] Batch mode — top {TOP_N_COINS} coins, '
          f'max {BATCH_SIZE} to Claude, threshold {PRE_SCORE_THRESHOLD}/100')
    await asyncio.sleep(15)

    while True:
        try:
            # ── Build coin list ──────────────────────────────────────────────
            top_coins  = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)
            pump_coins = pump_scanner.get_pump_symbols()
            seen, merged = set(), []
            for c in (pump_coins + top_coins):
                if c not in seen:
                    seen.add(c)
                    merged.append(c)
            top_coins = merged[:TOP_N_COINS + len(pump_coins)]
            if pump_coins:
                print(f'[PumpScanner] Prioritising {len(pump_coins)}: {pump_coins[:5]}')

            print(f'[AutoScanner] Phase 1 — rule-based filter for {len(top_coins)} coins')
            await broadcast({
                'type': 'scan_start', 'coins': top_coins, 'count': len(top_coins),
                'message': f'Phase 1: rule filter for {len(top_coins)} coins...',
            })

            # ── Phase 1: rule-based filter — ZERO Claude calls ──────────────
            candidates: List[Dict] = []
            for symbol in top_coins:
                try:
                    data      = feed.get_full_analysis(symbol, '1h')
                    rule_sig  = _rule_signal(data)

                    if rule_sig['action'] == 'WAIT':
                        await broadcast({'type': 'scan_skip', 'symbol': symbol,
                                         'reason': rule_sig.get('reason', 'Rule gate')})
                        await asyncio.sleep(0.2)
                        continue

                    candidates.append((rule_sig, data))
                    await broadcast({'type': 'scan_candidate', 'symbol': symbol,
                                     'pre_score': rule_sig['confidence']})
                    print(f'[AutoScanner] {symbol} — candidate '
                          f'(conf={rule_sig["confidence"]}, {rule_sig["action"]})')
                    await asyncio.sleep(0.3)

                except Exception as e:
                    print(f'[AutoScanner] Data error {symbol}: {e}')
                    await asyncio.sleep(0.3)

            # Sort by confidence, limit to BATCH_SIZE for veto phase
            candidates.sort(key=lambda x: x[0].get('confidence', 0), reverse=True)
            batch = candidates[:BATCH_SIZE]

            # ── Phase 2: LLM veto — one call per candidate ───────────────────
            signals_found: List[Dict] = []

            if batch:
                syms = [d['symbol'] for _, d in batch]
                print(f'[AutoScanner] Phase 2 — LLM veto for {len(batch)} candidates: {syms}')
                await broadcast({
                    'type': 'scan_status',
                    'message': (f'Phase 2: LLM veto for {len(batch)} rule-approved setups '
                                f'({len(top_coins)-len(batch)} pre-filtered)...'),
                    'candidates': syms,
                })

                for rule_sig, data in batch:
                    signal = _make_signal(data)
                    sym    = signal.get('symbol', data.get('symbol', ''))
                    action = signal.get('action', 'WAIT')
                    conf   = signal.get('confidence', 0)

                    await broadcast({'type': 'signal', 'symbol': sym, 'signal': signal})

                    if action in ('LONG', 'SHORT'):
                        log_trade(f"SIGNAL | {action} | {sym} | conf:{conf}% | "
                                  f"{signal.get('setup_type','')} | "
                                  f"veto:{signal.get('veto_decision','APPROVE')}")
                        signals_found.append(signal)
                        notifier.send_signal(signal)
                        if AUTO_TRADE:
                            await maybe_execute(signal)
                    else:
                        print(f'[AutoScanner] {sym} — WAIT after veto '
                              f'(conf:{conf}%, {signal.get("veto_decision","")}: '
                              f'{signal.get("veto_reason","")})')

            else:
                print('[AutoScanner] No candidates passed rule filter — LLM skipped entirely')

            await broadcast({
                'type':          'scan_complete',
                'total_scanned': len(top_coins),
                'candidates':    len(batch),
                'signals_found': len(signals_found),
                'next_scan_in':  SCAN_INTERVAL,
                'message': (f'Done — {len(top_coins)} scanned → {len(batch)} rule-approved '
                            f'→ {len(signals_found)} signal(s) after LLM veto. '
                            f'Next in {SCAN_INTERVAL//60}min.'),
            })
            notifier.send_scan_summary(len(top_coins), len(signals_found), SCAN_INTERVAL)
            print(f'[AutoScanner] Done. {len(batch)} LLM calls, {len(signals_found)} signals. '
                  f'Sleeping {SCAN_INTERVAL}s...')

        except Exception as e:
            print(f'[AutoScanner] Critical error: {e}')
            await asyncio.sleep(60)

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


async def bg_balance_broadcast():
    """Every 30s: sync live Weex balance to risk manager, then broadcast."""
    while True:
        await asyncio.sleep(30)
        try:
            await sync_balance_to_risk()
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
    await sync_balance_to_risk()   # always size from live wallet, never a stale env var
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
            trade_manager.register({
                'id': order_id, 'symbol': signal['symbol'], 'side': signal['action'],
                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'qty': qty,
                'atr': signal.get('atr', 0), 'invalidation': signal.get('invalidation', 0),
            })
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
    """Trigger an immediate batch scan (same two-phase logic as auto-scanner)."""
    async def _run():
        top_coins = feed.get_top_coins_by_volume(top_n=TOP_N_COINS)
        await broadcast({'type': 'scan_start', 'coins': top_coins, 'count': len(top_coins),
                         'message': f'Manual scan — Phase 1: collecting data ({len(top_coins)} coins)...'})
        candidates: List[Dict] = []
        for symbol in top_coins:
            try:
                data     = feed.get_full_analysis(symbol, '1h')
                rule_sig = _rule_signal(data)
                if rule_sig['action'] == 'WAIT':
                    await broadcast({'type': 'scan_skip', 'symbol': symbol,
                                     'reason': rule_sig.get('reason', 'Rule gate')})
                    await asyncio.sleep(0.2)
                    continue
                candidates.append((rule_sig, data))
                await broadcast({'type': 'scan_candidate', 'symbol': symbol,
                                 'pre_score': rule_sig['confidence']})
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f'[ManualScan] {symbol}: {e}')

        candidates.sort(key=lambda x: x[0].get('confidence', 0), reverse=True)
        batch = candidates[:BATCH_SIZE]
        signals_found: List[Dict] = []

        if batch:
            await broadcast({'type': 'scan_status',
                             'message': f'Phase 2: LLM veto for {len(batch)} rule-approved setups...',
                             'candidates': [d['symbol'] for _, d in batch]})
            for rule_sig, data in batch:
                signal = _make_signal(data)
                sym    = signal.get('symbol', data.get('symbol', ''))
                action = signal.get('action', 'WAIT')
                await broadcast({'type': 'signal', 'symbol': sym, 'signal': signal})
                if action in ('LONG', 'SHORT'):
                    signals_found.append(signal)
                    notifier.send_signal(signal)
                    if AUTO_TRADE:
                        await maybe_execute(signal)

        await broadcast({'type': 'scan_complete', 'total_scanned': len(top_coins),
                         'candidates': len(batch), 'signals_found': len(signals_found),
                         'next_scan_in': SCAN_INTERVAL,
                         'message': f'Manual scan done — {len(batch)} rule-approved → {len(signals_found)} signal(s) after LLM veto.'})
    background_tasks.add_task(_run)
    return {'status': 'scan_started', 'coins': TOP_N_COINS, 'batch_size': BATCH_SIZE}


@app.post('/analyze/{symbol:path}')
async def analyze_manual(symbol: str, background_tasks: BackgroundTasks):
    """Manual single-coin analysis (for testing specific coins)."""
    try:
        ex_symbol = symbol.replace('-', '/').upper()
        data      = feed.get_full_analysis(ex_symbol, '1h')
        signal    = _make_signal(data)
        background_tasks.add_task(broadcast, {'type': 'signal', 'symbol': ex_symbol, 'signal': signal})
        action = signal.get('action', 'WAIT')
        conf   = signal.get('confidence', 0)
        if action in ('LONG', 'SHORT'):
            log_trade(f"SIGNAL | {action} | {ex_symbol} | conf:{conf}% | "
                      f"{signal.get('setup_type','')} | veto:{signal.get('veto_decision','APPROVE')}")
            if AUTO_TRADE:
                background_tasks.add_task(maybe_execute, signal)
        return signal
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

@app.get('/weex/test')
async def test_weex():
    """Test WEEX API credentials and connectivity."""
    result = weex.test_connection()
    return result

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

    leverage = signal.get('leverage')
    if leverage and entry:
        # Modal-selected leverage — derive qty from free balance × leverage / price
        bal = weex.get_balance()
        usdt_free = (bal.get('USDT') or {}).get('free', 0) or risk.settings.account_usdt
        raw_qty = usdt_free * int(leverage) / entry
        qty = round(raw_qty, 4)
        sizing = {'qty': qty, 'leverage': int(leverage), 'risk_mode': 'manual'}
    else:
        sizing = risk.position_size(entry, sl, confidence=conf)
        qty = sizing.get('qty', 0)

    if qty <= 0:
        return {'executed': False, 'reason': 'Calculated qty is zero'}

    sym    = signal.get('symbol', '')
    action = signal.get('action', '')
    result = weex.place_order(sym, action, qty, entry, sl, tp1, tp2, tp3)
    if result:
        order_id = result.get('id', '')
        risk.record_open(order_id, signal)
        trade_manager.register({
            'id': order_id, 'symbol': sym, 'side': action,
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'qty': qty,
            'atr': signal.get('atr', 0), 'invalidation': signal.get('invalidation', 0),
        })
        await broadcast({'type': 'trade_opened', 'order': result, 'signal': signal})
        notifier.send_trade_opened(sym, action, qty, entry, sl, tp1)
        log_trade(f"MANUAL | {action} {qty} {sym} @ {entry} | SL:{sl} TP1:{tp1} | lev:{leverage or 'risk-based'}")
    return {'executed': bool(result), 'order': result, 'sizing': sizing}

@app.post('/backtest/{symbol:path}')
async def run_backtest(symbol: str):
    """Run a backtest for a symbol using historical OHLCV data."""
    try:
        from backtest import Backtester
        ex_symbol   = symbol.replace('-', '/').upper()
        initial_cap = risk.settings.account_usdt or 10_000.0
        bt = Backtester(feed, ex_symbol, timeframe='1h', initial_capital=initial_cap)
        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: bt.run(lookback_candles=500)
        )
        # Downsample equity curve so the JSON response stays compact
        ec   = results.get('equity_curve', [])
        step = max(1, len(ec) // 100)
        results['equity_curve'] = ec[::step]
        return results
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

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
    print(f'  Account USDT : fetched from WEEX on startup')
    print(f'  Top N Coins  : {TOP_N_COINS}')
    print(f'  Scan Every   : {SCAN_INTERVAL}s ({SCAN_INTERVAL//60} min)')
    print(f'  Min Confidence: 72%')
    print('  Listening    : http://127.0.0.1:8000')
    print('  Dashboard    : open trading_dashboard.html')
    print('  >> NO MANUAL SCAN BUTTON NEEDED -- fully automatic')
    print('=' * 60)
    uvicorn.run(app, host=os.getenv('HOST', '127.0.0.1'),
                port=int(os.getenv('PORT', '8000')))