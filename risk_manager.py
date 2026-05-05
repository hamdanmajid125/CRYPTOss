from dataclasses import dataclass
from typing import Tuple, List, Dict
import time
import datetime
import json
import os


@dataclass
class RiskSettings:
    account_usdt:     float = 10000.0
    risk_pct:         float = 1.0
    max_trade_usdt:   float = 150.0
    min_confidence:   int   = 72
    daily_loss_limit: float = 250.0
    min_rr:           float = 3.0
    max_concurrent:   int   = 3


class RiskManager:
    def __init__(self, settings: RiskSettings, state_file: str = 'risk_state.json'):
        self.settings   = settings
        self.state_file = state_file
        self.open_trades: List[Dict] = []

        # Daily stats (reset at midnight UTC)
        self._day_key: str = self._utc_day()
        self.daily_trades:   int   = 0
        self.daily_wins:     int   = 0
        self.daily_losses:   int   = 0
        self.daily_pnl:      float = 0.0
        self.best_trade:     float = 0.0
        self.worst_trade:    float = 0.0

        # Drawdown protection
        self.peak_balance:   float = settings.account_usdt
        self.current_balance: float = settings.account_usdt

        # Consecutive loss tracking
        self.consecutive_losses: int = 0
        self.pause_until: float = 0.0   # epoch seconds

        # Correlation tracking: symbol -> side
        self.btc_eth_sides: Dict[str, str] = {}

        self._load_state()

    # ── STATE PERSISTENCE ─────────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                'day_key':            self._day_key,
                'consecutive_losses': self.consecutive_losses,
                'pause_until':        self.pause_until,
                'daily_pnl':          self.daily_pnl,
                'daily_trades':       self.daily_trades,
                'daily_wins':         self.daily_wins,
                'daily_losses':       self.daily_losses,
                'best_trade':         self.best_trade,
                'worst_trade':        self.worst_trade,
                'current_balance':    self.current_balance,
                'peak_balance':       self.peak_balance,
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f'[Risk] State save failed: {e}')

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            # Always restore balance curve
            self.current_balance = state.get('current_balance', self.settings.account_usdt)
            self.peak_balance    = state.get('peak_balance',    self.settings.account_usdt)
            self.consecutive_losses = state.get('consecutive_losses', 0)
            self.pause_until        = state.get('pause_until', 0.0)
            # Restore daily stats only if the saved day matches today
            if state.get('day_key') == self._utc_day():
                self.daily_pnl    = state.get('daily_pnl',    0.0)
                self.daily_trades = state.get('daily_trades',  0)
                self.daily_wins   = state.get('daily_wins',    0)
                self.daily_losses = state.get('daily_losses',  0)
                self.best_trade   = state.get('best_trade',    0.0)
                self.worst_trade  = state.get('worst_trade',   0.0)
            print(f'[Risk] State restored from {self.state_file}')
        except Exception as e:
            print(f'[Risk] State load failed: {e}')

    def clear_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    # ── DAILY RESET ────────────────────────────────────────────────────────────

    def _utc_day(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')

    def _reset_daily_if_needed(self):
        today = self._utc_day()
        if today != self._day_key:
            self._day_key      = today
            self.daily_trades  = 0
            self.daily_wins    = 0
            self.daily_losses  = 0
            self.daily_pnl     = 0.0
            self.best_trade    = 0.0
            self.worst_trade   = 0.0

    # ── CAN TRADE ──────────────────────────────────────────────────────────────

    def can_trade(self, signal: Dict) -> Tuple[bool, str]:
        self._reset_daily_if_needed()

        if signal.get('action') == 'WAIT':
            return False, 'Signal is WAIT'

        # Pause check (consecutive losses)
        if time.time() < self.pause_until:
            remaining = int((self.pause_until - time.time()) / 60)
            return False, f'BOT PAUSED — {remaining} min remaining (consecutive losses / large loss)'

        # Daily loss limit
        if self.daily_pnl <= -self.settings.daily_loss_limit:
            return False, f'Daily loss limit hit (${self.daily_pnl:.2f}). Trading blocked for today.'

        # Confidence gate
        conf = signal.get('confidence', 0)
        if conf < self.settings.min_confidence:
            return False, f'Confidence {conf}% below minimum {self.settings.min_confidence}%'

        # Max concurrent trades
        if len(self.open_trades) >= self.settings.max_concurrent:
            return False, f'Max concurrent trades ({self.settings.max_concurrent}) reached'

        # Drawdown protection
        dd_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if dd_pct >= 10:
            return False, f'10% drawdown from peak hit. All trading halted. Drawdown: {dd_pct:.1f}%'

        # R:R check
        rr = self._parse_rr(signal.get('rr', '0'))
        if rr < self.settings.min_rr:
            entry = signal.get('entry', 0)
            sl = signal.get('sl', 0)
            tp = signal.get('tp1', 0)
            if entry and sl and tp:
                risk_d = abs(entry - sl)
                reward_d = abs(tp - entry)
                rr = reward_d / risk_d if risk_d > 0 else 0
            if rr < self.settings.min_rr:
                return False, f'R:R {rr:.2f} below minimum {self.settings.min_rr}'

        # TP1 ATR-proximity gate (requires atr field added by get_signal/get_batch_signals)
        atr_v    = float(signal.get('atr', 0) or 0)
        tp1_v    = float(signal.get('tp1', 0) or 0)
        entry_v  = float(signal.get('entry', 0) or 0)
        action_v = signal.get('action', '')
        if atr_v > 0 and tp1_v > 0 and entry_v > 0 and action_v in ('LONG', 'SHORT'):
            tp1_dist = (tp1_v - entry_v) if action_v == 'LONG' else (entry_v - tp1_v)
            if tp1_dist > 0:
                tp1_atr = tp1_dist / atr_v
                if tp1_atr < 0.5:
                    return False, (f'TP1 too close: {tp1_atr:.2f}x ATR (min 0.5x) — '
                                   f'fees eat profit at this distance')
                if tp1_atr > 3.0:
                    return False, (f'TP1 too far: {tp1_atr:.2f}x ATR (max 3.0x) — '
                                   f'low probability of hitting before reversal')

        # Correlation filter
        sym = signal.get('symbol', '')
        action = signal.get('action', '')
        ok, reason = self._correlation_check(sym, action)
        if not ok:
            return False, reason

        return True, 'OK'

    def _correlation_check(self, symbol: str, action: str) -> Tuple[bool, str]:
        is_btc = 'BTC' in symbol
        is_eth = 'ETH' in symbol
        if not (is_btc or is_eth):
            return True, 'OK'

        btc_eth_open = [t for t in self.open_trades
                        if 'BTC' in t['signal'].get('symbol', '') or
                           'ETH' in t['signal'].get('symbol', '')]
        if len(btc_eth_open) >= 2:
            return False, 'Correlation limit: max 2 positions in BTC/ETH combined'

        btc_side = next((t['signal']['action'] for t in self.open_trades
                         if 'BTC' in t['signal'].get('symbol', '')), None)
        eth_side = next((t['signal']['action'] for t in self.open_trades
                         if 'ETH' in t['signal'].get('symbol', '')), None)

        # Never long BTC while short ETH (or vice versa) — high correlation risk
        if is_btc and action == 'LONG' and eth_side == 'SHORT':
            return False, 'Correlation conflict: LONG BTC + SHORT ETH'
        if is_btc and action == 'SHORT' and eth_side == 'LONG':
            return False, 'Correlation conflict: SHORT BTC + LONG ETH'
        if is_eth and action == 'LONG' and btc_side == 'SHORT':
            return False, 'Correlation conflict: LONG ETH + SHORT BTC'
        if is_eth and action == 'SHORT' and btc_side == 'LONG':
            return False, 'Correlation conflict: SHORT ETH + LONG BTC'

        return True, 'OK'

    # ── POSITION SIZING (Kelly-influenced) ─────────────────────────────────────

    def position_size(self, entry: float, sl: float,
                      confidence: int = 70) -> Dict:
        base_risk = self.settings.account_usdt * self.settings.risk_pct / 100
        base_risk = min(base_risk, self.settings.max_trade_usdt)

        # Kelly scaling by confidence
        if confidence >= 80:
            risk_usdt = min(base_risk * 2.0, self.settings.max_trade_usdt)
        elif confidence >= 65:
            risk_usdt = base_risk
        else:
            risk_usdt = base_risk * 0.5

        # Drawdown reduction: 5% drawdown → halve size
        dd_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if dd_pct >= 5:
            risk_usdt *= 0.5

        sl_distance_pct = abs(entry - sl) / entry
        if sl_distance_pct <= 0:
            return {'usdt': 0, 'qty': 0, 'risk_usdt': 0}

        position_usdt = risk_usdt / sl_distance_pct
        position_usdt = min(position_usdt, self.settings.max_trade_usdt * 10)
        qty = position_usdt / entry

        return {
            'risk_usdt':        round(risk_usdt, 2),
            'position_usdt':    round(position_usdt, 2),
            'qty':              round(qty, 6),
            'sl_distance_pct':  round(sl_distance_pct * 100, 2),
            'kelly_multiplier': 2.0 if confidence >= 80 else 1.0,
        }

    # ── TRADE RECORDING ────────────────────────────────────────────────────────

    def record_open(self, trade_id: str, signal: dict):
        self.open_trades.append({
            'id': trade_id,
            'signal': signal,
            'time': time.time(),
        })
        self.daily_trades += 1
        self._save_state()

    def record_close(self, trade_id: str, pnl_usdt: float):
        self.open_trades = [t for t in self.open_trades if t['id'] != trade_id]
        self.daily_pnl  += pnl_usdt
        self.current_balance += pnl_usdt

        if pnl_usdt > 0:
            self.daily_wins += 1
            self.consecutive_losses = 0
            self.best_trade = max(self.best_trade, pnl_usdt)
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
        else:
            self.daily_losses += 1
            self.consecutive_losses += 1
            self.worst_trade = min(self.worst_trade, pnl_usdt)
            self._check_pause_conditions(pnl_usdt)
        self._save_state()

    def _check_pause_conditions(self, pnl_usdt: float):
        if self.consecutive_losses >= 5:
            self.pause_until = time.time() + 259200  # 72 hours
            print(f'[Risk] 5 consecutive losses — pausing trading for 72 hours')
        elif self.consecutive_losses >= 3:
            self.pause_until = time.time() + 7200  # 2 hours
            print(f'[Risk] 3 consecutive losses — pausing trading for 2 hours')

        loss_pct = abs(pnl_usdt) / self.settings.account_usdt * 100
        if loss_pct >= 3:
            self.pause_until = max(self.pause_until, time.time() + 3600)  # 1 hour
            print(f'[Risk] Single loss >{loss_pct:.1f}% of account — pausing 1 hour')

    # ── LIVE BALANCE SYNC ──────────────────────────────────────────────────────

    def update_account_balance(self, live_balance: float):
        """Sync risk math with live wallet balance from exchange."""
        if live_balance <= 0:
            return
        self.settings.account_usdt = live_balance
        self.current_balance = live_balance
        if live_balance > self.peak_balance:
            self.peak_balance = live_balance

    # ── STATS ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        self._reset_daily_if_needed()
        win_rate = (self.daily_wins / self.daily_trades * 100) if self.daily_trades else 0
        dd_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        paused = time.time() < self.pause_until
        pause_remaining = max(0, int((self.pause_until - time.time()) / 60)) if paused else 0

        return {
            'date':                self._day_key,
            'trades_taken':        self.daily_trades,
            'wins':                self.daily_wins,
            'losses':              self.daily_losses,
            'win_rate':            round(win_rate, 1),
            'total_pnl':           round(self.daily_pnl, 2),
            'best_trade':          round(self.best_trade, 2),
            'worst_trade':         round(self.worst_trade, 2),
            'consecutive_losses':  self.consecutive_losses,
            'peak_balance':        round(self.peak_balance, 2),
            'current_balance':     round(self.current_balance, 2),
            'drawdown_pct':        round(dd_pct, 2),
            'open_trades':         len(self.open_trades),
            'paused':              paused,
            'pause_remaining_min': pause_remaining,
            'daily_loss_limit':    self.settings.daily_loss_limit,
            'daily_loss_used_pct': round(abs(min(self.daily_pnl, 0)) / self.settings.daily_loss_limit * 100, 1),
        }

    # ── PARTIAL PROFIT ENGINE (Phase 2B) ──────────────────────────────────────

    def scale_out_plan(self, entry: float, qty: float,
                       tp1: float, tp2: float, tp3: float) -> dict:
        """
        Returns a scale-out plan: qty to close at each TP level and new SL after each hit.
        Default split: 40% at TP1, 40% at TP2, 20% runner to TP3.
        """
        qty_tp1    = round(qty * 0.40, 6)
        qty_tp2    = round(qty * 0.40, 6)
        qty_runner = round(qty - qty_tp1 - qty_tp2, 6)
        return {
            'tp1':    {'price': tp1, 'qty': qty_tp1,    'new_sl': entry},   # move SL to BE
            'tp2':    {'price': tp2, 'qty': qty_tp2,    'new_sl': tp1},     # trail SL to TP1
            'runner': {'price': tp3, 'qty': qty_runner, 'new_sl': tp2},     # trail SL to TP2
        }

    def expected_value(self, entry: float, sl: float,
                       tp1: float, win_rate: float = 0.45) -> dict:
        """Quick EV calculation to confirm a trade is positive-expectancy."""
        risk   = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr     = round(reward / risk, 2) if risk else 0
        ev     = round(win_rate * reward - (1 - win_rate) * risk, 4)
        return {'rr': rr, 'ev': ev, 'positive': ev > 0}

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def _parse_rr(self, rr_str: str) -> float:
        try:
            return float(str(rr_str).split(':')[-1])
        except Exception:
            return 0.0

    def is_paused(self) -> bool:
        return time.time() < self.pause_until
