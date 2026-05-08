"""Historical OHLCV loader — fetches from ccxt, caches to local Parquet files."""
import os
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def _parquet_path(symbol: str, timeframe: str, start: str, end: str) -> Path:
    safe = symbol.replace('/', '_')
    return DATA_DIR / f'{safe}_{timeframe}_{start}_{end}.parquet'


def load_ohlcv(
    symbol: str,
    timeframe: str = '1h',
    start: str = '2024-01-01',
    end: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV data for *symbol* between *start* and *end* (YYYY-MM-DD strings).
    Caches to backtest/data/*.parquet — re-use on subsequent calls.
    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC datetime.
    """
    if end is None:
        end = pd.Timestamp.utcnow().strftime('%Y-%m-%d')

    cache_path = _parquet_path(symbol, timeframe, start, end)
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        print(f'[DataLoader] {symbol} loaded from cache ({len(df)} bars)')
        return df

    print(f'[DataLoader] Fetching {symbol} {timeframe} {start} → {end} from Binance …')
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

    since_ms = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end,   tz='UTC').timestamp() * 1000)

    bars = []
    while since_ms < end_ms:
        try:
            chunk = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        except Exception as e:
            print(f'[DataLoader] fetch error: {e}')
            time.sleep(2)
            continue

        if not chunk:
            break
        bars.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= end_ms or len(chunk) < 1000:
            break
        since_ms = last_ts + 1
        time.sleep(ex.rateLimit / 1000)

    if not bars:
        raise RuntimeError(f'No data returned for {symbol} {timeframe} {start}→{end}')

    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[df.index < pd.Timestamp(end, tz='UTC')]
    df = df[~df.index.duplicated(keep='last')]

    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    df.to_parquet(cache_path)
    print(f'[DataLoader] {symbol}: {len(df)} bars fetched and cached to {cache_path.name}')
    return df
