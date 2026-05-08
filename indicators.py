"""Pure indicator computation — shared by data_feed.py and the backtester."""
import pandas as pd
import pandas_ta_classic as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all TA columns to a copy of df. Returns the augmented DataFrame."""
    df = df.copy()

    df['ema20']  = ta.ema(df['close'], length=20)
    df['ema50']  = ta.ema(df['close'], length=50)
    ema200       = ta.ema(df['close'], length=200)
    df['ema200'] = ema200 if ema200 is not None else float('nan')
    df['rsi']    = ta.rsi(df['close'], length=14)
    df['atr']    = ta.atr(df['high'], df['low'], df['close'], length=14)

    macd = ta.macd(df['close'])
    df['macd']     = macd['MACD_12_26_9']
    df['macd_sig'] = macd['MACDs_12_26_9']

    bb = ta.bbands(df['close'])
    df['bb_upper'] = bb[[c for c in bb.columns if c.startswith('BBU')][0]]
    df['bb_lower'] = bb[[c for c in bb.columns if c.startswith('BBL')][0]]
    df['bb_mid']   = bb[[c for c in bb.columns if c.startswith('BBM')][0]]

    df['volume_sma'] = ta.sma(df['volume'], length=20)

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None:
        adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
        df['adx'] = adx_df[adx_col]
    else:
        df['adx'] = float('nan')

    return df
