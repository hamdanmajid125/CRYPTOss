import time


def with_retry(fn, retries: int = 3, base_delay: float = 2.0):
    """
    Retries fn up to `retries` times with exponential backoff.
    Catches transient network/exchange errors by class name matching.
    Non-transient errors are re-raised immediately.
    """
    _TRANSIENT = ('NetworkError', 'RequestTimeout', 'ExchangeNotAvailable',
                  'ConnectionError', 'Timeout', 'RequestException', 'ReadTimeout')
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            cls = type(e).__name__
            if not any(t in cls for t in _TRANSIENT):
                raise
            last_exc = e
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f'[Retry] {cls} on attempt {attempt + 1}/{retries}, retrying in {delay:.0f}s')
                time.sleep(delay)
    raise last_exc
