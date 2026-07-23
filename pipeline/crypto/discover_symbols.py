import json
import urllib.request
import xml.etree.ElementTree as ET

_LISTING = ("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
            "?delimiter=/&prefix=data/spot/monthly/klines/")
_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def list_usdt_symbols() -> list[str]:
    try:
        symbols, token = [], None
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        while True:
            url = _LISTING + (f"&marker={token}" if token else "")
            root = ET.fromstring(_get(url))
            for cp in root.findall(f"{ns}CommonPrefixes/{ns}Prefix"):
                sym = cp.text.rstrip("/").split("/")[-1]
                if sym.endswith("USDT"):
                    symbols.append(sym)
            truncated = root.findtext(f"{ns}IsTruncated") == "true"
            token = root.findtext(f"{ns}NextMarker")
            if not truncated or not token:
                break
        if symbols:
            return sorted(set(symbols))
    except Exception:
        pass
    info = json.loads(_get(_EXCHANGE_INFO))
    return sorted({s["symbol"] for s in info["symbols"] if s["symbol"].endswith("USDT")})
