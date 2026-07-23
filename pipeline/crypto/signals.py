from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _pivot(panel_df, value):
    return panel_df.pivot(index="open_time", columns="symbol", values=value).sort_index()

def compute_momentum(panel_df: pd.DataFrame, formation: int = 12, skip: int = 1) -> pd.DataFrame:
    close = _pivot(panel_df, "close")
    # 12-1 momentum: skip 1 bar, then ratio over 12 bars, subtract 1 for return
    momentum = close.shift(skip) / close.shift(formation) - 1
    long = momentum.stack().reset_index()
    long.columns = ["open_time", "symbol", "momentum"]
    return long.dropna(subset=["momentum"]).sort_values(["open_time", "symbol"]).reset_index(drop=True)

def compute_net_taker_flow(panel_df: pd.DataFrame, window: int = 11) -> pd.DataFrame:
    volume = _pivot(panel_df, "volume")
    taker = _pivot(panel_df, "taker_buy_volume")
    net = 2 * taker - volume  # taker_buy - taker_sell
    net_sum = net.shift(1).rolling(window).sum()
    vol_sum = volume.shift(1).rolling(window).sum()
    flow = net_sum / vol_sum
    long = flow.stack().reset_index()
    long.columns = ["open_time", "symbol", "flow"]
    return long.dropna(subset=["flow"]).sort_values(["open_time", "symbol"]).reset_index(drop=True)
