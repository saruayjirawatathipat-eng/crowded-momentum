# Crowded Momentum

Does trading-volume information predict the cross-section of momentum returns —
and does the answer depend on timeframe? A three-phase study from U.S. equity
turnover to cryptocurrency order flow, done during a data-science internship at
Siam Commercial Bank (SCB).

**Headline finding, one honest sentence:** volume-confirmed momentum shows a
real, internally consistent edge at the weekly and daily timeframe but is
statistically indistinguishable from zero intraday (4-hour, 1-hour) — a
suggestive, timeframe-localized pattern, not a proven effect (the cleanest
non-overlapping test falls just short of conventional significance).

**Read the write-up:** _(link added after first deploy — see Task 9)_

## What this is

The study runs in three phases, each provoked by the previous phase's
failure rather than by a fresh guess:

1. **Phase 1 — turnover.** Do high-turnover ("crowded") momentum winners in
   the S&P 500 subsequently underperform? No — they outperform at every
   horizon, the opposite of the crowding-reversal hypothesis.
2. **Phase 2 — signed volume (OBV).** Turnover is unsigned, so the volume is
   signed via on-balance volume to separate accumulation from distribution.
   Result: a null, but one driven by the sample — a single equity bull market
   has almost no genuine price/flow divergence to test against.
3. **Phase 3 — real taker flow (CVD), timeframe swept.** Move to a
   multi-regime Binance crypto panel, replace the OBV proxy with true
   taker-signed cumulative volume delta, and sweep the identical test across
   monthly → weekly → daily → 4h → 1h. The effect appears at weekly/daily and
   vanishes intraday.

The full argument, data caveats, and worked robustness fixes (winsorizing a
LUNA-collapse-driven +5,199,900% forward return, an OBV divergence rate of
only 2.7% in the equity sample, etc.) are in `index.qmd`.

## Repo map

```
analysis/       Thin, test-covered layer that reads the committed data/
                 summaries and produces the tables and figure embedded in
                 index.qmd (analysis/figures.py).
data/            Three committed summary files — the only data checked into
                 this repo (see "Data" below).
pipeline/sp500/  Phase 1-2 pipeline: fetches S&P 500 prices/volume, forms
                 12-1 momentum deciles, splits by turnover then by OBV net
                 flow, computes forward returns.
pipeline/crypto/ Phase 3 pipeline: fetches Binance klines, builds a
                 point-in-time liquid-coin universe, resamples to five
                 timeframes, computes CVD net taker flow, and sweeps the
                 confirmed/divergent spread test across the timeframe ladder.
index.qmd        The paper itself — a Quarto document that runs analysis/
                 code live at render time to generate every table and the
                 significance figure, so the numbers in the HTML/PDF are
                 always reproduced from data/, never hand-copied.
tests/           Tests for analysis/figures.py; each pipeline stage also has
                 its own tests under pipeline/{sp500,crypto}/tests/.
```

## Data

Three small summary files are committed under `data/`:

- `data/sp500_turnover_forward_returns.parquet` — Phase 1 input
- `data/sp500_obv_forward_returns.parquet` — Phase 2 input
- `data/crypto_sweep_table.parquet` — Phase 3 input

These are the only data files in the repo, and they're what `analysis/figures.py`
and `index.qmd` read to build every table and the figure. The full pipelines
that produce them from raw exchange data (S&P 500 prices/volume via yfinance;
Binance klines via the public REST mirror) live under `pipeline/sp500/` and
`pipeline/crypto/`, run as numbered stages (`01_...py` through the final
report stage in each directory). That raw and intermediate data is heavy and
not committed — see `.gitignore` — so reproducing `data/` from scratch means
running each pipeline's numbered scripts in order, which will re-fetch from
the network.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest                 # verify tables/figure reproduce the paper's numbers
quarto render                     # build the site (and PDF) into _site/
```

`quarto render` requires [Quarto](https://quarto.org/) and, for the PDF
output, a LaTeX distribution (`quarto install tinytex` sets one up). Building
`_site/` regenerates every table and the figure from the committed `data/`
files at render time — there is no hand-maintained copy of a result anywhere
in `index.qmd`.

## License

See `LICENSE`.
