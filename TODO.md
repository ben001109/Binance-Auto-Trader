# BAT Roadmap TODO

## RL (Future)
- [ ] Build a replayable simulation environment with fixed market data slices.
- [ ] Define a stable reward function (PnL - fees - slippage - risk penalty).
- [ ] Start with a minimal action space (BUY/SELL/HOLD + fixed sizing).
- [ ] Add offline evaluation harness (policy vs baseline).

## Model & Data Improvements
- [ ] Multi-timeframe features (1m/5m/15m alignment).
- [ ] Multi-horizon targets (3/5/10 bars) with ensemble heads.
- [ ] Probability calibration (temperature scaling/Platt).
- [ ] Cost-aware targets (embed fees + slippage into labels).
- [ ] Asymmetric thresholds for BUY/SELL based on risk regime.
- [ ] Stratified training by volatility regime (trend vs range model).
- [ ] Sample curation: overweight high-signal periods.
- [ ] Walk-forward validation and drift monitoring.

## System Improvements
- [ ] Training/report dashboards in TUI (metrics + checkpoints).
- [ ] Config presets for risk modes (conservative/normal/aggressive).
- [ ] Automated data integrity checks and re-sync.
