# BAT Roadmap TODO

## RL (Future)
- [ ] Build a replayable simulation environment with fixed market data slices.
- [ ] Define a stable reward function (PnL - fees - slippage - risk penalty).
- [ ] Start with a minimal action space (BUY/SELL/HOLD + fixed sizing).
- [ ] Add offline evaluation harness (policy vs baseline).

## Model & Data Improvements
- [x] Multi-timeframe features (1m/5m/15m alignment).
- [ ] Multi-horizon targets (3/5/10 bars) with ensemble heads.
- [ ] Probability calibration (temperature scaling/Platt).
- [x] Cost-aware targets (embed fees + slippage into labels).
- [ ] Asymmetric thresholds for BUY/SELL based on risk regime.
- [ ] Stratified training by volatility regime (trend vs range model).
- [ ] Sample curation: overweight high-signal periods.
- [ ] Walk-forward validation and drift monitoring.

## System Improvements
- [ ] Training/report dashboards in TUI (metrics + checkpoints).
- [x] Config presets for risk modes (conservative/normal/aggressive).
- [x] Automated data integrity checks and re-sync.
- [ ] Telegram/Discord Notification integration (trade alerts, daily PnL).
- [ ] Docker containerization (Dockerfile + docker-compose).

## Advanced Execution
- [ ] Trailing Stop-Loss implementation.
- [ ] DCA (Dollar Cost Averaging) logic for losing positions.
- [ ] Smart Order Routing (split large orders to minimize slippage).

## Sentiment & External Data
- [ ] Integrate News Sentiment (e.g., CryptoPanic API).
- [ ] On-chain metric correlation (e.g., exchange inflow/outflow).
