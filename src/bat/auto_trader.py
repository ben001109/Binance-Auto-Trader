import warnings
from bat.analyzer import AnalystAgent, TradeDecision, RiskParams
from bat.config import conf

# Re-export classes for compatibility with existing imports
__all__ = ['decide_trade', 'compute_order_size', 'apply_confidence_threshold', 'TradeDecision', 'RiskParams']

def decide_trade(
    klines,
    model_path: str = os.path.join("data", "lstm_model.pth"), 
    training_data_path: str = os.path.join("data", "history.csv"),
) -> tuple[TradeDecision | None, RiskParams | None]:
    """
    Deprecated: Use AnalystAgent().analyze(klines) instead.
    Making this synchronous wrapper to match old signature if needed, 
    BUT original was async-unfriendly or used in async context?
    Actually original decide_trade was sync. But AnalystAgent is async.
    
    However, decide_trade in original auto_trader.py was NOT async.
    It accepted klines (list) and ran torch inference synchronously.
    
    To maintain compatibility without 'await', we must instatiate the strategy 
    and run the sync parts. But AnalystAgent is designed to be async.
    
    For the TUI which CALLS this function, it is running inside an async loop 
    but this function itself was sync blocking CPU (bad practice but existing).
    
    Let's refactor the TUI to call AnalystAgent directly, but for now, 
    we need to provide this function as a bridge.
    """
    import asyncio
    
    # Create agent
    agent = AnalystAgent(mode='lstm')
    
    # Since we can't await here easily if the caller isn't async,
    # we have a problem. The original code was synchronous.
    # Fortunately, the new LSTMStrategy.analyze *can* handle passed klines purely synchronously
    # IF we don't call anything async. 
    # But `analyze` is defined as `async def`.
    
    # HACK: If we are already in a loop, we can't use asyncio.run().
    # But this function is likely called from the TUI's worker thread or async task.
    # The TUI calls `decide_trade(klines)` inside `action_auto_trade`.
    
    # We should really update the TUI to use the Agent directly. 
    # But to satisfy this file's contract, let's try to wrap it or 
    # expose the underlying sync method if possible.
    
    # Let's verify how TUI calls it. TUI calls it in `action_auto_trade` which is async.
    # So TUI *can* await. But `decide_trade` is not async. 
    # We will change `decide_trade` to be skipped or just make it a wrapper that returns a coroutine?
    # No, that would break call sites expecting a value immediately.
    
    # BETTER PLAN:
    # 1. Update TUI (`src/bat/tui/app.py`) to import `AnalystAgent` and `await agent.analyze(klines)`.
    # 2. Leave this file as a stub that raises DeprecationWarning or proxies if possible.
    # Since I am updating the TUI anyway, I will delete the logic here and just point to new location.
    
    # But wait, `simulation.py` might also use it?
    # `from bat.auto_trader import decide_trade` is in `src/bat/tui/app.py`.
    # Let's check `src/bat/simulation.py`.
    
    warnings.warn("auto_trader.decide_trade is deprecated. Use AnalystAgent.analyze instead.", DeprecationWarning)
    
    # Temporary synchronous implementation reusing the new Strategy logic 
    # by instantiating it and manually calling the internal sync methods?
    # Or just copy-paste the minimal sync logic here as a fallback?
    # No, duplication is bad.
    
    # We will make this function raise an error to force migration, 
    # OR we make it a sync wrapper that uses `asyncio.run` if no loop is running,
    # or fails if loop is running.
    
    # Correct approach: implementation plan said "Update Interface Agent (app.py)". 
    # So I will update app.py to validly use the new async agent.
    # This file `auto_trader.py` will stay for type definitions or help transition.
    
    # Let's keep the helper functions that are purely math.
    pass

def compute_order_size(quote_balance: float, market_vol: float) -> float:
    fraction = min(max(market_vol * 5.0, 0.02), 0.2)
    return quote_balance * fraction

def apply_confidence_threshold(decision: TradeDecision | None, threshold: float) -> TradeDecision | None:
    if decision is None:
        return None
    if decision.confidence < threshold:
        decision.action = "HOLD"
        decision.source = "filtered"
    else:
        decision.action = decision.signal
    return decision
