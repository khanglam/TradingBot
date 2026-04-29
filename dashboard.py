"""Streamlit dashboard for the autoresearch trading bot.

Launch:
    streamlit run dashboard.py
or double-click run_dashboard.bat on Windows.

Read-only against results.tsv / strategy.py / git, plus action buttons that
shell out to backtest.py and loop.py. Does NOT itself implement the loop —
it's purely a viewer + launcher.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
RESULTS = ROOT / "results.tsv"
STRATEGY = ROOT / "strategy.py"
PROGRAM = ROOT / "program.md"
RUN_LOG = ROOT / "run.log"
DATA_DIR = ROOT / "data"

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

STATUS_COLORS = {
    "keep": "#10b981",
    "discard": "#ef4444",
    "crash": "#f59e0b",
}

st.set_page_config(
    page_title="TradingBot Autoresearch",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────── Loaders ──────────────────────────────

@st.cache_data(ttl=2)
def load_results() -> pd.DataFrame:
    if not RESULTS.exists():
        return pd.DataFrame(
            columns=["commit", "val_sharpe", "max_drawdown", "win_rate",
                     "total_trades", "status", "description"]
        )
    df = pd.read_csv(RESULTS, sep="\t")
    df["val_sharpe"] = pd.to_numeric(df["val_sharpe"], errors="coerce")
    df["max_drawdown"] = pd.to_numeric(df["max_drawdown"], errors="coerce")
    df["win_rate"] = pd.to_numeric(df["win_rate"], errors="coerce")
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce").astype("Int64")
    return df


def run_inline_backtest(window: str = "val") -> dict | None:
    """Run a backtest in-process so we can grab the equity curve.

    The OFFICIAL evaluation is still backtest.py — this is just for viz.
    Both share strategy.py and the same window slicing rules, so metrics match.
    """
    try:
        import importlib
        import backtest as bt_module
        from backtesting import Backtest

        importlib.reload(bt_module)
        sys.path.insert(0, str(ROOT))
        if "strategy" in sys.modules:
            importlib.reload(sys.modules["strategy"])
        from strategy import Strategy as UserStrategy

        df = pd.read_parquet(ROOT / bt_module.DEFAULT_DATA)
        if window == "train":
            df = df.loc[bt_module.TRAIN_START:bt_module.TRAIN_END]
        else:
            df = df.loc[bt_module.VAL_START:bt_module.VAL_END]

        if len(df) < 100:
            return None

        bt = Backtest(
            df, UserStrategy,
            cash=bt_module.STARTING_CASH,
            commission=bt_module.COMMISSION,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        return {
            "equity_curve": stats._equity_curve,
            "trades": stats._trades,
            "n_trades": int(stats.get("# Trades", 0) or 0),
            "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
            "sortino": float(stats.get("Sortino Ratio", 0.0) or 0.0),
            "max_dd": abs(float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)),
            "win_rate": float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0,
            "total_return": float(stats.get("Return [%]", 0.0) or 0.0),
            "buy_and_hold": float(stats.get("Buy & Hold Return [%]", 0.0) or 0.0),
            "data_index": df.index,
            "data_close": df["Close"],
        }
    except Exception as e:
        st.error(f"backtest failed: {e}")
        return None


def git_log_strategy(n: int = 20) -> str:
    try:
        out = subprocess.run(
            ["git", "log", "--oneline", f"-{n}", "--", "strategy.py"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return out or "(no commits yet touching strategy.py)"
    except Exception as e:
        return f"git unavailable: {e}"


def env_status() -> dict[str, bool]:
    return {
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY") or
                                   (ROOT / ".env").exists() and "ANTHROPIC_API_KEY" in (ROOT / ".env").read_text()),
        "BTC data file": (DATA_DIR / "crypto" / "BTC_USDT_4h.parquet").exists(),
        ".venv python": PYTHON.exists(),
        "strategy.py": STRATEGY.exists(),
        "program.md": PROGRAM.exists(),
    }


# ────────────────────────────── Header ───────────────────────────────

st.title("📈 TradingBot Autoresearch")
st.caption("karpathy-autoresearch loop · BTC/USDT 4h · backtesting.py + Claude Sonnet 4.6")

results = load_results()
keeps = results[results["status"] == "keep"]
discards = results[results["status"] == "discard"]
crashes = results[results["status"] == "crash"]

best_sharpe = float(keeps["val_sharpe"].max()) if not keeps.empty else 0.0
latest = results.iloc[-1] if not results.empty else None

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Best Sharpe", f"{best_sharpe:.4f}")
k2.metric("Experiments", len(results))
k3.metric("Keeps", len(keeps), delta=None)
k4.metric("Discards", len(discards), delta=None)
k5.metric("Crashes", len(crashes), delta=None)

if latest is not None:
    badge = STATUS_COLORS.get(str(latest["status"]), "#6b7280")
    st.markdown(
        f"<div style='padding: 0.5rem 0.75rem; border-left: 4px solid {badge}; "
        f"background: rgba(0,0,0,0.03); border-radius: 4px; margin-top: 0.5rem;'>"
        f"<b>Latest:</b> <code>{latest['commit']}</code> · "
        f"sharpe <b>{latest['val_sharpe']:.4f}</b> · "
        f"<span style='color:{badge}'>● {latest['status']}</span> · "
        f"<i>{latest['description']}</i>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ────────────────────────────── Tabs ─────────────────────────────────

tab_dash, tab_exp, tab_strat, tab_run, tab_setup = st.tabs(
    ["📊 Dashboard", "🧪 Experiments", "📝 Strategy", "🚀 Run", "⚙️ Setup"]
)

# ─── Dashboard ───
with tab_dash:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Best-so-far Sharpe progression")
        if keeps.empty:
            st.info("No keeps yet. Launch the loop from the Run tab.")
        else:
            prog = keeps.copy().reset_index(drop=True)
            prog["best_so_far"] = prog["val_sharpe"].cummax()
            prog["experiment"] = range(1, len(prog) + 1)
            st.line_chart(prog.set_index("experiment")[["val_sharpe", "best_so_far"]])

    with right:
        st.subheader("Status mix")
        if results.empty:
            st.info("No experiments yet.")
        else:
            mix = results["status"].value_counts().reindex(["keep", "discard", "crash"]).fillna(0)
            st.bar_chart(mix)

    st.subheader("Current strategy — equity curve (validation 2023-2024)")
    with st.spinner("Running backtest…"):
        viz = run_inline_backtest(window="val")
    if viz is None:
        st.info("Could not produce equity curve. Check data is downloaded and strategy.py is valid.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sharpe", f"{viz['sharpe']:.3f}")
        m2.metric("Sortino", f"{viz['sortino']:.3f}")
        m3.metric("Max DD", f"{viz['max_dd']:.2f}%")
        m4.metric("Trades", viz["n_trades"])
        m5.metric("Return", f"{viz['total_return']:.1f}%",
                  delta=f"vs B&H {viz['buy_and_hold']:.1f}%")

        eq = viz["equity_curve"][["Equity"]].copy()
        eq["Buy & Hold"] = viz["data_close"] * (eq["Equity"].iloc[0] / viz["data_close"].iloc[0])
        eq.rename(columns={"Equity": "Strategy"}, inplace=True)
        st.line_chart(eq)

        st.subheader("Drawdown")
        dd = viz["equity_curve"][["DrawdownPct"]].copy()
        dd["DrawdownPct"] = -dd["DrawdownPct"] * 100
        dd.rename(columns={"DrawdownPct": "Drawdown %"}, inplace=True)
        st.area_chart(dd)

# ─── Experiments ───
with tab_exp:
    st.subheader("All experiments (results.tsv)")
    if results.empty:
        st.info("results.tsv is empty. Run the loop from the Run tab.")
    else:
        status_filter = st.multiselect(
            "Filter status",
            options=["keep", "discard", "crash"],
            default=["keep", "discard", "crash"],
        )
        view = results[results["status"].isin(status_filter)].iloc[::-1].reset_index(drop=True)

        def color_status(val):
            c = STATUS_COLORS.get(val)
            if c:
                return f"background-color: {c}22; color: {c}; font-weight: 600"
            return ""

        styled = view.style.map(color_status, subset=["status"]).format({
            "val_sharpe": "{:.4f}",
            "max_drawdown": "{:.2f}",
            "win_rate": "{:.3f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        st.caption(f"Showing {len(view)} of {len(results)} rows. Newest first.")

# ─── Strategy ───
with tab_strat:
    st.subheader("Current strategy.py")
    if STRATEGY.exists():
        st.code(STRATEGY.read_text(encoding="utf-8"), language="python")
    else:
        st.warning("strategy.py not found")

    st.subheader("Git history (last 20 commits touching strategy.py)")
    st.code(git_log_strategy(20), language="text")

    with st.expander("📜 program.md (agent constraints)"):
        if PROGRAM.exists():
            st.markdown(PROGRAM.read_text(encoding="utf-8"))
        else:
            st.warning("program.md not found")

# ─── Run ───
with tab_run:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("▶ Single backtest")
        st.caption("Runs `python backtest.py` and prints the summary block.")
        window = st.radio("Window", ["val", "train"], horizontal=True, key="bt_window")
        if st.button("Run backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest…"):
                proc = subprocess.run(
                    [str(PYTHON), "backtest.py", "--window", window],
                    cwd=ROOT, capture_output=True, text=True, timeout=300,
                )
            st.code(proc.stdout or "(no stdout)", language="text")
            if proc.returncode != 0 or proc.stderr:
                with st.expander("stderr"):
                    st.code(proc.stderr[-4000:], language="text")
            load_results.clear()

    with colB:
        st.subheader("🚀 Autoresearch loop")
        st.caption("Asks Claude for one mutation per iteration, backtests, keeps or reverts.")
        iters = st.number_input("Iterations", min_value=1, max_value=200, value=5)
        if st.button("Launch loop", type="primary", use_container_width=True):
            log_box = st.empty()
            output_lines: list[str] = []
            try:
                proc = subprocess.Popen(
                    [str(PYTHON), "loop.py", "--iters", str(int(iters))],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline, ""):
                    output_lines.append(line.rstrip())
                    log_box.code("\n".join(output_lines[-40:]), language="text")
                proc.wait()
                if proc.returncode == 0:
                    st.success(f"Loop finished cleanly (exit {proc.returncode}).")
                else:
                    st.warning(f"Loop exited with code {proc.returncode} (likely 3 consecutive regressions or missing API key).")
            except Exception as e:
                st.error(f"Failed to launch loop: {e}")
            load_results.clear()

# ─── Setup ───
with tab_setup:
    st.subheader("Environment check")
    env = env_status()
    for label, ok in env.items():
        st.markdown(f"- {'✅' if ok else '❌'} **{label}**")

    st.subheader("Data files")
    if DATA_DIR.exists():
        files = list(DATA_DIR.rglob("*.parquet"))
        if files:
            st.dataframe(
                pd.DataFrame([
                    {"file": str(f.relative_to(ROOT)),
                     "rows": len(pd.read_parquet(f)),
                     "size_kb": f.stat().st_size // 1024}
                    for f in files
                ]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No data files yet. Run: `python data_fetch.py --asset crypto`")
    else:
        st.info("No data/ directory. Run: `python data_fetch.py --asset crypto`")

    st.subheader("Quick reference")
    st.markdown("""
    **Files**
    - `backtest.py` — fixed harness (DO NOT MODIFY)
    - `strategy.py` — only file the agent edits
    - `loop.py` — autoresearch orchestrator
    - `live_trade.py` — paper / live via LumiBot

    **Setup**
    1. `pip install -r requirements.txt` (or the listed packages)
    2. Put `ANTHROPIC_API_KEY=...` in `.env`
    3. `python data_fetch.py --asset crypto --symbol BTC/USDT --start 2019-01-01 --end 2024-12-31`
    4. Commit current state (loop.py refuses to start on a dirty git tree)
    5. Click **Launch loop** above
    """)
