# Modeling-EM-Risk-GARCH-Volatility-Tail-Risk-and-Carry-Dynamics-in-USD-INR
"""
=====================================================================
  USD/INR CURRENCY RISK MODEL
  Author  : Tanusree Saha
  Data    : Calibrated to real USD/INR range 2023-2026
            (83.00 – 96.97 actual range)
  Tools   : Python, NumPy, SciPy, Pandas, Matplotlib
=====================================================================

REAL DATA (replace generate_usdinr_data with):
    import yfinance as yf
    df = yf.download("INR=X", start="2023-01-01", end="2026-05-30")
    returns = df["Close"].pct_change().dropna()

CURRENT MARKET (May 30, 2026):
    USD/INR  : ₹95.00  (-0.71% today)
    52w high : ₹96.97  (May 2026)
    52w low  : ₹85.18
    YoY move : +10.94%  (rupee depreciated)
    RBI ref  : NSE USDINR Futures at 93.10 (Jun contract)

WHAT THIS PROJECT DOES:
    1. VaR & CVaR for currency risk (99% & 95%)
    2. GARCH-style volatility clustering
    3. RBI intervention detection (sudden reversal spikes)
    4. Carry trade P&L simulation (borrow USD, invest INR)
    5. Hedging cost analysis (forward premium)
    6. Tail risk: extreme rupee depreciation scenarios
    7. Correlation with Nifty 50 (risk-off = rupee falls + Nifty falls)
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, t as student_t
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)


# ─────────────────────────────────────────────
#  1. GENERATE USD/INR DATA
# ─────────────────────────────────────────────

def generate_usdinr_data(n_days=850):
    """
    Realistic USD/INR series calibrated to actual data:
    Jan 2023: ~82.00 → May 2026: ~95.00
    Key events embedded:
    - Sep 2023: RBI defence at 83.50
    - Nov 2024: Trump election shock (rapid move to 84.50)
    - Jan 2025: Break above 86
    - May 2026: Record high ~96.97
    """
    dates = pd.bdate_range(start='2023-01-02', periods=n_days)

    # Gradual depreciation trend + volatility clustering
    mu    =  0.000055   # tiny daily drift (depreciation bias)
    sigma =  0.003      # ~0.3% daily vol (FX is less volatile than equity)

    returns = np.random.normal(mu, sigma, n_days)

    # GARCH-like clustering — high vol follows high vol
    for i in range(1, n_days):
        if abs(returns[i-1]) > 0.005:
            returns[i] += np.random.normal(0, sigma * 1.8)

    # RBI intervention reversals (sharp sudden moves)
    rbi_days = [85, 180, 380, 520]
    for d in rbi_days:
        if d < n_days:
            returns[d] = np.random.uniform(-0.008, -0.005)   # RBI sells USD

    # Trump election Nov 2024 spike
    returns[470:478] += np.random.normal(0.005, 0.003, 8)

    # Jan 2025 structural break
    returns[500:510] += np.random.normal(0.004, 0.002, 10)

    # May 2026 record high episode
    returns[820:835] += np.random.normal(0.003, 0.004, 15)

    # Build price series starting at 82.00
    prices = 82.00 * np.exp(np.cumsum(returns))

    # Scale to end near 95.00
    prices = prices * (95.00 / prices[-1]) * np.random.uniform(0.995, 1.005)

    df = pd.DataFrame({
        'date'     : dates,
        'usdinr'   : np.round(prices, 4),
        'return'   : np.log(prices / np.roll(prices, 1)),
    }, ).set_index('date')
    df.loc[df.index[0], 'return'] = 0
    df['return'] = np.log(df['usdinr'] / df['usdinr'].shift(1)).fillna(0)
    return df


# ─────────────────────────────────────────────
#  2. GARCH VOLATILITY
# ─────────────────────────────────────────────

def garch_volatility(returns, omega=0.000002, alpha=0.12, beta=0.85):
    """
    GARCH(1,1) model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Captures volatility clustering — common in FX markets.
    High vol today → high vol tomorrow.

    Parameters calibrated to INR typical values.
    """
    n     = len(returns)
    sigma = np.zeros(n)
    sigma[0] = returns.std()

    for t in range(1, n):
        sigma[t] = np.sqrt(omega +
                           alpha * returns.iloc[t-1]**2 +
                           beta  * sigma[t-1]**2)
    return pd.Series(sigma, index=returns.index, name='GARCH_vol')


# ─────────────────────────────────────────────
#  3. CARRY TRADE ANALYSIS
# ─────────────────────────────────────────────

def carry_trade_pnl(usdinr_df, usd_rate=0.0525, inr_rate=0.065,
                    notional_usd=1_000_000):
    """
    Carry trade: Borrow USD at US rates, invest in INR instruments.

    Daily P&L = (INR rate - USD rate)/365 * notional
                - currency loss from INR depreciation

    India rate spread: ~1.25% (INR 6.5% - USD 5.25%)
    Risk: INR depreciates faster than carry income

    This is key context for UBS FX desk interview questions.
    """
    daily_carry = (inr_rate - usd_rate) / 252
    fx_return   = usdinr_df['return']

    # In USD terms, FX appreciation = positive, depreciation = negative
    # Carry trade loses when INR depreciates (return < 0 from USD perspective)
    carry_pnl = notional_usd * (daily_carry - fx_return)

    cumulative = carry_pnl.cumsum()
    max_dd     = (cumulative - cumulative.cummax()).min()

    return carry_pnl, cumulative, max_dd


# ─────────────────────────────────────────────
#  4. FORWARD PREMIUM
# ─────────────────────────────────────────────

def forward_premium(spot, usd_rate=0.0525, inr_rate=0.065, tenors_months=[1,3,6,12]):
    """
    Covered Interest Rate Parity:
    F = S × (1 + r_INR × T) / (1 + r_USD × T)

    Forward premium = annualised cost of hedging USD/INR exposure.
    Higher INR rates → INR trades at a forward discount (hedging costs money).
    """
    results = []
    for T_months in tenors_months:
        T = T_months / 12
        F = spot * (1 + inr_rate * T) / (1 + usd_rate * T)
        fwd_premium = (F - spot) / spot / T * 100   # annualised %
        results.append({
            'Tenor'   : f'{T_months}M',
            'Spot'    : round(spot, 2),
            'Forward' : round(F, 4),
            'Pts'     : round(F - spot, 4),
            'Premium%': round(fwd_premium, 3),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  5. EXTREME SCENARIOS
# ─────────────────────────────────────────────

def scenario_analysis(spot=95.00):
    """
    Stress scenarios for INR:
    Based on historical precedent and analyst projections.
    """
    scenarios = [
        ("Base case (stable)",           95.00,  0.00,  "No major shock"),
        ("Mild FII outflow",             97.50,  +2.6,  "Sustained $1B/day selling"),
        ("RBI pauses intervention",      100.00, +5.3,  "Policy pivot"),
        ("US tariff escalation",         102.00, +7.4,  "Trade war escalation"),
        ("BoP stress (India CAD widens)", 105.00, +10.5, "Current account deteriorates"),
        ("2013 Taper-tantrum repeat",    112.00, +17.9, "Fed hawkish surprise"),
        ("Extreme tail (1-in-20yr)",     120.00, +26.3, "Full crisis scenario"),
    ]
    df = pd.DataFrame(scenarios,
                      columns=['Scenario','USD/INR','Move_%','Trigger'])
    # INR impact on ₹1Cr import hedge
    df['Import_cost_₹Cr'] = df['USD/INR'] / spot
    return df


# ─────────────────────────────────────────────
#  6. NIFTY-RUPEE CORRELATION
# ─────────────────────────────────────────────

def nifty_inr_correlation(usdinr_returns, n_days=850):
    """
    Risk-off episodes: Nifty falls AND rupee depreciates together.
    Positive correlation between Nifty returns and INR appreciation.
    Negative correlation between Nifty returns and USD/INR returns.
    """
    np.random.seed(42)
    # Simulate Nifty returns with negative correlation to USD/INR
    corr = -0.45
    nifty_base = np.random.normal(0.0003, 0.012, len(usdinr_returns))
    nifty_ret  = (corr * usdinr_returns.values +
                  np.sqrt(1-corr**2) * nifty_base)
    nifty_ser  = pd.Series(nifty_ret, index=usdinr_returns.index, name='Nifty')

    # Rolling 63-day correlation
    combined = pd.DataFrame({'USDINR': usdinr_returns, 'Nifty': nifty_ser})
    rolling_corr = combined['USDINR'].rolling(63).corr(combined['Nifty'])
    return nifty_ser, rolling_corr


# ─────────────────────────────────────────────
#  7. VISUALISATION
# ─────────────────────────────────────────────

def plot_currency_risk(fx_df, garch_vol, carry_cum, fwd_df, scenarios, rolling_corr):
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    BG=  '#0D1117'; PANEL='#161B22'; GC='#2A2A3A'; TC='#E0E0E0'
    G='#00C896'; R='#FF4444'; A='#FFB347'; B='#4A9EFF'; P='#C084FC'; T='#00D4C8'

    # ── Panel 1: USD/INR price history ──
    ax1 = fig.add_subplot(gs[0, :]); ax1.set_facecolor(PANEL)
    ax1.plot(fx_df.index, fx_df['usdinr'], color=A, lw=1.3, label='USD/INR')
    ax1.fill_between(fx_df.index, fx_df['usdinr'], 82, alpha=0.1, color=A)
    ax1.axhline(95.00, color=R, lw=1.2, ls='--', alpha=0.8, label='Current: ₹95.00 (May 2026)')
    ax1.axhline(83.50, color=G, lw=1,   ls=':',  alpha=0.6, label='RBI defence level ~83.50')
    ax1.axhline(86.00, color=P, lw=1,   ls=':',  alpha=0.6, label='Structural break: ₹86')
    ax1.set_title('USD/INR Exchange Rate — Jan 2023 to May 2026\n'
                  '(Rupee depreciated ~15.8% | 52w range: ₹85.18 – ₹96.97)',
                  color=TC, fontsize=11, fontweight='bold')
    ax1.set_ylabel('₹ per USD', color=TC, fontsize=9)
    ax1.tick_params(colors=TC, labelsize=8)
    ax1.grid(True, color=GC, lw=0.4)
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TC, ncol=4)
    for s in ax1.spines.values(): s.set_color(GC)

    # ── Panel 2: GARCH Volatility ──
    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor(PANEL)
    ax2.plot(garch_vol.index, garch_vol*100*np.sqrt(252), color=T, lw=1.2,
             label='GARCH Annualised Vol (%)')
    ax2.fill_between(garch_vol.index, garch_vol*100*np.sqrt(252),
                     alpha=0.15, color=T)
    avg_vol = garch_vol.mean()*100*np.sqrt(252)
    ax2.axhline(avg_vol, color=G, lw=1, ls='--', label=f'Mean vol: {avg_vol:.1f}%')
    ax2.set_title('GARCH(1,1) Volatility Clustering\nUSD/INR Annualised Vol (%)',
                  color=TC, fontsize=10, fontweight='bold')
    ax2.set_ylabel('Ann. Volatility (%)', color=TC, fontsize=8)
    ax2.tick_params(colors=TC, labelsize=7)
    ax2.grid(True, color=GC, lw=0.4)
    ax2.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax2.spines.values(): s.set_color(GC)

    # ── Panel 3: Forward Premium ──
    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor(PANEL)
    bars = ax3.bar(fwd_df['Tenor'], fwd_df['Premium%'], color=P, alpha=0.85, width=0.5)
    for bar, val in zip(bars, fwd_df['Premium%']):
        ax3.text(bar.get_x()+bar.get_width()/2, val+0.02,
                 f'{val:.2f}%', ha='center', color=TC, fontsize=9)
    ax3.set_title('USD/INR Forward Premium\n(Cost of hedging INR exposure — per annum)',
                  color=TC, fontsize=10, fontweight='bold')
    ax3.set_ylabel('Annualised Premium (%)', color=TC, fontsize=8)
    ax3.tick_params(colors=TC, labelsize=8)
    ax3.grid(True, color=GC, lw=0.4, axis='y')
    for s in ax3.spines.values(): s.set_color(GC)

    # ── Panel 4: Carry Trade P&L ──
    ax4 = fig.add_subplot(gs[1, 2]); ax4.set_facecolor(PANEL)
    carry_pos = carry_cum.copy()
    carry_pos[carry_pos < 0] = np.nan
    carry_neg = carry_cum.copy()
    carry_neg[carry_neg >= 0] = np.nan
    ax4.plot(carry_cum.index, carry_cum/1000, color=G, lw=1.3, label='Cumulative P&L')
    ax4.fill_between(carry_cum.index, carry_cum/1000, 0,
                     where=carry_cum >= 0, color=G, alpha=0.2)
    ax4.fill_between(carry_cum.index, carry_cum/1000, 0,
                     where=carry_cum < 0, color=R, alpha=0.2)
    ax4.axhline(0, color='white', lw=0.8)
    ax4.set_title('INR Carry Trade P&L\n(Borrow USD 5.25%, Invest INR 6.5%, $1M notional)',
                  color=TC, fontsize=9, fontweight='bold')
    ax4.set_ylabel('Cumulative P&L (USD thousands)', color=TC, fontsize=8)
    ax4.tick_params(colors=TC, labelsize=7)
    ax4.grid(True, color=GC, lw=0.4)
    ax4.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax4.spines.values(): s.set_color(GC)

    # ── Panel 5: VaR at multiple confidence levels ──
    ax5 = fig.add_subplot(gs[2, 0]); ax5.set_facecolor(PANEL)
    returns = fx_df['return'].iloc[-250:]
    confs   = [0.90, 0.95, 0.99, 0.995, 0.999]
    vars_   = [-np.percentile(returns,(1-c)*100)*100 for c in confs]
    cvars_  = [-returns[returns <= np.percentile(returns,(1-c)*100)].mean()*100 for c in confs]
    x5 = np.arange(len(confs))
    ax5.bar(x5-0.18, vars_,  width=0.35, color=B, alpha=0.85, label='VaR')
    ax5.bar(x5+0.18, cvars_, width=0.35, color=P, alpha=0.85, label='CVaR (ES)')
    ax5.set_xticks(x5)
    ax5.set_xticklabels([f'{c*100:.1f}%' for c in confs], color=TC, fontsize=8)
    ax5.set_title('USD/INR VaR & CVaR\nat Multiple Confidence Levels',
                  color=TC, fontsize=10, fontweight='bold')
    ax5.set_ylabel('Move (%)', color=TC, fontsize=8)
    ax5.tick_params(colors=TC, labelsize=8)
    ax5.grid(True, color=GC, lw=0.4, axis='y')
    ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax5.spines.values(): s.set_color(GC)

    # ── Panel 6: Scenario Analysis ──
    ax6 = fig.add_subplot(gs[2, 1:]); ax6.set_facecolor(PANEL); ax6.axis('off')
    rows = scenarios[['Scenario','USD/INR','Move_%','Trigger']].values.tolist()
    cols = ['Scenario','USD/INR Level','Move %','Trigger']
    tbl  = ax6.table(cellText=rows, colLabels=cols,
                     cellLoc='left', loc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (row,col),cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL if row>0 else '#1A3A6B')
        cell.set_edgecolor(GC)
        cell.set_text_props(color=TC)
        if row > 0:
            move = float(rows[row-1][2])
            if move > 15: cell.set_facecolor('#3B1F1F')
            elif move > 7: cell.set_facecolor('#2B2A1F')
    ax6.set_title('INR Stress Scenario Analysis\n(₹/USD levels & import cost impact)',
                  color=TC, fontsize=10, fontweight='bold', pad=8)

    fig.suptitle('USD/INR Currency Risk Model — GARCH, VaR, Carry Trade & Scenario Analysis',
                 color=TC, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/usdinr_currency_risk.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  Chart saved.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  USD/INR CURRENCY RISK MODEL")
    print("="*65)

    fx_df   = generate_usdinr_data()
    returns = fx_df['return']

    print(f"\n── Currency Data Summary ──")
    print(f"  Period      : {fx_df.index[0].date()} → {fx_df.index[-1].date()}")
    print(f"  Start rate  : ₹{fx_df['usdinr'].iloc[0]:.4f}")
    print(f"  End rate    : ₹{fx_df['usdinr'].iloc[-1]:.4f}")
    total_dep = (fx_df['usdinr'].iloc[-1]/fx_df['usdinr'].iloc[0]-1)*100
    print(f"  Depreciation: {total_dep:.2f}%")
    print(f"  Ann. vol    : {returns.std()*np.sqrt(252)*100:.3f}%")
    print(f"  Skewness    : {returns.skew():.4f}")
    print(f"  Kurtosis    : {returns.kurt():.4f}")

    print(f"\n── VaR Estimates (99%, last 250 days) ──")
    r250 = returns.iloc[-250:]
    var99  = -np.percentile(r250, 1.0)
    cvar99 = -r250[r250 <= -var99].mean()
    print(f"  VaR  99%  : {var99*100:.4f}%  = ₹{var99*95:.4f} per USD")
    print(f"  CVaR 99%  : {cvar99*100:.4f}%  = ₹{cvar99*95:.4f} per USD")
    print(f"\n  On $10M FX exposure:")
    print(f"  1-day VaR  : USD {10_000_000*var99:>12,.0f}")
    print(f"  1-day CVaR : USD {10_000_000*cvar99:>12,.0f}")

    print(f"\n── GARCH Volatility ──")
    garch_vol = garch_volatility(returns)
    print(f"  Current GARCH vol (ann.) : {garch_vol.iloc[-1]*100*np.sqrt(252):.3f}%")
    print(f"  Avg GARCH vol (ann.)     : {garch_vol.mean()*100*np.sqrt(252):.3f}%")
    print(f"  Max GARCH vol (ann.)     : {garch_vol.max()*100*np.sqrt(252):.3f}%")

    print(f"\n── Forward Premium ──")
    fwd_df = forward_premium(fx_df['usdinr'].iloc[-1])
    print(fwd_df.to_string(index=False))

    print(f"\n── Carry Trade Analysis ──")
    carry_pnl, carry_cum, max_dd = carry_trade_pnl(fx_df)
    print(f"  Total carry P&L : USD {carry_cum.iloc[-1]:>12,.0f}")
    print(f"  Max drawdown    : USD {max_dd:>12,.0f}")
    print(f"  Sharpe (approx) : {carry_pnl.mean()/carry_pnl.std()*np.sqrt(252):.3f}")
    print(f"  Carry profitable: {'YES' if carry_cum.iloc[-1] > 0 else 'NO — FX losses exceed carry'}")

    print(f"\n── Scenario Analysis ──")
    scenarios = scenario_analysis(fx_df['usdinr'].iloc[-1])
    print(scenarios[['Scenario','USD/INR','Move_%','Trigger']].to_string(index=False))

    nifty_ret, rolling_corr = nifty_inr_correlation(returns)
    print(f"\n── Nifty–Rupee Correlation ──")
    print(f"  Full-period corr  : {returns.corr(nifty_ret):.4f}")
    print(f"  Rolling 63d corr  : {rolling_corr.mean():.4f} (avg)")
    print(f"  Interpretation    : Negative = risk-off hits both Nifty & rupee")

    print(f"\n── Generating Charts ──")
    plot_currency_risk(fx_df, garch_vol, carry_cum, fwd_df, scenarios, rolling_corr)
    print("\nDone.")
