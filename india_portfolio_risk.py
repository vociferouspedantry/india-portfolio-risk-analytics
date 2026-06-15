"""
India Portfolio Risk Analytics  — v2
20-Stock Equal-Weight Portfolio · NSE · 5-Year Monte Carlo Simulation
Multi-page PDF: standard + novel risk insights.
Author: Sahej Verma
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, gaussian_kde, skew, kurtosis
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings, io
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0a0c0f"
SURFACE = "#111318"
BORDER  = "#1e2330"
TEXT    = "#e8eaf0"
MUTED   = "#6b7280"
ACCENT  = "#f0b429"
ACCENT2 = "#e05c3a"
ACCENT3 = "#3ab8e0"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
PURPLE  = "#b06fe0"

SECTOR_COLORS = {
    "IT":         "#3ab8e0",
    "Financials": "#f0b429",
    "Energy":     "#e05c3a",
    "FMCG":       "#2ecc71",
    "Healthcare": "#b06fe0",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "font.size":        8,
})

# ── Universe ──────────────────────────────────────────────────────────────────
STOCKS = [
    ("TCS",        "Tata Consultancy",     "IT",         0.148, 0.22),
    ("INFY",       "Infosys",              "IT",         0.132, 0.24),
    ("WIPRO",      "Wipro",                "IT",         0.095, 0.26),
    ("HCLTECH",    "HCL Technologies",     "IT",         0.162, 0.25),
    ("HDFCBANK",   "HDFC Bank",            "Financials", 0.141, 0.23),
    ("ICICIBANK",  "ICICI Bank",           "Financials", 0.178, 0.26),
    ("KOTAKBANK",  "Kotak Mahindra",       "Financials", 0.128, 0.22),
    ("AXISBANK",   "Axis Bank",            "Financials", 0.115, 0.29),
    ("RELIANCE",   "Reliance Industries",  "Energy",     0.165, 0.24),
    ("ONGC",       "ONGC",                 "Energy",     0.072, 0.27),
    ("NTPC",       "NTPC",                 "Energy",     0.088, 0.21),
    ("POWERGRID",  "Power Grid Corp",      "Energy",     0.096, 0.19),
    ("HINDUNILVR", "Hindustan Unilever",   "FMCG",       0.112, 0.17),
    ("ITC",        "ITC Limited",          "FMCG",       0.098, 0.16),
    ("NESTLEIND",  "Nestle India",         "FMCG",       0.124, 0.18),
    ("BRITANNIA",  "Britannia Industries", "FMCG",       0.135, 0.19),
    ("SUNPHARMA",  "Sun Pharma",           "Healthcare", 0.145, 0.25),
    ("DRREDDY",    "Dr. Reddy's Labs",     "Healthcare", 0.118, 0.23),
    ("CIPLA",      "Cipla",                "Healthcare", 0.107, 0.22),
    ("DIVISLAB",   "Divi's Laboratories",  "Healthcare", 0.158, 0.28),
]

tickers  = [s[0] for s in STOCKS]
names    = [s[1] for s in STOCKS]
sectors  = [s[2] for s in STOCKS]
ann_rets = np.array([s[3] for s in STOCKS])
ann_vols = np.array([s[4] for s in STOCKS])
N        = len(STOCKS)
DAYS     = 1260
DT       = 1/252
RF       = 0.065
np.random.seed(42)
sector_list = list(dict.fromkeys(sectors))

# ── Simulate ──────────────────────────────────────────────────────────────────
def build_corr():
    C = np.eye(N)
    for i in range(N):
        for j in range(i+1, N):
            rho = 0.62 if sectors[i] == sectors[j] else 0.28
            rho += np.random.uniform(-0.06, 0.06)
            C[i, j] = C[j, i] = rho
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-6, None)
    C = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)

corr_matrix  = build_corr()
L_chol       = np.linalg.cholesky(corr_matrix)
daily_vols   = ann_vols * np.sqrt(DT)
daily_drifts = (ann_rets - 0.5 * ann_vols**2) * DT

Z        = np.random.randn(DAYS, N)
log_rets = daily_drifts + daily_vols * (Z @ L_chol.T)
pct_rets = np.expm1(log_rets)

w            = np.full(N, 1/N)
port_log     = log_rets @ w
port_pct     = pct_rets @ w
cum_log      = np.concatenate([[0], np.cumsum(port_log)])
cum_values   = np.exp(cum_log)

rolling_max  = np.maximum.accumulate(cum_values)
drawdown     = (cum_values - rolling_max) / rolling_max * 100
max_dd       = drawdown.min()
max_dd_idx   = np.argmin(drawdown)

ann_ret_port = (cum_values[-1] ** (252/DAYS) - 1) * 100
ann_vol_port = port_pct.std() * np.sqrt(252) * 100
sharpe       = (ann_ret_port - RF*100) / ann_vol_port

cov          = np.cov(log_rets.T) * 252
port_var     = float(w @ cov @ w)
port_vol_ann = float(np.sqrt(port_var))
mrc          = cov @ w
rc           = w * mrc
rc_pct       = rc / port_var * 100

stock_ann_rets_sim = (np.exp(log_rets.sum(axis=0)) ** (252/DAYS) - 1) * 100
stock_ann_vols_sim = pct_rets.std(axis=0) * np.sqrt(252) * 100
stock_sharpes      = (stock_ann_rets_sim - RF*100) / stock_ann_vols_sim
stock_betas        = np.array([
    np.cov(pct_rets[:,i], port_pct)[0,1] / port_pct.var() for i in range(N)])

def var_cvar(returns, c=0.95):
    s  = np.sort(returns)
    ix = int((1-c)*len(s))
    return -s[ix]*100, -s[:ix].mean()*100

var_95,  cvar_95  = var_cvar(port_pct, 0.95)
var_99,  cvar_99  = var_cvar(port_pct, 0.99)
stock_var95  = np.array([var_cvar(pct_rets[:,i])[0] for i in range(N)])
stock_cvar95 = np.array([var_cvar(pct_rets[:,i])[1] for i in range(N)])

ROLL        = 63
roll_vol    = np.array([port_pct[i-ROLL:i].std()*np.sqrt(252)*100
                         for i in range(ROLL, len(port_pct)+1)])
roll_ret    = np.array([port_pct[i-ROLL:i].mean()*252*100
                         for i in range(ROLL, len(port_pct)+1)])
roll_sr     = (roll_ret - RF*100) / roll_vol

roll_vol_all = np.array([port_pct[max(0,i-21):i].std()*np.sqrt(252)*100
                           for i in range(1, len(port_pct)+1)])
vol_q33 = np.percentile(roll_vol_all, 33)
vol_q66 = np.percentile(roll_vol_all, 66)
regimes = np.where(roll_vol_all < vol_q33, 0,
          np.where(roll_vol_all < vol_q66, 1, 2))

div_ratio   = (w * np.sqrt(np.diag(cov))).sum() / port_vol_ann
calmar      = ann_ret_port / abs(max_dd)
emp_corr    = np.corrcoef(log_rets.T)

# Efficient frontier cloud
np.random.seed(99)
n_sim = 600
ef_rets, ef_vols, ef_srs = [], [], []
for _ in range(n_sim):
    ww = np.random.dirichlet(np.ones(N))
    r  = float((ww * stock_ann_rets_sim).sum())
    v  = float(np.sqrt(ww @ cov @ ww) * 100)
    ef_rets.append(r); ef_vols.append(v); ef_srs.append((r-RF*100)/v)
ef_rets = np.array(ef_rets)
ef_vols = np.array(ef_vols)
ef_srs  = np.array(ef_srs)

# Sector aggregates
sector_rc = {s: rc_pct[np.array(sectors)==s].sum() for s in sector_list}
sector_wt = {s: (np.array(sectors)==s).mean()*100 for s in sector_list}

# Higher-moment stats
port_skew = skew(port_pct)
port_kurt = kurtosis(port_pct)  # excess kurtosis

# ── Rolling correlation (IT vs Financials) — a novel insight ─────────────────
it_idx  = [i for i,s in enumerate(sectors) if s=="IT"]
fin_idx = [i for i,s in enumerate(sectors) if s=="Financials"]
it_ret  = pct_rets[:,it_idx].mean(axis=1)
fin_ret = pct_rets[:,fin_idx].mean(axis=1)
ROLL2   = 42
roll_corr_if = np.array([
    np.corrcoef(it_ret[i-ROLL2:i], fin_ret[i-ROLL2:i])[0,1]
    for i in range(ROLL2, len(it_ret)+1)
])

# ── Omega Ratio ───────────────────────────────────────────────────────────────
threshold = 0.0
omega = port_pct[port_pct > threshold].sum() / abs(port_pct[port_pct < threshold].sum())

# ── Pain Index & Ulcer Index ──────────────────────────────────────────────────
pain_index  = np.abs(drawdown).mean()
ulcer_index = np.sqrt((drawdown**2).mean())

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 13, 8.5
FIGS = []

def save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    buf.seek(0)
    FIGS.append(buf)
    plt.close(fig)

def stitle(ax, txt):
    ax.text(0, 1.03, txt, transform=ax.transAxes,
            fontsize=8, color=ACCENT, fontweight='bold', va='bottom')

def note(ax, txt, x=0.01, y=0.96):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=6, color=MUTED, va='top')

dates = np.arange(len(cum_values))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Standard Overview (4 panels)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30)

# P1A — Cumulative return (regime-coloured segments)
ax = fig.add_subplot(gs[0, 0])
regime_colors = [GREEN, ACCENT, ACCENT2]
for i in range(len(cum_values)-1):
    r = regimes[min(i, len(regimes)-1)]
    ax.plot([dates[i], dates[i+1]], [cum_values[i], cum_values[i+1]],
            color=regime_colors[r], lw=1.2, solid_capstyle='round')
stitle(ax, "CUMULATIVE RETURN  (₹1 invested)")
note(ax, "Colour = volatility regime: green=calm  amber=normal  red=stress")
ax.set_ylabel("Portfolio Value (₹)", color=MUTED)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.set_xticks([0,252,504,756,1008,1260])
ax.set_xticklabels(['Y0','Y1','Y2','Y3','Y4','Y5'])
ax.grid(True, alpha=0.4)
ax.annotate(f"₹{cum_values[-1]:.2f}", xy=(dates[-1], cum_values[-1]),
            xytext=(-50, -16), textcoords='offset points',
            fontsize=8, color=ACCENT, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1))

# P1B — Drawdown
ax = fig.add_subplot(gs[0, 1])
ax.fill_between(dates, drawdown, 0, color=ACCENT2, alpha=0.4)
ax.plot(dates, drawdown, color=ACCENT2, lw=0.9)
ax.axhline(max_dd, color=RED, lw=1, ls='--', alpha=0.7)
ax.text(max_dd_idx+20, max_dd+0.3, f"Max DD {max_dd:.1f}%", color=RED, fontsize=7)
stitle(ax, "PORTFOLIO DRAWDOWN (%)")
ax.set_xticks([0,252,504,756,1008,1260])
ax.set_xticklabels(['Y0','Y1','Y2','Y3','Y4','Y5'])
ax.set_ylabel("Drawdown (%)", color=MUTED)
ax.grid(True, alpha=0.4)

# P1C — Risk contribution bars
ax = fig.add_subplot(gs[1, 0])
sort_idx  = np.argsort(rc_pct)[::-1]
bar_cols  = [SECTOR_COLORS[sectors[i]] for i in sort_idx]
ax.barh(range(N), rc_pct[sort_idx], color=bar_cols, height=0.65, alpha=0.85)
ax.axvline(5.0, color=ACCENT, lw=1.2, ls='--', alpha=0.8, label='Equal weight 5%')
ax.set_yticks(range(N))
ax.set_yticklabels([tickers[i] for i in sort_idx], fontsize=6)
ax.set_xlabel("Risk Contribution (%)", color=MUTED)
stitle(ax, "RISK CONTRIBUTION vs CAPITAL WEIGHT")
ax.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, axis='x', alpha=0.4)

# P1D — Rolling 63-day Sharpe
ax = fig.add_subplot(gs[1, 1])
rx = np.arange(ROLL, ROLL+len(roll_sr))
ax.fill_between(rx, roll_sr, 0, where=roll_sr>=0, color=GREEN, alpha=0.35)
ax.fill_between(rx, roll_sr, 0, where=roll_sr<0,  color=ACCENT2, alpha=0.35)
ax.plot(rx, roll_sr, color=TEXT, lw=0.8)
ax.axhline(0,      color=BORDER, lw=1)
ax.axhline(sharpe, color=ACCENT, lw=1, ls='--', alpha=0.7)
ax.text(rx[-1]-130, sharpe+0.05, f"Full-period SR {sharpe:.2f}",
        color=ACCENT, fontsize=7)
stitle(ax, "ROLLING 63-DAY SHARPE RATIO")
ax.set_xticks([ROLL,252,504,756,1008,1260])
ax.set_xticklabels(['','Y1','Y2','Y3','Y4','Y5'])
ax.set_ylabel("Sharpe Ratio", color=MUTED)
ax.grid(True, alpha=0.4)

patches = [mpatches.Patch(color=SECTOR_COLORS[s], label=s) for s in sector_list]
fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=7,
           facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("INDIA PORTFOLIO RISK ANALYTICS  ·  20-Stock  ·  5Y Monte Carlo  ·  Equal Weight",
             fontsize=11, color=ACCENT, fontweight='bold', y=1.01)
save_fig(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Novel Insight Set A: Regime · Frontier · Scatter · Sector Divergence
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.36)

# P2A — Return distribution by regime
ax = fig.add_subplot(gs[0, 0])
for ri, (rname, rcol) in enumerate(zip(['Calm','Normal','Stress'],
                                        [GREEN, ACCENT, ACCENT2])):
    mask = regimes == ri
    data = port_pct[mask]
    if len(data) > 5:
        kde  = gaussian_kde(data, bw_method=0.4)
        xg   = np.linspace(port_pct.min(), port_pct.max(), 300)
        ax.plot(xg*100, kde(xg)/100, color=rcol, lw=1.8,
                label=f'{rname} ({mask.sum()}d)')
mu_p, sig_p = port_pct.mean(), port_pct.std()
xn = np.linspace(mu_p - 4*sig_p, mu_p + 4*sig_p, 300)
ax.plot(xn*100, norm.pdf(xn, mu_p, sig_p)/100,
        color=TEXT, lw=0.9, ls='--', alpha=0.5, label='Normal fit')
ax.axvline(-var_95, color=RED, lw=1, ls=':', alpha=0.8)
ax.text(-var_95-0.05, ax.get_ylim()[1]*0.7 if ax.get_ylim()[1]>0 else 1,
        f'VaR95\n{var_95:.2f}%', color=RED, fontsize=6, ha='right')
stitle(ax, "RETURN DISTRIBUTION BY REGIME")
ax.set_xlabel("Daily Return (%)", color=MUTED)
ax.legend(fontsize=5.8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, alpha=0.3)

# P2B — Risk/Return scatter coloured by Sharpe
ax = fig.add_subplot(gs[0, 1])
sc = ax.scatter(stock_ann_vols_sim, stock_ann_rets_sim,
                c=stock_sharpes, cmap='RdYlGn', s=60, zorder=5,
                vmin=stock_sharpes.min(), vmax=stock_sharpes.max())
for i, t in enumerate(tickers):
    ax.annotate(t, (stock_ann_vols_sim[i], stock_ann_rets_sim[i]),
                fontsize=5.5, color=MUTED, xytext=(3,3), textcoords='offset points')
ax.scatter([ann_vol_port*100], [ann_ret_port], marker='*',
           color=ACCENT, s=180, zorder=10, label='Portfolio')
plt.colorbar(sc, ax=ax, label='Sharpe', pad=0.02)
ax.set_xlabel("Ann. Vol (%)", color=MUTED)
ax.set_ylabel("Ann. Return (%)", color=MUTED)
stitle(ax, "RISK / RETURN SCATTER  (colour = Sharpe)")
ax.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, alpha=0.3)

# P2C — Efficient frontier cloud
ax = fig.add_subplot(gs[0, 2])
sc2 = ax.scatter(ef_vols, ef_rets, c=ef_srs, cmap='plasma', s=12, alpha=0.55)
ax.scatter([ann_vol_port*100], [ann_ret_port], marker='*',
           color=ACCENT, s=220, zorder=10, label='Equal-weight')
bi = np.argmax(ef_srs)
ax.scatter([ef_vols[bi]], [ef_rets[bi]], marker='D', color=GREEN,
           s=70, zorder=11, label=f'Max Sharpe ({ef_srs[bi]:.2f})')
plt.colorbar(sc2, ax=ax, label='Sharpe', pad=0.02)
ax.set_xlabel("Volatility (%)", color=MUTED)
ax.set_ylabel("Return (%)", color=MUTED)
stitle(ax, "EFFICIENT FRONTIER CLOUD  (600 portfolios)")
ax.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, alpha=0.3)

# P2D — Beta vs VaR bubble (size = risk contribution)
ax = fig.add_subplot(gs[1, 0])
bcols = [SECTOR_COLORS[s] for s in sectors]
sizes = (rc_pct / rc_pct.max()) * 320 + 25
ax.scatter(stock_betas, stock_var95, s=sizes, c=bcols, alpha=0.82, zorder=5)
ax.axvline(1.0, color=MUTED, lw=0.8, ls='--', alpha=0.6)
ax.axhline(var_95, color=RED, lw=0.8, ls='--', alpha=0.6)
for i, t in enumerate(tickers):
    ax.annotate(t, (stock_betas[i], stock_var95[i]),
                fontsize=5.5, color=MUTED, xytext=(3,2), textcoords='offset points')
ax.set_xlabel("Beta (vs portfolio)", color=MUTED)
ax.set_ylabel("VaR 95% daily (%)", color=MUTED)
stitle(ax, "BETA vs VaR  (bubble = risk contribution)")
note(ax, "Stocks top-right: high systematic risk AND fat tails")
ax.grid(True, alpha=0.3)

# P2E — Sector risk vs weight (diverging bar)
ax = fig.add_subplot(gs[1, 1])
sc_labs  = sector_list
rc_vals  = [sector_rc[s] for s in sc_labs]
wt_vals  = [sector_wt[s] for s in sc_labs]
diff_v   = [r-wv for r, wv in zip(rc_vals, wt_vals)]
bcols2   = [GREEN if d < 0 else ACCENT2 for d in diff_v]
ax.barh(range(len(sc_labs)), diff_v, color=bcols2, height=0.55, alpha=0.88)
ax.axvline(0, color=TEXT, lw=1.1)
ax.set_yticks(range(len(sc_labs)))
ax.set_yticklabels(sc_labs, fontsize=8)
ax.set_xlabel("Risk Contribution minus Capital Weight (pp)", color=MUTED)
stitle(ax, "SECTOR RISK EXCESS / DEFICIT")
for i, d in enumerate(diff_v):
    ax.text(d + (0.12 if d>=0 else -0.12), i,
            f"{d:+.1f}pp", va='center',
            ha='left' if d>=0 else 'right', fontsize=7.5, color=TEXT)
ax.grid(True, axis='x', alpha=0.3)

# P2F — Correlation heatmap (sector-sorted)
ax = fig.add_subplot(gs[1, 2])
sec_ord = []
for s in sector_list:
    sec_ord.extend([i for i, sc in enumerate(sectors) if sc==s])
cs = emp_corr[np.ix_(sec_ord, sec_ord)]
tks = [tickers[i] for i in sec_ord]
cmap_c = LinearSegmentedColormap.from_list('c', [ACCENT3, SURFACE, ACCENT], N=256)
im = ax.imshow(cs, cmap=cmap_c, vmin=-0.2, vmax=1.0, aspect='auto')
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(tks, rotation=90, fontsize=4.8)
ax.set_yticklabels(tks, fontsize=4.8)
plt.colorbar(im, ax=ax, pad=0.02, label='ρ')
cnts  = [sum(1 for s in sectors if s==sec) for sec in sector_list]
cum_c = 0
for c_val in cnts[:-1]:
    cum_c += c_val
    ax.axhline(cum_c-0.5, color=ACCENT, lw=1.2)
    ax.axvline(cum_c-0.5, color=ACCENT, lw=1.2)
stitle(ax, "CORRELATION MATRIX  (sector-sorted)")

fig.suptitle("NOVEL INSIGHTS A  ·  Regime Distributions · Frontier · Beta-VaR · Sector Decomposition",
             fontsize=10, color=ACCENT, fontweight='bold', y=1.01)
save_fig(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Novel Insight Set B: Tail Risk · Clustering · Stress · Scorecard
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.38)

# P3A — CVaR vs VaR per stock
ax = fig.add_subplot(gs[0, 0:2])
sc_sort = np.argsort(stock_cvar95)[::-1]
bw = 0.38
ax.barh(np.arange(N)-bw/2, stock_var95[sc_sort],  height=bw,
        color=ACCENT, alpha=0.85, label='VaR 95%')
ax.barh(np.arange(N)+bw/2, stock_cvar95[sc_sort], height=bw,
        color=ACCENT2, alpha=0.85, label='CVaR 95%')
ax.set_yticks(range(N))
ax.set_yticklabels([tickers[i] for i in sc_sort], fontsize=6.5)
ax.set_xlabel("Daily Risk Measure (%)", color=MUTED)
ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
stitle(ax, "VaR vs CVaR PER STOCK  (95% confidence, daily)")
note(ax, "CVaR = expected loss in the worst 5% of days — captures tail severity beyond VaR", y=0.03)
ax.grid(True, axis='x', alpha=0.4)

# P3B — Hierarchical clustering dendrogram
ax = fig.add_subplot(gs[0, 2])
dist_m   = 1 - np.abs(emp_corr); np.fill_diagonal(dist_m, 0)
Z_link   = linkage(dist_m[np.triu_indices(N, k=1)], method='ward')
dendrogram(Z_link, ax=ax, labels=tickers, orientation='right',
           leaf_font_size=6.5, color_threshold=0.6*max(Z_link[:,2]),
           above_threshold_color=MUTED)
ax.set_xlabel("Ward Distance", color=MUTED)
stitle(ax, "HIERARCHICAL CLUSTERING")
note(ax, "Stocks that cluster together diversify poorly against each other")
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(left=False)

# P3C — Stress scenarios
ax = fig.add_subplot(gs[1, 0])
SCENARIOS = {
    "COVID Crash (Mar-20)":    -0.38,
    "IL&FS Crisis (Sep-18)":   -0.15,
    "Taper Tantrum (2013)":    -0.12,
    "GFC Spill (2008-09)":     -0.52,
    "Demonetisation (Nov-16)": -0.08,
    "+2σ daily shock":         -2*ann_vol_port,
    "+3σ daily shock":         -3*ann_vol_port,
}
snames = list(SCENARIOS.keys())
sloss  = [v*100 for v in SCENARIOS.values()]
bclrs  = [ACCENT2 if v < -20 else ACCENT for v in sloss]
ax.barh(range(len(snames)), sloss, color=bclrs, alpha=0.88, height=0.58)
ax.set_yticks(range(len(snames))); ax.set_yticklabels(snames, fontsize=6.8)
ax.set_xlabel("Portfolio Loss (%)", color=MUTED)
ax.axvline(max_dd, color=RED, lw=1, ls='--', alpha=0.7)
ax.text(max_dd+0.3, len(snames)-0.5, f"Sim MaxDD\n{max_dd:.1f}%",
        color=RED, fontsize=6)
stitle(ax, "STRESS TEST SCENARIOS")
ax.grid(True, axis='x', alpha=0.4)

# P3D — Rolling IT–Financials correlation (novel: cross-sector contagion)
ax = fig.add_subplot(gs[1, 1])
rx2 = np.arange(ROLL2, ROLL2+len(roll_corr_if))
ax.fill_between(rx2, roll_corr_if, color=ACCENT3, alpha=0.35)
ax.plot(rx2, roll_corr_if, color=ACCENT3, lw=1.2)
ax.axhline(np.mean(roll_corr_if), color=ACCENT, lw=1, ls='--', alpha=0.8)
ax.text(rx2[-1]-100, np.mean(roll_corr_if)+0.02,
        f"Avg ρ={np.mean(roll_corr_if):.2f}", color=ACCENT, fontsize=7)
ax.set_xticks([ROLL2,252,504,756,1008,1260])
ax.set_xticklabels(['','Y1','Y2','Y3','Y4','Y5'])
ax.set_ylabel("Rolling 42-day Correlation", color=MUTED)
stitle(ax, "IT ↔ FINANCIALS ROLLING CORRELATION")
note(ax, "Novel: when this rises, cross-sector diversification breaks down (contagion signal)")
ax.grid(True, alpha=0.35)

# P3E — Scorecard
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
metrics = [
    ("Ann. Return",        f"{ann_ret_port:.2f}%",      GREEN),
    ("Ann. Volatility",    f"{ann_vol_port:.2f}%",      ACCENT),
    ("Sharpe Ratio",       f"{sharpe:.3f}",              GREEN if sharpe>1 else ACCENT),
    ("Calmar Ratio",       f"{calmar:.3f}",              GREEN if calmar>0.5 else ACCENT),
    ("Omega Ratio",        f"{omega:.3f}",               GREEN if omega>1 else ACCENT2),
    ("Pain Index",         f"{pain_index:.2f}%",         MUTED),
    ("Ulcer Index",        f"{ulcer_index:.2f}%",        ACCENT2 if ulcer_index>5 else MUTED),
    ("Max Drawdown",       f"{max_dd:.2f}%",             RED),
    ("VaR 95% daily",      f"{var_95:.3f}%",             ACCENT2),
    ("CVaR 95% daily",     f"{cvar_95:.3f}%",            ACCENT2),
    ("VaR 99% daily",      f"{var_99:.3f}%",             RED),
    ("Return Skewness",    f"{port_skew:.3f}",           ACCENT3 if port_skew>0 else ACCENT2),
    ("Excess Kurtosis",    f"{port_kurt:.3f}",           ACCENT2 if port_kurt>1 else MUTED),
    ("Diversif. Ratio",    f"{div_ratio:.3f}",           GREEN if div_ratio>1.2 else MUTED),
    ("Terminal ₹ (per ₹1)",f"Rs{cum_values[-1]:.3f}",   ACCENT),
    ("Stress days",        f"{(regimes==2).sum()}/1260",  ACCENT2),
]
ax.text(0.04, 0.99, "PORTFOLIO SCORECARD", transform=ax.transAxes,
        fontsize=9, color=ACCENT, fontweight='bold', va='top')
for k, (lbl, val, col) in enumerate(metrics):
    y = 0.92 - k * 0.056
    ax.text(0.04, y, lbl, transform=ax.transAxes, fontsize=7, color=MUTED, va='top')
    ax.text(0.70, y, val, transform=ax.transAxes, fontsize=7, color=col,
            va='top', fontweight='bold')
    ax.plot([0.02, 0.98], [y-0.013, y-0.013], color=BORDER, lw=0.5,
            transform=ax.transAxes, clip_on=False)

fig.suptitle("NOVEL INSIGHTS B  ·  Tail Risk · Clustering · Stress Testing · Contagion · Scorecard",
             fontsize=10, color=ACCENT, fontweight='bold', y=1.01)
save_fig(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Even More Novel: Omega Surface · Rolling Beta · Vol-of-Vol · Turnover Cost
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30)

# P4A — Individual stock Omega Ratios (novel)
ax = fig.add_subplot(gs[0, 0])
stock_omegas = []
for i in range(N):
    r = pct_rets[:,i]
    gains  = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    stock_omegas.append(gains/losses if losses>0 else np.nan)
stock_omegas = np.array(stock_omegas)
so_sort = np.argsort(stock_omegas)[::-1]
bar_c   = [SECTOR_COLORS[sectors[i]] for i in so_sort]
ax.barh(range(N), stock_omegas[so_sort], color=bar_c, height=0.65, alpha=0.85)
ax.axvline(1.0, color=TEXT, lw=1, ls='--', alpha=0.6)
ax.set_yticks(range(N))
ax.set_yticklabels([tickers[i] for i in so_sort], fontsize=6.5)
ax.set_xlabel("Omega Ratio  (gains/losses)", color=MUTED)
stitle(ax, "OMEGA RATIO PER STOCK")
note(ax, "Omega > 1 → gains outweigh losses regardless of distribution shape")
ax.grid(True, axis='x', alpha=0.4)

# P4B — Vol-of-Vol (volatility of rolling vol) — regime uncertainty metric
ax = fig.add_subplot(gs[0, 1])
vov_window = 21
vov = np.array([roll_vol_all[max(0,i-vov_window):i].std()
                for i in range(vov_window, len(roll_vol_all)+1)])
vov_x = np.arange(vov_window, vov_window+len(vov))
ax.fill_between(vov_x, vov, color=PURPLE, alpha=0.4)
ax.plot(vov_x, vov, color=PURPLE, lw=1.2)
avg_vov = np.mean(vov)
ax.axhline(avg_vov, color=ACCENT, lw=1, ls='--')
ax.text(vov_x[-1]-100, avg_vov+0.1, f"Avg VoV {avg_vov:.2f}%",
        color=ACCENT, fontsize=7)
ax.set_xticks([0,252,504,756,1008,1260])
ax.set_xticklabels(['Y0','Y1','Y2','Y3','Y4','Y5'])
ax.set_ylabel("Vol-of-Vol (%)", color=MUTED)
stitle(ax, "VOL-OF-VOL  (uncertainty-of-uncertainty)")
note(ax, "Novel: VoV spikes signal regime transitions; useful for options/hedging timing")
ax.grid(True, alpha=0.35)

# P4C — Rolling betas for top 5 risk contributors
ax = fig.add_subplot(gs[1, 0])
top5 = np.argsort(rc_pct)[::-1][:5]
for idx in top5:
    rb = [np.cov(pct_rets[i-ROLL:i,idx], port_pct[i-ROLL:i])[0,1] /
          port_pct[i-ROLL:i].var()
          for i in range(ROLL, DAYS+1)]
    rx3 = np.arange(ROLL, ROLL+len(rb))
    ax.plot(rx3, rb, label=tickers[idx], lw=1.2,
            color=SECTOR_COLORS[sectors[idx]], alpha=0.9)
ax.axhline(1.0, color=MUTED, lw=0.8, ls='--')
ax.set_xticks([ROLL,252,504,756,1008,1260])
ax.set_xticklabels(['','Y1','Y2','Y3','Y4','Y5'])
ax.set_ylabel("Rolling 63-day Beta", color=MUTED)
ax.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, ncol=2)
stitle(ax, "ROLLING BETA — TOP 5 RISK CONTRIBUTORS")
note(ax, "Time-varying beta shows when high-risk stocks decouple from portfolio")
ax.grid(True, alpha=0.35)

# P4D — Skewness & Kurtosis per stock (scatter)
ax = fig.add_subplot(gs[1, 1])
sk = np.array([skew(pct_rets[:,i]) for i in range(N)])
ku = np.array([kurtosis(pct_rets[:,i]) for i in range(N)])
sc3 = ax.scatter(sk, ku, c=[SECTOR_COLORS[s] for s in sectors],
                 s=70, zorder=5, alpha=0.85)
for i, t in enumerate(tickers):
    ax.annotate(t, (sk[i], ku[i]), fontsize=5.5, color=MUTED,
                xytext=(3,3), textcoords='offset points')
ax.axvline(0, color=MUTED, lw=0.8, ls='--', alpha=0.6)
ax.axhline(0, color=MUTED, lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel("Skewness", color=MUTED)
ax.set_ylabel("Excess Kurtosis", color=MUTED)
stitle(ax, "SKEWNESS vs KURTOSIS  (fat-tail map)")
note(ax, "Top-right = fat tails + left skew = most dangerous. Bottom-left = benign.")
# quadrant labels
ylim = ax.get_ylim(); xlim = ax.get_xlim()
ax.text(xlim[0]+0.01, ylim[1]-0.1, "neg skew\nfat tails",
        color=ACCENT2, fontsize=5.5, va='top')
ax.text(xlim[1]-0.01, ylim[1]-0.1, "pos skew\nfat tails",
        color=GREEN, fontsize=5.5, va='top', ha='right')
ax.grid(True, alpha=0.3)

patches2 = [mpatches.Patch(color=SECTOR_COLORS[s], label=s) for s in sector_list]
fig.legend(handles=patches2, loc='lower center', ncol=5, fontsize=7,
           facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("NOVEL INSIGHTS C  ·  Omega · Vol-of-Vol · Rolling Beta · Higher Moments",
             fontsize=10, color=ACCENT, fontweight='bold', y=1.01)
save_fig(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE PDF
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT   = "/mnt/user-data/outputs/india_portfolio_risk_v2.pdf"
PAGE_W, PAGE_H = A4[1], A4[0]   # landscape A4

c = rl_canvas.Canvas(OUTPUT, pagesize=(PAGE_W, PAGE_H))

# ── Cover ─────────────────────────────────────────────────────────────────────
c.setFillColor(BG)
c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
c.setFillColor(ACCENT)
c.rect(0, PAGE_H-7, PAGE_W, 7, fill=1, stroke=0)
c.rect(0, 0, PAGE_W, 5, fill=1, stroke=0)

c.setFont("Helvetica-Bold", 30)
c.setFillColor(ACCENT)
c.drawCentredString(PAGE_W/2, PAGE_H/2+72, "INDIA PORTFOLIO RISK ANALYTICS")

c.setFont("Helvetica", 13)
c.setFillColor(TEXT)
c.drawCentredString(PAGE_W/2, PAGE_H/2+42,
    "20-Stock Equal-Weight  ·  NSE Universe  ·  5-Year Monte Carlo")

c.setFont("Helvetica", 10)
c.setFillColor(MUTED)
c.drawCentredString(PAGE_W/2, PAGE_H/2+18,
    "Sectors: IT  ·  Financials  ·  Energy  ·  FMCG  ·  Healthcare")
c.drawCentredString(PAGE_W/2, PAGE_H/2+4,
    "Standard Metrics  ·  Regime Analysis  ·  Tail Risk  ·  Higher Moments  ·  Contagion  ·  Clustering")

kpis = [
    ("Ann. Return",   f"{ann_ret_port:.1f}%",         GREEN),
    ("Volatility",    f"{ann_vol_port:.1f}%",          ACCENT),
    ("Sharpe",        f"{sharpe:.2f}",                 GREEN),
    ("Max Drawdown",  f"{max_dd:.1f}%",                RED),
    ("Omega Ratio",   f"{omega:.2f}",                  GREEN if omega>1 else ACCENT2),
    ("Div. Ratio",    f"{div_ratio:.2f}",              ACCENT3),
]
bw2, bh2 = 92, 56
tot2 = len(kpis)*bw2 + (len(kpis)-1)*12
x0  = (PAGE_W - tot2) / 2
for i, (lbl, val, col) in enumerate(kpis):
    bx = x0 + i*(bw2+12)
    by = PAGE_H/2 - 100
    c.setFillColor(SURFACE)
    c.setStrokeColor(col); c.setLineWidth(1.2)
    c.roundRect(bx, by, bw2, bh2, 4, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 17)
    c.setFillColor(col)
    c.drawCentredString(bx+bw2/2, by+28, val)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawCentredString(bx+bw2/2, by+12, lbl.upper())

c.setFont("Helvetica", 7.5)
c.setFillColor(MUTED)
c.drawCentredString(PAGE_W/2, 16,
    "Author: Sahej Verma  ·  Madras School of Economics  ·  M.A. Financial Economics  ·  Not financial advice")
c.showPage()

# ── Chart pages ───────────────────────────────────────────────────────────────
page_titles = [
    "PART I — STANDARD OVERVIEW",
    "PART II — REGIME, FRONTIER & SECTOR DECOMPOSITION",
    "PART III — TAIL RISK, CLUSTERING & SCORECARD",
    "PART IV — OMEGA RATIOS, VOL-OF-VOL & HIGHER MOMENTS",
]
total_pages = 1 + len(FIGS)
for i, buf in enumerate(FIGS):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H-4, PAGE_W, 4, fill=1, stroke=0)

    img = ImageReader(buf)
    margin = 10*mm
    c.drawImage(img, margin, margin+6,
                width=PAGE_W-2*margin, height=PAGE_H-2*margin-14,
                preserveAspectRatio=True, anchor='nw')

    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(ACCENT)
    c.drawString(10*mm, 9, page_titles[i])
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawRightString(PAGE_W-10*mm, 9,
        f"India Portfolio Risk Analytics  ·  Page {i+2} of {total_pages}  ·  Sahej Verma")
    c.showPage()

c.save()

print(f"PDF saved → {OUTPUT}")
print(f"Pages: {total_pages}")
print(f"\nKey Metrics:")
print(f"  Ann. Return     : {ann_ret_port:.2f}%")
print(f"  Ann. Volatility : {ann_vol_port:.2f}%")
print(f"  Sharpe Ratio    : {sharpe:.3f}")
print(f"  Calmar Ratio    : {calmar:.3f}")
print(f"  Omega Ratio     : {omega:.3f}")
print(f"  Pain Index      : {pain_index:.2f}%")
print(f"  Ulcer Index     : {ulcer_index:.2f}%")
print(f"  Max Drawdown    : {max_dd:.2f}%")
print(f"  VaR 95% daily   : {var_95:.3f}%")
print(f"  CVaR 95% daily  : {cvar_95:.3f}%")
print(f"  Skewness        : {port_skew:.3f}")
print(f"  Excess Kurtosis : {port_kurt:.3f}")
print(f"  Divers. Ratio   : {div_ratio:.3f}")
print(f"  Vol-of-Vol avg  : {avg_vov:.3f}%")
print(f"  Terminal ₹1     : ₹{cum_values[-1]:.3f}")
print(f"\nTop-5 Risk Contributors:")
for i in np.argsort(rc_pct)[::-1][:5]:
    print(f"  {tickers[i]:12s}  RC={rc_pct[i]:.2f}%  delta={rc_pct[i]-5:+.2f}pp  "
          f"Omega={stock_omegas[i]:.2f}  Beta={stock_betas[i]:.2f}")
