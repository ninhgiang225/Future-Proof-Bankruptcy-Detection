"""UI helper utilities for the Bankruptcy Risk Prediction app."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Color palette — modern fintech theme
# ---------------------------------------------------------------------------
COLORS = {
    "safe":         "#10b981",  # emerald-500 — healthy / away from bankruptcy
    "safe_bg":      "#ecfdf5",  # emerald-50  — banner background
    "safe_border":  "#059669",  # emerald-600 — banner accent
    "warn":         "#f59e0b",  # amber-500   — medium risk
    "warn_bg":      "#fffbeb",  # amber-50
    "warn_border":  "#d97706",  # amber-600
    "danger":       "#ef4444",  # rose-500    — high risk / toward bankruptcy
    "danger_bg":    "#fef2f2",  # rose-50
    "danger_border":"#dc2626",  # rose-600
    "accent":       "#6366f1",  # indigo-500  — global importance bars
    "gauge_bar":    "#1e293b",  # slate-800   — gauge needle
    "text_dark":    "#0f172a",  # slate-900   — banner text
}

# ---------------------------------------------------------------------------
# Feature categories — maps each of the 95 features into semantic groups
# Column names have leading spaces stripped on load, so use stripped names here.
# ---------------------------------------------------------------------------

FEATURE_CATEGORIES = {
    "Profitability": [
        "ROA(C) before interest and depreciation before interest",
        "ROA(A) before interest and % after tax",
        "ROA(B) before interest and depreciation after tax",
        "Operating Gross Margin",
        "Realized Sales Gross Margin",
        "Operating Profit Rate",
        "Pre-tax net Interest Rate",
        "After-tax net Interest Rate",
        "Non-industry income and expenditure/revenue",
        "Continuous interest rate (after tax)",
        "Operating Expense Rate",
        "Research and development expense rate",
        "Cash flow rate",
        "Interest-bearing debt interest rate",
        "Tax rate (A)",
        "Total income/Total expense",
        "Total expense/Assets",
        "Gross Profit to Sales",
        "Net Income to Total Assets",
        "Net Income to Stockholder's Equity",
        "Net Income Flag",
    ],
    "Per-Share Metrics": [
        "Net Value Per Share (B)",
        "Net Value Per Share (A)",
        "Net Value Per Share (C)",
        "Persistent EPS in the Last Four Seasons",
        "Cash Flow Per Share",
        "Revenue Per Share (Yuan ¥)",
        "Operating Profit Per Share (Yuan ¥)",
        "Per Share Net profit before tax (Yuan ¥)",
        "Revenue per person",
        "Operating profit per person",
        "Allocation rate per person",
    ],
    "Growth": [
        "Realized Sales Gross Profit Growth Rate",
        "Operating Profit Growth Rate",
        "After-tax Net Profit Growth Rate",
        "Regular Net Profit Growth Rate",
        "Continuous Net Profit Growth Rate",
        "Total Asset Growth Rate",
        "Net Value Growth Rate",
        "Total Asset Return Growth Rate Ratio",
        "Cash Reinvestment %",
    ],
    "Liquidity": [
        "Current Ratio",
        "Quick Ratio",
        "Working Capital to Total Assets",
        "Quick Assets/Total Assets",
        "Current Assets/Total Assets",
        "Cash/Total Assets",
        "Quick Assets/Current Liability",
        "Cash/Current Liability",
        "Current Liability to Assets",
        "Operating Funds to Liability",
        "Inventory/Working Capital",
        "Inventory/Current Liability",
        "No-credit Interval",
    ],
    "Turnover": [
        "Total Asset Turnover",
        "Accounts Receivable Turnover",
        "Average Collection Days",
        "Inventory Turnover Rate (times)",
        "Fixed Assets Turnover Frequency",
        "Net Worth Turnover Rate (times)",
        "Current Asset Turnover Rate",
        "Quick Asset Turnover Rate",
        "Working capitcal Turnover Rate",
        "Cash Turnover Rate",
        "Cash Flow to Sales",
    ],
    "Asset Composition": [
        "Fixed Assets to Assets",
        "Current Liability to Liability",
        "Current Liability to Equity",
        "Current Liabilities/Liability",
        "Working Capital/Equity",
        "Current Liabilities/Equity",
        "Long-term Liability to Current Assets",
        "Retained Earnings to Total Assets",
        "Current Liability to Current Assets",
        "Liability-Assets Flag",
        "Total assets to GNP price",
        "Cash Flow to Total Assets",
        "Cash Flow to Liability",
        "CFO to Assets",
        "Cash Flow to Equity",
        "Equity to Long-term Liability",
    ],
    "Leverage": [
        "Interest Expense Ratio",
        "Total debt/Total net worth",
        "Debt ratio %",
        "Net worth/Assets",
        "Long-term fund suitability ratio (A)",
        "Borrowing dependency",
        "Contingent liabilities/Net worth",
        "Operating profit/Paid-in capital",
        "Net profit before tax/Paid-in capital",
        "Inventory and accounts receivable/Net value",
        "Liability to Equity",
        "Degree of Financial Leverage (DFL)",
        "Interest Coverage Ratio (Interest expense to EBIT)",
        "Equity to Liability",
    ],
}


def get_all_categorized_features() -> list[str]:
    """Return a flat list of all features across every category."""
    feats: list[str] = []
    for group in FEATURE_CATEGORIES.values():
        feats.extend(group)
    return feats


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_shap_explainer(_model):
    """Create and cache a SHAP TreeExplainer for the model."""
    import shap
    return shap.TreeExplainer(_model)


def compute_shap_values(_explainer, input_df: pd.DataFrame):
    """Compute SHAP values for a single-row input DataFrame."""
    return _explainer(input_df)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def risk_gauge(probability: float) -> go.Figure:
    """Return a Plotly gauge chart for the bankruptcy risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "Bankruptcy Risk Score"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": COLORS["gauge_bar"]},
            "steps": [
                {"range": [0, 30], "color": COLORS["safe_bg"]},
                {"range": [30, 70], "color": COLORS["warn_bg"]},
                {"range": [70, 100], "color": COLORS["danger_bg"]},
            ],
            "threshold": {
                "line": {"color": COLORS["danger"], "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_waterfall_plot(shap_values, feature_names: list[str], top_n: int = 15) -> go.Figure:
    """Horizontal bar chart mimicking a SHAP waterfall for the top N features."""
    vals = shap_values.values[0]
    base_value = shap_values.base_values[0]

    indices = np.argsort(np.abs(vals))[-top_n:]
    indices = indices[np.argsort(np.abs(vals[indices]))]  # ascending for horizontal bar

    top_vals = vals[indices]
    top_names = [feature_names[i] for i in indices]
    colors = [COLORS["danger"] if v > 0 else COLORS["safe"] for v in top_vals]

    fig = go.Figure(go.Bar(
        x=top_vals,
        y=top_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Contributions (base value = {base_value:.4f})",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="",
        height=max(400, top_n * 32),
        margin=dict(l=250, r=60, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_top_contributors(shap_values, feature_names: list[str], top_n: int = 10) -> tuple[go.Figure, pd.DataFrame]:
    """Bar chart + table of the top N SHAP contributors."""
    vals = shap_values.values[0]
    indices = np.argsort(np.abs(vals))[-top_n:][::-1]  # descending

    top_vals = vals[indices]
    top_names = [feature_names[i] for i in indices]
    colors = [COLORS["danger"] if v > 0 else COLORS["safe"] for v in top_vals]

    fig = go.Figure(go.Bar(
        x=top_vals,
        y=top_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Contributors",
        xaxis_title="SHAP Value",
        height=max(350, top_n * 35),
        margin=dict(l=250, r=60, t=60, b=40),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    df = pd.DataFrame({
        "Feature": top_names,
        "SHAP Value": top_vals,
        "Direction": ["Toward Bankruptcy" if v > 0 else "Away from Bankruptcy" for v in top_vals],
    })
    return fig, df


def global_feature_importance(model, feature_names: list[str], top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of model-level feature importance (top N)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation="h",
        marker_color=COLORS["accent"],
    ))
    fig.update_layout(
        title=f"Top {top_n} Global Feature Importances",
        xaxis_title="Importance",
        height=max(400, top_n * 30),
        margin=dict(l=250, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
