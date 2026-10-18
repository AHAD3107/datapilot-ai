from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st

from datapilot.analyst import get_analyst_client
from datapilot.context import build_audit_context
from datapilot.evaluation import EvaluationBundle, build_decision, evaluate_dataset
from datapilot.policy import (
    get_privacy_status_description,
    is_privacy_mode_enabled,
    set_privacy_mode,
)
from datapilot.profiling import infer_task, profile_dataframe
from datapilot.report import build_markdown_report, generate_executive_summary
from datapilot.sample_data import make_sample_sales
from datapilot.theme import (
    apply_chart_theme,
    get_audit_pipeline_visual_html,
    get_theme_css,
    get_welcome_screen_html,
    render_icon,
)

# Load local environment configuration without overwriting existing system variables
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# Configure Streamlit page layout and appearance (strictly no emojis)
st.set_page_config(
    page_title="DataPilot AI | Prediction Readiness Platform",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize persistent session states
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "welcome_shown" not in st.session_state:
    st.session_state["welcome_shown"] = False
if "nav_section" not in st.session_state:
    st.session_state["nav_section"] = "Overview"
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "audit_result" not in st.session_state:
    st.session_state["audit_result"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None
if "dataset_label" not in st.session_state:
    st.session_state["dataset_label"] = ""

# Inject Theme CSS for the active theme
theme_name = st.session_state["theme"]
st.markdown(get_theme_css(theme_name), unsafe_allow_html=True)

# Render animated welcome screen once per session
if not st.session_state["welcome_shown"]:
    st.markdown(get_welcome_screen_html(theme_name), unsafe_allow_html=True)
    st.session_state["welcome_shown"] = True


# --- Credential & Status Helpers ---

def get_tabpfn_token() -> str:
    token = os.getenv("TABPFN_TOKEN", "").strip()
    if token:
        return token
    try:
        token = str(st.secrets.get("TABPFN_TOKEN", "")).strip()
        if token:
            os.environ["TABPFN_TOKEN"] = token
        return token
    except Exception:
        return ""


def get_ai_api_key() -> str:
    key = os.getenv("AI_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets.get("AI_API_KEY", "")).strip() or str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if key:
            os.environ["AI_API_KEY"] = key
        return key
    except Exception:
        return ""


def metric_pretty(name: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.4f}" if name in {"MAE", "RMSE"} else f"{value:.3f}"


# --- Dataset Loading & State Management ---

if st.session_state["df"] is None:
    st.session_state["df"] = make_sample_sales()
    st.session_state["dataset_label"] = "sample_sales.csv"

df = st.session_state["df"]
dataset_label = st.session_state["dataset_label"]
profile = profile_dataframe(df)
result = st.session_state.get("audit_result")


# --- Sidebar Navigation & Controls ---

with st.sidebar:
    # Wordmark Header
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:0.25rem 0 1.25rem;">
            <div style="background:var(--dp-primary-soft);border:1px solid var(--dp-primary);color:var(--dp-primary);border-radius:10px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;">
                {render_icon("compass", size=22, color="var(--dp-primary)")}
            </div>
            <div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.15rem;font-weight:800;letter-spacing:-0.03em;color:var(--dp-text);">
                    DataPilot <span style="color:var(--dp-primary);">AI</span>
                </div>
                <div style="font-size:0.72rem;color:var(--dp-text-muted);font-weight:500;">
                    Prediction Readiness
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean Navigation Sections (strictly professional icons)
    nav_options = [
        "Overview",
        "Data Quality",
        "Model Evaluation",
        "Leakage Audit",
        "AI Analyst",
        "Reports",
        "Settings",
    ]

    current_index = nav_options.index(st.session_state["nav_section"]) if st.session_state["nav_section"] in nav_options else 0
    selected_nav = st.radio(
        "Navigation",
        nav_options,
        index=current_index,
        label_visibility="collapsed",
    )
    st.session_state["nav_section"] = selected_nav

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # Privacy Mode Section
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;font-weight:700;color:var(--dp-text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">
            {render_icon("shield", size=14, color="var(--dp-text-muted)")} Privacy Control
        </div>
        """,
        unsafe_allow_html=True,
    )
    privacy_toggle = st.toggle(
        "Disable external AI calls (Privacy Mode)",
        value=is_privacy_mode_enabled(),
        help="When enabled, strictly blocks all outbound network calls to hosted TabPFN and external LLM APIs. Local profiling, local baselines, and deterministic explanations remain fully functional.",
    )
    set_privacy_mode(privacy_toggle)
    privacy_info = get_privacy_status_description()

    if privacy_toggle:
        st.markdown(
            f"""
            <div class="dp-badge dp-badge-success" style="width:100%;justify-content:center;padding:0.4rem;margin:0.25rem 0 0.5rem;">
                {render_icon("lock", size=14, color="var(--dp-success)")} Privacy Mode Active
            </div>
            <div style="font-size:0.75rem;color:var(--dp-text-secondary);line-height:1.4;">
                External network inference is blocked. Local models and deterministic summaries are used.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="dp-badge dp-badge-warning" style="width:100%;justify-content:center;padding:0.4rem;margin:0.25rem 0 0.5rem;">
                {render_icon("unlock", size=14, color="var(--dp-warning)")} External Inference Permitted
            </div>
            <div style="font-size:0.75rem;color:var(--dp-text-secondary);line-height:1.4;">
                Configured hosted services (TabPFN, Gemini) may be invoked if required.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # Service Status Block (clean indicators without emojis)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;font-weight:700;color:var(--dp-text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">
            {render_icon("cloud", size=14, color="var(--dp-text-muted)")} Outbound Services
        </div>
        """,
        unsafe_allow_html=True,
    )
    if privacy_toggle:
        st.caption("• TabPFN: Blocked by application policy")
        st.caption("• Google Gemini: Blocked by application policy")
    else:
        tabpfn_ready = bool(get_tabpfn_token())
        gemini_ready = bool(get_ai_api_key())
        tabpfn_icon = render_icon("check_circle", size=13, color="var(--dp-success)") if tabpfn_ready else render_icon("warning", size=13, color="var(--dp-warning)")
        gemini_icon = render_icon("check_circle", size=13, color="var(--dp-success)") if gemini_ready else render_icon("info", size=13, color="var(--dp-text-muted)")

        st.markdown(
            f"""
            <div style="font-size:0.78rem;line-height:1.6;color:var(--dp-text);">
                <div style="display:flex;align-items:center;gap:6px;">{tabpfn_icon} TabPFN: <strong>{'Configured' if tabpfn_ready else 'Not configured'}</strong></div>
                <div style="display:flex;align-items:center;gap:6px;">{gemini_icon} Gemini: <strong>{'Configured' if gemini_ready else 'Local fallback'}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # Active Workspace Context Chip
    ds_rows = len(df)
    ds_cols = df.shape[1]
    st.markdown(
        f"""
        <div style="background:var(--dp-surface-2);border:1px solid var(--dp-border);border-radius:10px;padding:0.6rem 0.8rem;">
            <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--dp-text-muted);font-weight:700;">Active Dataset</div>
            <div style="font-weight:700;font-size:0.85rem;color:var(--dp-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{dataset_label}</div>
            <div style="font-size:0.75rem;color:var(--dp-text-muted);">{ds_rows:,} rows · {ds_cols} columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Main Application Top Bar (Clean Alignment & Breathing Room) ---

header_left, header_right = st.columns([3.2, 1], vertical_alignment="center")
with header_left:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;padding:0.2rem 0;flex-wrap:wrap;">
            <span class="dp-eyebrow">{render_icon('compass', size=15, color='var(--dp-primary)')} DataPilot AI</span>
            <span style="color:var(--dp-border);font-size:0.9rem;">/</span>
            <span style="font-size:0.88rem;color:var(--dp-text);font-weight:700;">{st.session_state['nav_section']}</span>
            <span style="color:var(--dp-border);font-size:0.9rem;">·</span>
            <span style="font-size:0.82rem;color:var(--dp-text-secondary);font-weight:500;">{dataset_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    is_dark = st.session_state["theme"] == "dark"
    theme_btn_label = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
    theme_btn_icon = ":material/light_mode:" if is_dark else ":material/dark_mode:"

    if st.button(theme_btn_label, icon=theme_btn_icon, use_container_width=True):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

st.markdown("<hr style='margin:0.6rem 0 1.25rem 0;'>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: OVERVIEW DASHBOARD
# ==============================================================================
if st.session_state["nav_section"] == "Overview":
    # Hero Section: Two-Column Balanced Composition
    with st.container(border=True):
        st.markdown('<div class="dp-hero-marker"></div>', unsafe_allow_html=True)
        hero_left, hero_right = st.columns([1.22, 1], gap="large")
        with hero_left:
            st.markdown(
                f"""
                <div class="dp-hero-copy">
                    <div class="dp-eyebrow">
                        {render_icon('shield', size=14, color='var(--dp-primary)')} VERIFY BEFORE DEPLOYMENT
                    </div>
                    <div class="dp-brand-title">Predictions deserve evidence.</div>
                    <p class="dp-subtitle">
                        Discover data quality issues, detect leakage, benchmark models on held-out data, and understand the risks before trusting a prediction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cta_col1, cta_col2 = st.columns([1.12, 1], gap="small")
            with cta_col1:
                if st.button("Start a data audit", type="primary", icon=":material/arrow_forward:", use_container_width=True, key="hero_start_audit_btn"):
                    st.session_state["nav_section"] = "Data Quality"
                    st.rerun()
            with cta_col2:
                if st.button("Explore sample data", icon=":material/database:", use_container_width=True, key="hero_explore_sample_btn"):
                    st.session_state["df"] = make_sample_sales()
                    st.session_state["dataset_label"] = "sample_sales.csv"
                    st.session_state["audit_result"] = None
                    st.rerun()
        with hero_right:
            visual_html = get_audit_pipeline_visual_html(theme_name)
            if hasattr(st, "html"):
                st.html(visual_html)
            else:
                st.markdown(visual_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.5rem 0 1.25rem 0;'>", unsafe_allow_html=True)

    # Dataset Source Selector & Workflow Guide
    src_col, guide_col = st.columns([1.4, 1], gap="large")
    with src_col:
        st.markdown(
            f"""
            <div class="dp-card" style="margin-bottom:1rem;">
                <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.3rem;display:flex;align-items:center;gap:8px;">
                    {render_icon('database', size=18, color='var(--dp-primary)')} Select Dataset Source
                </div>
                <div style="font-size:0.82rem;color:var(--dp-text-secondary);margin-bottom:0.75rem;">
                    Use the deterministic synthetic retail benchmark or upload an external CSV (max 10 MB).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        default_source_idx = 0 if st.session_state["dataset_label"] == "sample_sales.csv" or not st.session_state["dataset_label"] else 1
        source_mode = st.radio(
            "Dataset Source",
            ["Synthetic Retail Sales Benchmark", "Upload Custom CSV File"],
            index=default_source_idx,
            label_visibility="collapsed",
            horizontal=True,
        )
        if source_mode == "Synthetic Retail Sales Benchmark":
            if st.session_state["dataset_label"] != "sample_sales.csv":
                st.session_state["df"] = make_sample_sales()
                st.session_state["dataset_label"] = "sample_sales.csv"
                st.session_state["audit_result"] = None
                st.rerun()
            st.caption("Weekly retail sales benchmark with store promotions, stock, lagged metrics, and future units target.")
        else:
            uploaded = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="Maximum 10 MB. Avoid personal, confidential, or regulated data.",
            )
            if uploaded is not None:
                if uploaded.size > 10 * 1024 * 1024:
                    st.error("This prototype accepts CSV files up to 10 MB.")
                else:
                    try:
                        new_df = pd.read_csv(uploaded, low_memory=False)
                        if new_df.empty:
                            st.error("The uploaded CSV has no rows.")
                        elif new_df.shape[1] < 2:
                            st.error("The dataset needs at least two columns: features and a target.")
                        elif len(new_df) > 20_000:
                            st.error("This prototype is limited to 20,000 rows for predictable memory use.")
                        elif new_df.shape[1] > 100:
                            st.error("This prototype is limited to 100 columns.")
                        else:
                            if st.session_state["dataset_label"] != uploaded.name:
                                st.session_state["df"] = new_df
                                st.session_state["dataset_label"] = uploaded.name
                                st.session_state["audit_result"] = None
                                st.rerun()
                    except Exception as exc:
                        st.error(f"Could not parse CSV: {type(exc).__name__}.")

    with guide_col:
        st.markdown(
            f"""
            <div class="dp-card">
                <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.6rem;display:flex;align-items:center;gap:8px;">
                    {render_icon('compass', size=18, color='var(--dp-primary)')} Verification Steps
                </div>
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <span class="dp-badge dp-badge-primary" style="padding:0.2rem 0.5rem;font-weight:800;">1</span>
                        <div>
                            <div style="font-weight:600;font-size:0.85rem;color:var(--dp-text);">Profile & Anomaly Audit</div>
                            <div style="font-size:0.75rem;color:var(--dp-text-secondary);">Inspect missingness, corrupted numerics, and leakage proxies.</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <span class="dp-badge dp-badge-primary" style="padding:0.2rem 0.5rem;font-weight:800;">2</span>
                        <div>
                            <div style="font-weight:600;font-size:0.85rem;color:var(--dp-text);">Evaluate Baselines & TabPFN</div>
                            <div style="font-size:0.75rem;color:var(--dp-text-secondary);">Run models on isolated held-out records with zero data leakage.</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;">
                        <span class="dp-badge dp-badge-primary" style="padding:0.2rem 0.5rem;font-weight:800;">3</span>
                        <div>
                            <div style="font-weight:600;font-size:0.85rem;color:var(--dp-text);">Grounded AI Analyst</div>
                            <div style="font-size:0.75rem;color:var(--dp-text-secondary);">Explain evaluation tradeoffs, data risks, and prioritized fixes.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:1.25rem 0;'>", unsafe_allow_html=True)

    # Normalized Dataset KPI Metric Cards (Uniform height, padding, and deltas)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;font-size:1.05rem;font-weight:700;color:var(--dp-text);margin-bottom:0.75rem;">
            {render_icon('database', size=18, color='var(--dp-primary)')} Dataset Metrics: <code>{dataset_label}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Total Rows",
        f"{profile['rows']:,}",
        delta="Eligible records",
        delta_color="off",
    )
    num_cnt = len(profile.get("numeric_columns", []))
    cat_cnt = len(profile.get("categorical_columns", []))
    dt_cnt = len(profile.get("date_columns", []))
    col_delta = f"{num_cnt} num · {cat_cnt} cat · {dt_cnt} date" if dt_cnt > 0 else f"{num_cnt} num · {cat_cnt} cat"

    metric_cols[1].metric(
        "Total Columns",
        f"{profile['columns']:,}",
        delta=col_delta,
        delta_color="off",
    )
    metric_cols[2].metric(
        "Missing Cells",
        f"{profile['missing_cells']:,}",
        delta=f"{profile['missing_pct']:.1f}% missing",
        delta_color="off",
    )
    metric_cols[3].metric(
        "Duplicate Rows",
        f"{profile['duplicate_rows']:,}",
        delta="Zero duplicates" if profile['duplicate_rows'] == 0 else f"{profile['duplicate_rows']} duplicates",
        delta_color="off",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Health & Evaluation Status Cards (Aligned tops & consistent spacing)
    health_col, eval_summary_col = st.columns([1.2, 1], gap="large")
    with health_col:
        st.markdown(
            f"""
            <div class="dp-card" style="min-height:190px;display:flex;flex-direction:column;justify-content:space-between;">
                <div>
                    <div style="font-weight:700;font-size:1rem;color:var(--dp-text);margin-bottom:0.75rem;display:flex;align-items:center;gap:8px;">
                        {render_icon('shield', size=16, color='var(--dp-primary)')} Data Quality Highlights
                    </div>
                    <div style="font-size:0.84rem;color:var(--dp-text-secondary);line-height:1.7;">
                        • <strong>Constant Columns:</strong> {len(profile.get('constant_columns', []))} detected<br>
                        • <strong>High-Cardinality Fields:</strong> {len(profile.get('high_cardinality_columns', []))} detected<br>
                        • <strong>Corrupted Numeric Columns:</strong> {len(profile.get('corrupted_numeric', []))} detected
                    </div>
                </div>
                <div style="font-size:0.76rem;color:var(--dp-text-muted);border-top:1px solid var(--dp-border);padding-top:0.5rem;margin-top:0.5rem;">
                    Estimated Memory: ~{profile.get('memory_mb', 0.1):.2f} MB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with eval_summary_col:
        if result is not None:
            bundle_obj = result["bundle"]
            dec_obj = build_decision(bundle_obj)
            badge_class = "dp-badge-success" if dec_obj["label"] == "PROMISING" else ("dp-badge-warning" if dec_obj["label"] == "INCOMPLETE EVALUATION" else "dp-badge-secondary")
            st.markdown(
                f"""
                <div class="dp-card" style="min-height:190px;display:flex;flex-direction:column;justify-content:space-between;">
                    <div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                            <span style="font-weight:700;font-size:1rem;color:var(--dp-text);display:flex;align-items:center;gap:8px;">
                                {render_icon('analytics', size=16, color='var(--dp-primary)')} Evaluation Status
                            </span>
                            <span class="dp-badge {badge_class}">{dec_obj['label']}</span>
                        </div>
                        <div style="font-size:0.84rem;color:var(--dp-text-secondary);line-height:1.6;margin-bottom:0.5rem;">
                            Target: <code>{bundle_obj.target}</code> ({bundle_obj.task})<br>
                            Held-out Partition: {bundle_obj.test_rows:,} rows
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Evaluation Results", icon=":material/arrow_forward:", use_container_width=True):
                st.session_state["nav_section"] = "Model Evaluation"
                st.rerun()
        else:
            st.markdown(
                f"""
                <div class="dp-card" style="min-height:190px;display:flex;flex-direction:column;justify-content:space-between;">
                    <div>
                        <div style="font-weight:700;font-size:1rem;color:var(--dp-text);margin-bottom:0.75rem;display:flex;align-items:center;gap:8px;">
                            {render_icon('analytics', size=16, color='var(--dp-primary)')} Evaluation Status
                        </div>
                        <div style="font-size:0.84rem;color:var(--dp-text-secondary);line-height:1.6;">
                            No models evaluated yet for this dataset.<br>
                            Execute an audit to benchmark TabPFN against naive and tree baselines.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Configure & Run Prediction Audit", icon=":material/play_arrow:", type="primary", use_container_width=True):
                st.session_state["nav_section"] = "Model Evaluation"
                st.rerun()


# ==============================================================================
# VIEW 2: DATA QUALITY & ANOMALIES
# ==============================================================================
elif st.session_state["nav_section"] == "Data Quality":
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="dp-eyebrow">{render_icon('database', size=14, color='var(--dp-primary)')} Profile & Diagnostics</div>
            <div class="dp-brand-title">Data Quality & Formatting Anomalies</div>
            <div class="dp-subtitle">
                Systematic inspection of missingness, duplicate records, high-cardinality fields, and corrupted-numeric strings across <code>{dataset_label}</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Missing Values Distribution Chart
    col_chart, col_anomalies = st.columns([1.3, 1], gap="large")
    with col_chart:
        st.markdown(
            f"""
            <div style="font-weight:700;font-size:0.95rem;color:var(--dp-text);margin-bottom:0.4rem;display:flex;align-items:center;gap:6px;">
                {render_icon('analytics', size=16, color='var(--dp-primary)')} Missing Values by Column (%)
            </div>
            """,
            unsafe_allow_html=True,
        )
        frame = profile["column_profile"]
        missing_df = frame.loc[frame["missing"] > 0].sort_values("missing_pct", ascending=True).tail(12)
        if missing_df.empty:
            st.success("No missing values detected in any column of this dataset.")
        else:
            fig = px.bar(
                missing_df,
                x="missing_pct",
                y="column",
                orientation="h",
                labels={"missing_pct": "Missing (%)", "column": "Column"},
                color="missing_pct",
                color_continuous_scale=["#38BDF8", "#F87171"],
            )
            fig = apply_chart_theme(fig, theme_name)
            fig.update_layout(coloraxis_showscale=False, height=max(260, len(missing_df) * 32))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_anomalies:
        st.markdown(
            f"""
            <div style="font-weight:700;font-size:0.95rem;color:var(--dp-text);margin-bottom:0.4rem;display:flex;align-items:center;gap:6px;">
                {render_icon('warning', size=16, color='var(--dp-warning)')} Structural Warnings
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not profile["warnings"]:
            st.info("No structural warnings detected.")
        else:
            for w in profile["warnings"]:
                badge_type = "dp-badge-danger" if w.get("level") == "error" else "dp-badge-warning"
                st.markdown(
                    f"""
                    <div class="dp-card" style="margin-bottom:0.6rem;padding:0.75rem 1rem;">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.25rem;">
                            <strong style="color:var(--dp-text);font-size:0.85rem;">{w['title']}</strong>
                            <span class="dp-badge {badge_type}">{w.get('level', 'warning').upper()}</span>
                        </div>
                        <div style="font-size:0.78rem;color:var(--dp-text-secondary);line-height:1.4;">{w['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<hr style='margin:1.25rem 0;'>", unsafe_allow_html=True)

    # Corrupted Numeric Section
    st.markdown(
        f"""
        <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.5rem;display:flex;align-items:center;gap:8px;">
            {render_icon('warning', size=18, color='var(--dp-warning)')} Corrupted-Numeric & Format Inconsistencies
        </div>
        """,
        unsafe_allow_html=True,
    )
    corrupted = profile.get("corrupted_numeric", [])
    if not corrupted:
        st.success("No corrupted-numeric columns identified. All string columns represent genuine categoricals or identifiers.")
    else:
        for cn in corrupted:
            samples_str = ", ".join(f"<code>{s}</code>" for s in cn.get("sample_values", []))
            st.markdown(
                f"""
                <div class="dp-card" style="margin-bottom:0.65rem;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.35rem;flex-wrap:wrap;gap:8px;">
                        <span style="font-weight:700;font-size:0.92rem;color:var(--dp-text);">Column: <code>{cn['column']}</code></span>
                        <span class="dp-badge dp-badge-warning">{cn['clean_rate']:.1f}% parseable as numeric</span>
                    </div>
                    <div style="font-size:0.82rem;color:var(--dp-text-secondary);margin-bottom:0.3rem;">{cn['explanation']}</div>
                    <div style="font-size:0.78rem;color:var(--dp-text-secondary);">Representative formatted examples: {samples_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='margin:1.25rem 0;'>", unsafe_allow_html=True)

    # Full Column Profile Table
    st.markdown(
        f"""
        <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.5rem;display:flex;align-items:center;gap:8px;">
            {render_icon('table', size=18, color='var(--dp-primary)')} Full Column Profile
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(profile["column_profile"], use_container_width=True, hide_index=True, height=280)


# ==============================================================================
# VIEW 3: MODEL EVALUATION WORKSPACE
# ==============================================================================
elif st.session_state["nav_section"] == "Model Evaluation":
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="dp-eyebrow">{render_icon('analytics', size=14, color='var(--dp-primary)')} Benchmarking Engine</div>
            <div class="dp-brand-title">Model Evaluation & Held-Out Comparison</div>
            <div class="dp-subtitle">
                Configure prediction target, partition methodology, and feature exclusions. Evaluate naive baselines, Random Forest, and TabPFN on identical held-out rows.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Configuration Parameters Card
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="font-weight:700;font-size:0.95rem;color:var(--dp-text);margin-bottom:0.6rem;display:flex;align-items:center;gap:8px;">
                {render_icon('settings', size=16, color='var(--dp-primary)')} Audit Parameters
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Prediction Target", df.columns, index=len(df.columns) - 1)
        inferred = infer_task(df[target])
        task = c2.selectbox("Task Type", ["regression", "classification"], index=0 if inferred == "regression" else 1)
        split = c3.selectbox("Split Strategy", ["Chronological split", "Random holdout"])

        time_col = None
        if split == "Chronological split":
            date_candidates = [c for c in df.columns if c != target and ("date" in c.lower() or "week" in c.lower() or "time" in c.lower())]
            time_col = st.selectbox(
                "Chronological Date Column",
                date_candidates or [c for c in df.columns if c != target],
                help="Rows sharing the same date are kept in the same chronological partition.",
            )

        profile_target = profile_dataframe(df, target=target)
        suggested_exclusions = profile_target.get("suggested_exclusions", [])

        excluded_cols = st.multiselect(
            "Feature Exclusions (Recommended for Target Leakage)",
            options=[c for c in df.columns if c != target],
            default=suggested_exclusions,
            help="Features listed here are strictly excluded from all model training and test inputs.",
        )

        run_btn = st.button("Run Prediction Audit", icon=":material/play_arrow:", type="primary", use_container_width=True)

    if run_btn:
        with st.status("Executing prediction audit pipeline...", expanded=True) as status_box:
            st.write("Fitting local baselines and evaluating held-out rows...")
            bundle = evaluate_dataset(
                df,
                target=target,
                task=task,
                split_strategy="chronological" if split == "Chronological split" else "random",
                time_col=time_col,
                excluded_columns=excluded_cols,
                run_tabpfn=True,
            )
            st.write("Synthesizing executive prediction-readiness summary...")
            summary_text = generate_executive_summary(
                profile_target,
                bundle,
                dataset_label=dataset_label,
                use_llm=True,
            )
            report_md = build_markdown_report(
                profile_target,
                bundle,
                dataset_label=dataset_label,
                executive_summary=summary_text,
            )
            st.session_state["audit_result"] = {
                "bundle": bundle,
                "profile": profile_target,
                "executive_summary": summary_text,
                "report": report_md,
            }
            status_box.update(label="Prediction audit complete!", state="complete", expanded=False)
            st.rerun()

    # Display Current Audit Results
    current_audit = st.session_state.get("audit_result")
    if current_audit is not None:
        bundle_obj: EvaluationBundle = current_audit["bundle"]
        dec_obj = build_decision(bundle_obj)

        st.markdown("<hr style='margin:1.25rem 0;'>", unsafe_allow_html=True)

        # Decision Summary Banner
        dec_badge = "dp-badge-success" if dec_obj["label"] == "PROMISING" else ("dp-badge-warning" if dec_obj["label"] == "INCOMPLETE EVALUATION" else "dp-badge-secondary")
        st.markdown(
            f"""
            <div class="dp-card" style="margin-bottom:1.25rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;flex-wrap:wrap;gap:8px;">
                    <div style="font-size:0.8rem;text-transform:uppercase;font-weight:700;color:var(--dp-text-muted);letter-spacing:0.05em;">
                        Validation Decision Status
                    </div>
                    <span class="dp-badge {dec_badge}" style="font-size:0.85rem;padding:0.35rem 0.8rem;">
                        {dec_obj['label']}
                    </span>
                </div>
                <div style="font-size:0.92rem;color:var(--dp-text);line-height:1.5;margin-bottom:0.5rem;">
                    {dec_obj['reason']}
                </div>
                <div style="font-size:0.78rem;color:var(--dp-text-secondary);">
                    Target: <code>{bundle_obj.target}</code> · {bundle_obj.split_strategy} · {bundle_obj.train_rows:,} train / {bundle_obj.test_rows:,} test rows
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics Breakdown Grid
        chart_col, table_col = st.columns([1.2, 1], gap="large")
        with chart_col:
            st.markdown(
                f"""
                <div style="font-weight:700;font-size:0.95rem;color:var(--dp-text);margin-bottom:0.4rem;display:flex;align-items:center;gap:6px;">
                    {render_icon('analytics', size=16, color='var(--dp-primary)')} Held-Out Model Comparison
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_key = "MAE" if bundle_obj.task == "regression" else "Balanced accuracy"
            values = [{"Model": model, metric_key: metrics[metric_key]} for model, metrics in bundle_obj.metrics.items() if metric_key in metrics]
            if values:
                chart_df = pd.DataFrame(values)
                fig_model = px.bar(
                    chart_df,
                    x="Model",
                    y=metric_key,
                    color="Model",
                    color_discrete_sequence=["#38BDF8", "#34D399", "#818CF8"],
                    labels={"Model": "Model", metric_key: metric_key},
                )
                fig_model = apply_chart_theme(fig_model, theme_name)
                fig_model.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_model, use_container_width=True, config={"displayModeBar": False})
                st.caption("Lower is better for MAE/RMSE. Higher is better for accuracy and balanced accuracy.")

        with table_col:
            st.markdown(
                f"""
                <div style="font-weight:700;font-size:0.95rem;color:var(--dp-text);margin-bottom:0.4rem;display:flex;align-items:center;gap:6px;">
                    {render_icon('table', size=16, color='var(--dp-primary)')} Measured Held-Out Metrics
                </div>
                """,
                unsafe_allow_html=True,
            )
            metrics_df = pd.DataFrame(bundle_obj.metrics).T.reset_index(names="Model")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            st.markdown(
                f"""
                <div style="font-weight:700;font-size:0.9rem;color:var(--dp-text);margin:0.8rem 0 0.3rem;">
                    Model Execution Statuses
                </div>
                """,
                unsafe_allow_html=True,
            )
            for m_name, m_status in bundle_obj.model_status.items():
                status_icon = render_icon("check_circle", size=13, color="var(--dp-success)") if m_status == "completed" else render_icon("warning", size=13, color="var(--dp-warning)")
                st.markdown(
                    f"""
                    <div style="font-size:0.8rem;color:var(--dp-text);display:flex;align-items:center;gap:6px;margin-bottom:0.2rem;">
                        {status_icon} <strong>{m_name}:</strong> <code>{m_status}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if bundle_obj.warnings:
            st.markdown("<br>", unsafe_allow_html=True)
            for w in bundle_obj.warnings:
                st.warning(w)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Consult AI Analyst for Deep Explanation", icon=":material/psychology:", use_container_width=True):
            st.session_state["nav_section"] = "AI Analyst"
            st.rerun()
    else:
        st.info("Configure your target and click 'Run Prediction Audit' to generate held-out model comparisons.")


# ==============================================================================
# VIEW 4: TARGET LEAKAGE AUDIT
# ==============================================================================
elif st.session_state["nav_section"] == "Leakage Audit":
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="dp-eyebrow">{render_icon('shield', size=14, color='var(--dp-primary)')} Integrity Verification</div>
            <div class="dp-brand-title">Target Leakage & Proxy Audit</div>
            <div class="dp-subtitle">
                Detection of features that directly duplicate the target or introduce post-decision proxies that inflate validation scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _audit_res = st.session_state.get("audit_result")
    if _audit_res is not None:
        _bundle = _audit_res["bundle"]
        _leakage_target = _bundle.target
        _excluded_set: set[str] = set(_bundle.excluded_columns)
    else:
        _leakage_target = str(df.columns[-1])
        _excluded_set = set()
    profile_target = profile_dataframe(df, target=_leakage_target)
    leakage_findings = profile_target.get("leakage_findings", [])
    excluded_set = _excluded_set

    if not leakage_findings:
        st.success("No target leakage heuristics or direct target copies detected for the current target.")
    else:
        for finding in leakage_findings:
            is_excluded = finding["column"] in excluded_set
            sev = finding.get("severity", "warning").lower()
            badge_class = "dp-badge-danger" if sev in {"critical", "high"} else "dp-badge-warning"
            status_chip = f"<span class='dp-badge dp-badge-success'>{render_icon('check_circle', size=12, color='var(--dp-success)')} Excluded From Model Inputs</span>" if is_excluded else f"<span class='dp-badge dp-badge-danger'>{render_icon('warning', size=12, color='var(--dp-danger)')} Active in Model Inputs</span>"

            st.markdown(
                f"""
                <div class="dp-card" style="margin-bottom:0.85rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;flex-wrap:wrap;gap:8px;">
                        <span style="font-weight:700;font-size:0.95rem;color:var(--dp-text);">Feature: <code>{finding['column']}</code></span>
                        <div style="display:flex;gap:8px;flex-wrap:wrap;">
                            <span class="dp-badge {badge_class}">{finding.get('category', 'LEAKAGE').upper()}</span>
                            {status_chip}
                        </div>
                    </div>
                    <div style="font-size:0.84rem;color:var(--dp-text);margin-bottom:0.35rem;">
                        {finding['explanation']}
                    </div>
                    <div style="font-size:0.78rem;color:var(--dp-text-secondary);margin-bottom:0.25rem;">
                        <strong>Statistical Evidence:</strong> <code>{finding['evidence']}</code>
                    </div>
                    <div style="font-size:0.78rem;color:var(--dp-primary);">
                        <strong>Recommended Action:</strong> {finding['recommended_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# VIEW 5: GROUNDED AI ANALYST
# ==============================================================================
elif st.session_state["nav_section"] == "AI Analyst":
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="dp-eyebrow">{render_icon('brain', size=14, color='var(--dp-primary)')} Conversational Intelligence</div>
            <div class="dp-brand-title">Grounded AI Analyst</div>
            <div class="dp-subtitle">
                Ask targeted questions about your data quality, leakage findings, model evaluation tradeoffs, and recommended next steps.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_audit = st.session_state.get("audit_result")
    if current_audit is None:
        st.info("Run a prediction audit first in 'Model Evaluation' before chatting with the AI Analyst.")
        if st.button("Go to Model Evaluation", icon=":material/arrow_forward:", type="primary"):
            st.session_state["nav_section"] = "Model Evaluation"
            st.rerun()
    else:
        bundle_obj: EvaluationBundle = current_audit["bundle"]
        eval_profile = current_audit["profile"]

        # Context Indicator Chip
        privacy_status_text = "Privacy Mode Active (Deterministic local explanations)" if is_privacy_mode_enabled() else "External AI Enabled (Google Gemini)"
        st.markdown(
            f"""
            <div style="background:var(--dp-surface-2);border:1px solid var(--dp-border);border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:1.25rem;display:flex;justify-content:space-between;align-items:center;font-size:0.8rem;flex-wrap:wrap;gap:8px;">
                <span style="color:var(--dp-text);">Context: <strong>{dataset_label}</strong> · Target: <code>{bundle_obj.target}</code> · Held-Out: {bundle_obj.test_rows:,} rows</span>
                <span style="color:var(--dp-text-secondary);">{privacy_status_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested Questions (Strictly professional icons, no emojis)
        st.markdown("<div style='font-weight:700;font-size:0.85rem;color:var(--dp-text-muted);margin-bottom:0.4rem;'>Suggested Inquiries:</div>", unsafe_allow_html=True)
        q_cols = st.columns(3)
        suggested_q = None
        if q_cols[0].button("Explain Model Comparison", icon=":material/analytics:", use_container_width=True):
            suggested_q = "Explain the model comparison results and why one model performed better than the other."
        if q_cols[1].button("Investigate Target Leakage", icon=":material/shield:", use_container_width=True):
            suggested_q = "Explain any target leakage or proxy findings detected in this dataset."
        if q_cols[2].button("Prioritized Fixes", icon=":material/lightbulb:", use_container_width=True):
            suggested_q = "What are the most critical data quality or leakage issues I should fix first?"

        st.markdown("<br>", unsafe_allow_html=True)

        # Render Conversation Stream
        for msg in st.session_state["chat_messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="dp-chat-bubble-user">
                        <div style="font-size:0.75rem;font-weight:700;color:var(--dp-text-muted);margin-bottom:0.25rem;text-transform:uppercase;">User</div>
                        <div>{html.escape(msg['content'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="dp-chat-bubble-ai">
                        <div style="font-size:0.75rem;font-weight:700;color:var(--dp-primary);margin-bottom:0.35rem;text-transform:uppercase;display:flex;align-items:center;gap:6px;">
                            {render_icon('brain', size=14, color='var(--dp-primary)')} DataPilot AI Analyst
                        </div>
                        <div style="line-height:1.6;">{msg['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        user_input = st.chat_input("Ask a question about this audit result...")
        question_to_ask = suggested_q or user_input

        if question_to_ask:
            st.session_state["chat_messages"].append({"role": "user", "content": question_to_ask})
            with st.spinner("Analyzing verified audit context..."):
                ctx = build_audit_context(eval_profile, bundle_obj, dataset_label)
                client = get_analyst_client()
                response_text = client.explain_audit(ctx, question_to_ask)
                st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})
            st.rerun()

        if st.session_state["chat_messages"]:
            if st.button("Clear Conversation", icon=":material/delete_outline:"):
                st.session_state["chat_messages"] = []
                st.rerun()


# ==============================================================================
# VIEW 6: REPORTS & EXPORTS
# ==============================================================================
elif st.session_state["nav_section"] == "Reports":
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="dp-eyebrow">{render_icon('download', size=14, color='var(--dp-primary)')} Export Center</div>
            <div class="dp-brand-title">Audit Reports & Downloads</div>
            <div class="dp-subtitle">
                Export comprehensive prediction-readiness reports and held-out evaluation datasets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_audit = st.session_state.get("audit_result")
    if current_audit is None:
        st.info("Run an audit first in 'Model Evaluation' to generate downloadable reports.")
    else:
        bundle_obj: EvaluationBundle = current_audit["bundle"]
        exec_sum = current_audit.get("executive_summary", "")

        # Executive Summary Box
        if exec_sum:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.5rem;display:flex;align-items:center;gap:8px;">
                        {render_icon('file_text', size=18, color='var(--dp-primary)')} Executive Summary
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(exec_sum)

        st.markdown("<hr style='margin:1.25rem 0;'>", unsafe_allow_html=True)

        # Download Buttons Grid
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.download_button(
                "Download Report (Markdown)",
                data=current_audit["report"].encode("utf-8"),
                file_name="datapilot-prediction-readiness-report.md",
                mime="text/markdown",
                icon=":material/download:",
                use_container_width=True,
            )
        with d_col2:
            metrics_csv = pd.DataFrame(bundle_obj.metrics).T.to_csv().encode("utf-8")
            st.download_button(
                "Download Metrics (CSV)",
                data=metrics_csv,
                file_name="datapilot-model-metrics.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
            )
        with d_col3:
            preds_csv = bundle_obj.predictions.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Held-out Predictions (CSV)",
                data=preds_csv,
                file_name="datapilot-heldout-predictions.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
            )


# ==============================================================================
# VIEW 7: SETTINGS & CONFIGURATION
# ==============================================================================
elif st.session_state["nav_section"] == "Settings":
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="dp-eyebrow">{render_icon('settings', size=14, color='var(--dp-primary)')} Configuration</div>
            <div class="dp-brand-title">System Settings & Privacy Control</div>
            <div class="dp-subtitle">
                Manage external API integrations, privacy policies, and appearance preferences.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Privacy Mode Card
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.4rem;display:flex;align-items:center;gap:8px;">
                {render_icon('shield', size=18, color='var(--dp-primary)')} Application Air-Gap & Privacy Policy
            </div>
            """,
            unsafe_allow_html=True,
        )
        settings_priv_toggle = st.toggle(
            "Enable Privacy Mode (Disable External AI Calls)",
            value=is_privacy_mode_enabled(),
            key="settings_privacy_toggle",
            help="Strictly blocks external inference calls to hosted TabPFN and Google Gemini APIs.",
        )
        set_privacy_mode(settings_priv_toggle)
        st.caption(
            "When Privacy Mode is enabled, no dataset metadata, column names, or feature arrays leave this application. "
            "All profiling, train/test splitting, baselines, and deterministic explanations operate 100% locally."
        )

    # API Configuration Status
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="font-weight:700;font-size:1.05rem;color:var(--dp-text);margin-bottom:0.4rem;display:flex;align-items:center;gap:8px;">
                {render_icon('cloud', size=18, color='var(--dp-primary)')} API Integrations
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:0.85rem;line-height:1.6;color:var(--dp-text);">
                • <strong>Google Gemini API:</strong> {'Configured via AI_API_KEY' if get_ai_api_key() else 'Not configured (using deterministic local fallback)'}<br>
                • <strong>Model Identifier:</strong> <code>{os.getenv('AI_MODEL', 'gemini-2.5-flash')}</code><br>
                • <strong>Prior Labs TabPFN:</strong> {'Configured via TABPFN_TOKEN' if get_tabpfn_token() else 'Not configured (baseline models run locally)'}
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Footer ---
st.markdown(
    f"""
    <div style="padding-top:2.5rem;color:var(--dp-text-muted);font-size:0.78rem;text-align:center;">
        DataPilot AI · Prediction Readiness Platform · Research Prototype · Automated verification does not guarantee future operational performance.
    </div>
    """,
    unsafe_allow_html=True,
)
