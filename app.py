from __future__ import annotations

import os
import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from datapilot.evaluation import EvaluationBundle, build_decision, evaluate_dataset
from datapilot.profiling import infer_task, profile_dataframe
from datapilot.report import build_markdown_report
from datapilot.sample_data import make_sample_sales

st.set_page_config(
    page_title="DataPilot AI | Prediction readiness",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --dp-ink:#172923; --dp-muted:#65756d; --dp-green:#1f5b49; --dp-mint:#e7f3ec; --dp-bg:#f7f8f4; --dp-border:#e0e7e0; }
.stApp { background: var(--dp-bg); color: var(--dp-ink); font-family:'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: rgba(247,248,244,.88); }
[data-testid="stSidebar"] { background: #eef3ed; border-right:1px solid var(--dp-border); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--dp-ink); }
.block-container { padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1420px; }
.dp-eyebrow { font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--dp-green); }
.dp-title { font-family:'Manrope',sans-serif; font-size:clamp(2rem, 4vw, 3.3rem); line-height:1.07; letter-spacing:-.045em; font-weight:800; color:var(--dp-ink); margin:.55rem 0 .75rem; }
.dp-subtitle { font-size:1.04rem; color:var(--dp-muted); max-width:760px; line-height:1.65; }
.dp-banner { background:linear-gradient(115deg,#edf5ef 0%,#f7f8f4 64%,#eaf1e8 100%); border:1px solid #dce8dc; border-radius:24px; padding:1.55rem 1.7rem; margin:.5rem 0 1.4rem; }
.dp-card { background:#fff; border:1px solid var(--dp-border); border-radius:17px; padding:1rem 1.1rem; }
.dp-kicker { color:var(--dp-muted); font-size:.78rem; font-weight:600; }
.dp-value { color:var(--dp-ink); font-size:1.85rem; line-height:1.3; font-weight:800; letter-spacing:-.04em; }
.dp-small { color:var(--dp-muted); font-size:.87rem; line-height:1.5; }
.dp-pill { display:inline-block; padding:.27rem .58rem; border-radius:99px; font-size:.73rem; font-weight:700; background:#e3eee6; color:#215d47; }
.dp-section { font-family:'Manrope',sans-serif; color:var(--dp-ink); font-size:1.3rem; font-weight:800; letter-spacing:-.025em; margin:.25rem 0 .2rem; }
div.stButton > button[kind="primary"] { background:var(--dp-green); border:1px solid var(--dp-green); border-radius:12px; padding:.68rem 1.05rem; font-weight:700; min-height:3rem; }
div.stButton > button[kind="primary"]:hover { background:#174737; border-color:#174737; }
div.stButton > button { border-radius:11px; min-height:2.65rem; }
[data-testid="stMetric"] { background:#fff; padding:1rem 1.1rem; border:1px solid var(--dp-border); border-radius:16px; }
[data-testid="stMetricLabel"] { color:var(--dp-muted); }
[data-testid="stMetricValue"] { color:var(--dp-ink); font-weight:800; }
[data-testid="stDataFrame"] { border:1px solid var(--dp-border); border-radius:12px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:.5rem; background:transparent; }
.stTabs [data-baseweb="tab"] { background:#edf1eb; padding:.7rem 1.1rem; border-radius:10px; color:#52665b; }
.stTabs [aria-selected="true"] { background:#dfece2 !important; color:#174737 !important; }
hr { border-color:var(--dp-border); }
footer { visibility:hidden; }
@media(max-width:760px){ .dp-banner{padding:1rem;border-radius:16px}.block-container{padding-left:1rem;padding-right:1rem}.dp-title{font-size:2.2rem} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


def metric_pretty(name: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.4f}" if name == "MAE" or name == "RMSE" else f"{value:.3f}"


def render_warning(item: dict[str, str]) -> None:
    icon = {"error": "🛑", "warning": "⚠️", "info": "ℹ️"}.get(item.get("level", "info"), "ℹ️")
    with st.container(border=True):
        st.markdown(f"**{icon} {item['title']}**")
        st.caption(item["detail"])


def make_profile_chart(profile: dict) -> None:
    frame = profile["column_profile"]
    missing = frame.loc[frame["missing"] > 0].sort_values("missing_pct", ascending=True).tail(12)
    if missing.empty:
        st.success("No missing values detected in this dataset.")
    else:
        fig = px.bar(
            missing, x="missing_pct", y="column", orientation="h",
            labels={"missing_pct": "Missing values (%)", "column": "Column"},
            color="missing_pct", color_continuous_scale=["#dceee1", "#1f5b49"],
        )
        fig.update_layout(height=max(250, len(missing) * 34), margin=dict(l=4, r=8, t=8, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False, font=dict(family="DM Sans", color="#34473e"))
        fig.update_yaxes(title=None, gridcolor="#edf0eb")
        fig.update_xaxes(gridcolor="#edf0eb")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def model_metric_chart(bundle: EvaluationBundle) -> None:
    metric_name = "MAE" if bundle.task == "regression" else "Balanced accuracy"
    values = [{"Model": model, metric_name: metrics[metric_name]} for model, metrics in bundle.metrics.items() if metric_name in metrics]
    if not values:
        st.info("No compatible model metrics to chart yet.")
        return
    chart_df = pd.DataFrame(values)
    fig = px.bar(chart_df, x="Model", y=metric_name, color="Model",
                 color_discrete_sequence=["#9bb5a4", "#6b9680", "#1f5b49"],
                 labels={"Model": "Model", metric_name: metric_name})
    fig.update_layout(height=330, margin=dict(l=8, r=8, t=18, b=8), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans", color="#34473e"))
    fig.update_yaxes(gridcolor="#edf0eb")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Lower is better for MAE. Higher is better for balanced accuracy.")


# Sidebar
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin:.25rem 0 1.6rem'><div style='background:#1f5b49;color:white;border-radius:12px;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-size:20px'>↗</div><div><div style='font-size:1.2rem;font-weight:800;letter-spacing:-.04em;color:#172923'>DataPilot <span style='color:#1f5b49'>AI</span></div><div style='font-size:.73rem;color:#65756d'>Prediction readiness</div></div></div>", unsafe_allow_html=True)
    st.markdown("#### Workspace")
    st.caption("Audit the data before you trust the prediction.")
    st.markdown("---")
    st.markdown("**What this prototype checks**")
    st.markdown("- Data quality and missing values\n- Target leakage & proxy detection\n- Held-out model comparison\n- Evidence-based decision report")
    st.markdown("---")
    token_configured = bool(get_tabpfn_token())
    if token_configured:
        st.success("TabPFN API configured")
    else:
        st.warning("TabPFN API not configured")
        st.caption("Local models still work. Add TABPFN_TOKEN in the environment or Streamlit secrets to enable the partner model.")
    st.markdown("---")
    st.caption("Demo datasets are synthetic. Do not upload sensitive or confidential data to hosted inference services.")

# Header
st.markdown("<div class='dp-banner'><div class='dp-eyebrow'>DATA QUALITY · MODEL EVALUATION · DECISION SUPPORT</div><div class='dp-title'>Before you trust a prediction,<br>test the evidence.</div><div class='dp-subtitle'>DataPilot checks your spreadsheet, compares models on held-out data, and turns the results into a decision report you can inspect—not just a prediction you have to believe.</div></div>", unsafe_allow_html=True)

source_col, info_col = st.columns([1.45, 1], gap="large")
with source_col:
    st.markdown("<div class='dp-section'>Start with a dataset</div>", unsafe_allow_html=True)
    source_mode = st.radio("Dataset source", ["Use the sample sales dataset", "Upload my CSV"], horizontal=True, label_visibility="collapsed")
    df = None
    dataset_label = "sample_sales.csv"
    if source_mode == "Use the sample sales dataset":
        df = make_sample_sales()
        st.markdown("<div class='dp-small'>A generated weekly retail dataset with multiple products, promotions, stock, past sales features, and a future sales target. Safe to use in the demo.</div>", unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"], help="Maximum 10 MB. Avoid personal, confidential, or regulated data.")
        if uploaded is not None:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("This prototype accepts CSV files up to 10 MB.")
            else:
                try:
                    df = pd.read_csv(uploaded, low_memory=False)
                    dataset_label = uploaded.name
                    if df.empty:
                        st.error("The CSV has no rows.")
                        df = None
                except Exception as exc:
                    st.error(f"Could not read this CSV: {type(exc).__name__}. Check the file format and try again.")
with info_col:
    st.markdown("<div class='dp-section'>Your workflow</div>", unsafe_allow_html=True)
    for num, title, detail in [
        ("01", "Profile", "Find missing data, duplicates, and target leakage signals."),
        ("02", "Evaluate", "Use held-out data; compare TabPFN with simple baselines."),
        ("03", "Decide", "See what the metrics support—and where they do not."),
    ]:
        st.markdown(f"<div class='dp-card' style='display:flex;gap:14px;margin:.45rem 0'><div style='font-weight:800;color:#1f5b49;font-size:1.03rem'>{num}</div><div><div style='font-weight:700;color:#172923'>{title}</div><div class='dp-small'>{detail}</div></div></div>", unsafe_allow_html=True)

if df is None:
    st.info("Choose the sample dataset or upload a CSV to continue.")
    st.stop()
if df.shape[1] < 2:
    st.error("The dataset needs at least two columns: features and a target.")
    st.stop()
if len(df) > 20_000:
    st.error("This prototype is limited to 20,000 rows for predictable cloud memory use. Sample or aggregate your data before running it.")
    st.stop()
if df.shape[1] > 100:
    st.error("This prototype is limited to 100 columns. Remove irrelevant fields before evaluation.")
    st.stop()

profile = profile_dataframe(df)
metric_cols = st.columns(4)
for col, label, value, delta in [
    (metric_cols[0], "Rows", f"{profile['rows']:,}", None),
    (metric_cols[1], "Columns", f"{profile['columns']:,}", None),
    (metric_cols[2], "Missing cells", f"{profile['missing_cells']:,}", f"{profile['missing_pct']:.1f}% of all cells"),
    (metric_cols[3], "Duplicate rows", f"{profile['duplicate_rows']:,}", None),
]:
    col.metric(label, value, delta=delta, delta_color="off")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

st.markdown("<div class='dp-section'>Configure the audit</div>", unsafe_allow_html=True)
config_left, config_mid, config_right = st.columns([1, 1, 1.3], gap="medium")
with config_left:
    target = st.selectbox(
        "Prediction target",
        options=list(df.columns),
        index=(list(df.columns).index("units_next_week") if "units_next_week" in df.columns else len(df.columns)-1),
    )
    inferred = infer_task(df[target])
    task_options = ["regression", "classification"]
    task_index = task_options.index(inferred)
    task = st.selectbox("Prediction task", task_options, index=task_index, format_func=lambda x: x.title())

with config_mid:
    time_options = ["No time-based split"] + [str(c) for c in profile["date_columns"] if c != target]
    default_time = time_options.index("week_start") if "week_start" in time_options else 0
    time_choice = st.selectbox("Validation strategy", time_options, index=default_time, help="For forecasting or future outcomes, hold out the latest dates rather than randomly mixing past and future.")
    time_col = None if time_choice == "No time-based split" else time_choice

# Target-aware profiling for leakage checks and suggested exclusions
profile_target = profile_dataframe(df, target=target, time_col=time_col)

with config_right:
    feature_options = [str(c) for c in df.columns if c != target]
    suggested_exclude = [c for c in profile_target["constant_columns"] if c != target]
    suggested_exclude += [c for c in profile_target["suspicious_columns"] if c not in suggested_exclude and c != target]
    excluded_columns = st.multiselect(
        "Exclude features (optional)",
        feature_options,
        default=suggested_exclude,
        help="Exclude identifiers, target copies, or fields unavailable at prediction time. Leakage warnings are explainable heuristics; inspect the actual columns.",
    )

run_left, run_note = st.columns([1, 2.5], gap="medium")
with run_left:
    run_clicked = st.button("Run prediction audit", type="primary", use_container_width=True)
with run_note:
    st.caption("The evaluation holds out 20% of eligible rows. Baselines run locally; TabPFN runs only when its API token is configured.")

if run_clicked:
    try:
        with st.spinner("Profiling data, splitting rows, and evaluating models…"):
            evaluation_df = df.copy()
            # Refresh the target-aware profiler for the selected outcome
            eval_profile = profile_dataframe(evaluation_df, target=target, time_col=time_col)
            bundle = evaluate_dataset(
                evaluation_df,
                target=target,
                task=task,
                time_col=time_col,
                excluded_columns=excluded_columns,
                run_tabpfn=True,
            )
            # Carry profiler warnings into evaluation bundle
            bundle.warnings.extend(
                f"{item['title']}: {item['detail']}"
                for item in eval_profile["warnings"]
                if item["level"] in {"warning", "error"}
            )
            report = build_markdown_report(dataset_label, eval_profile, bundle)
            fingerprint = hashlib.sha256(pd.util.hash_pandas_object(evaluation_df, index=True).values.tobytes()).hexdigest()[:16]
            st.session_state["analysis_result"] = {
                "dataset_label": dataset_label,
                "fingerprint": fingerprint,
                "profile": eval_profile,
                "bundle": bundle,
                "report": report,
                "target": target,
                "task": task,
                "time_col": time_col,
                "excluded_columns": tuple(sorted(excluded_columns)),
            }
    except Exception as exc:
        st.error(f"Audit could not be completed: {type(exc).__name__}: {exc}")

result = st.session_state.get("analysis_result")
# Do not display stale results when the data or evaluation settings change.
current_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
if result and (
    result["dataset_label"] != dataset_label
    or result["fingerprint"] != current_fingerprint
    or result["target"] != target
    or result["task"] != task
    or result["time_col"] != time_col
    or result["excluded_columns"] != tuple(sorted(excluded_columns))
):
    result = None

st.markdown("---")
st.markdown("<div class='dp-section'>Dataset health</div>", unsafe_allow_html=True)
health_col, warning_col = st.columns([1, 1.2], gap="large")
with health_col:
    st.markdown("**Missingness by column**")
    make_profile_chart(profile_target)

with warning_col:
    st.markdown("**What the profiler noticed**")
    if profile_target["warnings"]:
        for item in profile_target["warnings"][:5]:
            render_warning(item)
        if len(profile_target["warnings"]) > 5:
            st.caption(f"And {len(profile_target['warnings']) - 5} additional warning(s).")
    else:
        st.success("No issues detected by these basic profiling checks. This is not proof that the dataset is clean.")

    # Dedicated Target Leakage Findings Inspector
    if profile_target.get("leakage_findings"):
        st.markdown("**Target leakage audit findings**")
        for finding in profile_target["leakage_findings"]:
            feat = finding["feature_name"]
            is_excl = feat in excluded_columns
            icon = "🛑" if finding["severity"] == "critical" else ("⚠️" if finding["severity"] == "high" else "ℹ️")
            status_text = "✅ Excluded from models" if is_excl else "🚨 ACTIVE in model inputs"
            with st.container(border=True):
                st.markdown(f"**{icon} `{feat}`: {finding['finding_category'].replace('_', ' ').title()}** ({status_text})")
                st.caption(f"{finding['explanation']} Evidence: {finding['evidence']}")
                st.caption(f"Recommended action: {finding['recommended_action']}")

    st.markdown("**Column profile**")
    st.dataframe(profile_target["column_profile"], use_container_width=True, hide_index=True, height=240)

if result:
    bundle: EvaluationBundle = result["bundle"]
    eval_profile = result["profile"]
    decision = build_decision(bundle)
    st.markdown("---")
    st.markdown("<div class='dp-section'>Evaluation report</div>", unsafe_allow_html=True)
    st.caption(
        f"Target: `{bundle.target}` · {bundle.task.title()} · {bundle.split_strategy} · "
        f"{bundle.train_rows:,} training / {bundle.test_rows:,} held-out rows · "
        f"Excluded features: {', '.join(bundle.excluded_columns) if bundle.excluded_columns else 'None'}"
    )
    status_col, reason_col = st.columns([1, 2.5], gap="medium")
    with status_col:
        label = decision["label"]
        if label == "PROMISING":
            st.success(label)
        elif label == "INCOMPLETE EVALUATION":
            st.warning(label)
        else:
            st.info(label)
    with reason_col:
        st.markdown(decision["reason"])
    met = st.columns(min(3, max(1, len(bundle.metrics))))
    metric_key = "MAE" if bundle.task == "regression" else "Balanced accuracy"
    for c, (model_name, model_metrics) in zip(met, bundle.metrics.items()):
        c.metric(model_name, metric_pretty(metric_key, model_metrics.get(metric_key, float("nan"))))
    result_tab, data_tab, report_tab = st.tabs(["Model comparison", "Held-out predictions", "Decision report"])
    with result_tab:
        chart_col, table_col = st.columns([1, 1.25], gap="large")
        with chart_col:
            st.markdown("**Primary metric comparison**")
            model_metric_chart(bundle)
        with table_col:
            st.markdown("**All evaluation metrics**")
            metrics_frame = pd.DataFrame(bundle.metrics).T.reset_index(names="Model")
            st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
            st.markdown("**Model execution status**")
            for model_name, status in bundle.model_status.items():
                icon = "✅" if status == "completed" else ("⚠️" if status in {"not configured", "failed"} else "ℹ️")
                st.write(f"{icon} **{model_name}:** {status}")
        if bundle.warnings:
            st.markdown("**Evaluation caveats**")
            for warning in bundle.warnings:
                st.warning(warning)
    with data_tab:
        st.markdown("**Predictions on held-out rows**")
        st.caption("The rows shown here were not used to fit the model. Row identifiers refer to the original CSV index and are not business identifiers.")
        st.dataframe(bundle.predictions.head(300), use_container_width=True, hide_index=True)
        st.download_button("Download held-out predictions (CSV)", bundle.predictions.to_csv(index=False).encode("utf-8"), file_name="datapilot-heldout-predictions.csv", mime="text/csv")
    with report_tab:
        st.markdown("**Downloadable prediction-readiness report**")
        st.markdown(result["report"])
        st.download_button("Download report (Markdown)", result["report"].encode("utf-8"), file_name="datapilot-prediction-readiness-report.md", mime="text/markdown")
        st.download_button("Download metrics (CSV)", pd.DataFrame(bundle.metrics).T.to_csv().encode("utf-8"), file_name="datapilot-model-metrics.csv", mime="text/csv")
else:
    st.markdown("---")
    st.markdown("<div class='dp-section'>Your decision report will appear here</div>", unsafe_allow_html=True)
    st.markdown("<div class='dp-card'><div style='font-weight:700;font-size:1.05rem;margin-bottom:.3rem'>No model result has been run yet</div><div class='dp-small'>Choose the target, review suggested exclusions, and run the audit. DataPilot will calculate the metrics from held-out rows; it will not invent a TabPFN result when the API is unavailable.</div></div>", unsafe_allow_html=True)

st.markdown("<div style='padding-top:2rem;color:#738178;font-size:.78rem;text-align:center'>DataPilot AI · Research prototype · Automated profiling and held-out evaluation do not guarantee future model performance.</div>", unsafe_allow_html=True)
