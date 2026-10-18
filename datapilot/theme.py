from __future__ import annotations

import html

# --- Theme Token Palettes ---

THEMES = {
    "light": {
        "name": "light",
        "bg": "#F5F7FB",
        "surface": "#FFFFFF",
        "surface_2": "#EEF2F7",
        "surface_3": "#E2E8F0",
        "text": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#64748B",
        "primary": "#047857",
        "primary_hover": "#065F46",
        "primary_soft": "#ECFDF5",
        "primary_text": "#065F46",
        "primary_btn_text": "#FFFFFF",
        "secondary": "#0369A1",
        "secondary_soft": "#F0F9FF",
        "border": "#D1D5DB",
        "border_focus": "#0369A1",
        "danger": "#DC2626",
        "danger_soft": "#FEF2F2",
        "warning": "#D97706",
        "warning_soft": "#FFFBEB",
        "success": "#16A34A",
        "success_soft": "#F0FDF4",
        "input_text": "#0F172A",
        "card_shadow": "0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.03)",
        "chart_font": "#0F172A",
        "chart_grid": "#E2E8F0",
        "chart_palette": ["#047857", "#0369A1", "#6366F1", "#D97706", "#0D9488"],
    },
    "dark": {
        "name": "dark",
        "bg": "#0B1220",
        "surface": "#111C2F",
        "surface_2": "#16243C",
        "surface_3": "#1E2D44",
        "text": "#F1F5F9",
        "text_secondary": "#B6C4D8",
        "text_muted": "#9AABC2",
        "primary": "#34D399",
        "primary_hover": "#10B981",
        "primary_soft": "rgba(52, 211, 153, 0.14)",
        "primary_text": "#34D399",
        "primary_btn_text": "#09101D",
        "secondary": "#38BDF8",
        "secondary_soft": "rgba(56, 189, 248, 0.14)",
        "border": "#2A3A50",
        "border_focus": "#38BDF8",
        "danger": "#F87171",
        "danger_soft": "rgba(248, 113, 113, 0.14)",
        "warning": "#FBBF24",
        "warning_soft": "rgba(251, 191, 36, 0.14)",
        "success": "#34D399",
        "success_soft": "rgba(52, 211, 153, 0.14)",
        "input_text": "#F1F5F9",
        "card_shadow": "0 4px 14px rgba(0, 0, 0, 0.35)",
        "chart_font": "#F1F5F9",
        "chart_grid": "#1E2D44",
        "chart_palette": ["#34D399", "#38BDF8", "#818CF8", "#FBBF24", "#2DD4BF"],
    },
}

# --- Accessible SVG Icon System (Pure SVGs, strictly zero emojis) ---

SVG_ICONS = {
    "analytics": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="m19 9-5 5-4-4-3 3"></path></svg>',
    "shield": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "shield_alert": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    "database": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    "brain": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-5.04z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-5.04z"></path></svg>',
    "check_circle": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    "warning": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    "info": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    "error": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
    "settings": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>',
    "download": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
    "lightbulb": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"></line><line x1="10" y1="22" x2="14" y2="22"></line><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path></svg>',
    "cloud": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path></svg>',
    "lock": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>',
    "unlock": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 9.9-1"></path></svg>',
    "sun": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>',
    "moon": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>',
    "table": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line><line x1="12" y1="3" x2="12" y2="21"></line></svg>',
    "sparkles": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>',
    "arrow_right": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>',
    "file_text": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
    "compass": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon></svg>',
}


def render_icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    """Render a clean, accessible SVG icon string without any emoji characters."""
    template = SVG_ICONS.get(name, SVG_ICONS["info"])
    return template.format(size=size, color=color)


def get_theme_css(theme_name: str = "dark") -> str:
    """Generate scoped CSS rules enforcing the light or dark design system.

    Provides WCAG AA/AAA compliant contrast ratios for headings, body text,
    secondary labels, form controls, radio navigations, toggles, badges, and metrics.
    """
    palette = THEMES.get(theme_name, THEMES["dark"])

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --dp-bg: {palette['bg']};
    --dp-surface: {palette['surface']};
    --dp-surface-2: {palette['surface_2']};
    --dp-surface-3: {palette['surface_3']};
    --dp-text: {palette['text']};
    --dp-text-secondary: {palette['text_secondary']};
    --dp-text-muted: {palette['text_muted']};
    --dp-primary: {palette['primary']};
    --dp-primary-hover: {palette['primary_hover']};
    --dp-primary-soft: {palette['primary_soft']};
    --dp-primary-text: {palette['primary_text']};
    --dp-primary-btn-text: {palette['primary_btn_text']};
    --dp-secondary: {palette['secondary']};
    --dp-secondary-soft: {palette['secondary_soft']};
    --dp-border: {palette['border']};
    --dp-border-focus: {palette['border_focus']};
    --dp-danger: {palette['danger']};
    --dp-danger-soft: {palette['danger_soft']};
    --dp-warning: {palette['warning']};
    --dp-warning-soft: {palette['warning_soft']};
    --dp-success: {palette['success']};
    --dp-success-soft: {palette['success_soft']};
    --dp-input-text: {palette['input_text']};
    --dp-shadow: {palette['card_shadow']};
}}

/* Base Layout & App Container */
.stApp {{
    background-color: var(--dp-bg) !important;
    color: var(--dp-text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

[data-testid="stHeader"] {{
    background-color: var(--dp-bg) !important;
    border-bottom: 1px solid var(--dp-border) !important;
    height: 3.25rem !important;
    z-index: 99 !important;
}}

[data-testid="stSidebar"] {{
    background-color: var(--dp-surface) !important;
    border-right: 1px solid var(--dp-border) !important;
}}

.block-container {{
    padding-top: 4.75rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
    box-sizing: border-box !important;
}}

/* Typography & Streamlit Markdown Containers */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {{
    color: var(--dp-text);
}}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    color: var(--dp-text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.025em !important;
}}

[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b {{
    color: var(--dp-text) !important;
}}

[data-testid="stMarkdownContainer"] code {{
    color: var(--dp-primary) !important;
    background: var(--dp-surface-2) !important;
    border: 1px solid var(--dp-border) !important;
    border-radius: 6px !important;
    padding: 0.15rem 0.4rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.86em !important;
}}

/* Sidebar Navigation Radio Overrides */
[data-testid="stSidebar"] [data-testid="stRadio"] {{
    background: transparent !important;
    padding: 0.2rem 0 !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {{
    gap: 4px !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {{
    background: transparent !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    border: 1px solid transparent !important;
}}

/* Inactive Nav Labels */
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span,
[data-testid="stSidebar"] [data-testid="stRadio"] label div {{
    color: var(--dp-text-secondary) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
}}

/* Nav Hover State */
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background-color: var(--dp-surface-2) !important;
    border-color: var(--dp-border) !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover span,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover div {{
    color: var(--dp-text) !important;
}}

/* Nav Active / Selected State */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {{
    background-color: var(--dp-surface-2) !important;
    border-color: var(--dp-primary) !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] span {{
    color: var(--dp-text) !important;
    font-weight: 700 !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:focus-within {{
    outline: 2px solid var(--dp-border-focus) !important;
    outline-offset: 1px !important;
}}

/* Privacy Mode Toggle & Tooltip Overrides */
div[data-testid="stToggle"] {{
    padding: 0.35rem 0 !important;
}}

div[data-testid="stToggle"] label p,
div[data-testid="stToggle"] label span,
div[data-testid="stToggle"] label div {{
    color: var(--dp-text) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
}}

div[data-testid="stTooltipIcon"],
[data-testid="stTooltipIcon"] {{
    color: var(--dp-text-muted) !important;
}}

div[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipIcon"] svg {{
    fill: var(--dp-text-muted) !important;
    stroke: var(--dp-text-muted) !important;
}}

/* Streamlit Captions & Secondary Text */
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] span,
.stCaption,
.stCaption p {{
    color: var(--dp-text-secondary) !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}}

[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] .stCaption p {{
    color: var(--dp-text-muted) !important;
}}

/* Main Dataset Source Radio Buttons */
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] div[data-baseweb="radio"] div {{
    color: var(--dp-text) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}}

div[data-testid="stRadio"] label:hover p,
div[data-testid="stRadio"] label:hover span {{
    color: var(--dp-text) !important;
}}

div[data-testid="stRadio"] div[data-baseweb="radio"] span[aria-hidden="true"] {{
    border-color: var(--dp-border) !important;
    background-color: var(--dp-surface) !important;
}}

div[data-testid="stRadio"] div[data-baseweb="radio"]:hover span[aria-hidden="true"] {{
    border-color: var(--dp-primary) !important;
}}

/* File Uploader */
[data-testid="stFileUploader"] {{
    background: transparent !important;
}}

[data-testid="stFileUploader"] section {{
    background-color: var(--dp-surface) !important;
    border: 1px dashed var(--dp-border) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    transition: border-color 0.2s ease !important;
}}

[data-testid="stFileUploader"] section:hover {{
    border-color: var(--dp-border-focus) !important;
}}

[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span {{
    color: var(--dp-text-secondary) !important;
}}

[data-testid="stFileUploader"] button {{
    background: var(--dp-surface-2) !important;
    color: var(--dp-text) !important;
    border: 1px solid var(--dp-border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
}}

[data-testid="stFileUploader"] button:hover {{
    background: var(--dp-surface-3) !important;
    border-color: var(--dp-border-focus) !important;
    color: var(--dp-text) !important;
}}

/* General Widget Labels & Controls */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {{
    color: var(--dp-text) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}}

/* Form inputs & selections */
div[data-baseweb="input"] {{
    background-color: var(--dp-surface) !important;
    border: 1px solid var(--dp-border) !important;
    border-radius: 10px !important;
}}

div[data-baseweb="input"]:focus-within {{
    border-color: var(--dp-border-focus) !important;
    box-shadow: 0 0 0 1px var(--dp-border-focus) !important;
}}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {{
    color: var(--dp-input-text) !important;
    background-color: transparent !important;
    font-size: 0.9rem !important;
}}

div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {{
    color: var(--dp-text-muted) !important;
    opacity: 0.8 !important;
}}

div[data-baseweb="select"] > div {{
    background-color: var(--dp-surface) !important;
    border: 1px solid var(--dp-border) !important;
    color: var(--dp-text) !important;
    border-radius: 10px !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
    color: var(--dp-text) !important;
}}

div[data-baseweb="select"] svg {{
    fill: var(--dp-text-muted) !important;
}}

/* Selectbox Dropdown Menu Popover */
div[data-baseweb="popover"] ul[role="listbox"] {{
    background-color: var(--dp-surface) !important;
    border: 1px solid var(--dp-border) !important;
    border-radius: 10px !important;
    box-shadow: var(--dp-shadow) !important;
}}

div[data-baseweb="popover"] li[role="option"] {{
    color: var(--dp-text) !important;
    background-color: transparent !important;
    font-size: 0.88rem !important;
}}

div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {{
    background-color: var(--dp-surface-2) !important;
    color: var(--dp-primary) !important;
}}

/* Button Styling (Both primary and secondary, supporting BaseWeb data-testid and kind) */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {{
    background: var(--dp-primary) !important;
    color: var(--dp-primary-btn-text) !important;
    border: 1px solid var(--dp-primary) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
}}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {{
    background: var(--dp-primary-hover) !important;
    border-color: var(--dp-primary-hover) !important;
    transform: translateY(-1px);
}}

div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="baseButton-secondary"],
div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {{
    border-radius: 10px !important;
    border: 1px solid var(--dp-border) !important;
    background: var(--dp-surface) !important;
    color: var(--dp-text) !important;
    transition: all 0.2s ease !important;
}}

div.stButton > button:hover {{
    border-color: var(--dp-border-focus) !important;
    background: var(--dp-surface-2) !important;
    color: var(--dp-text) !important;
}}

/* Streamlit Native Metric cards normalized */
[data-testid="stMetric"] {{
    background: var(--dp-surface) !important;
    border: 1px solid var(--dp-border) !important;
    border-radius: 14px !important;
    padding: 1.15rem 1.25rem !important;
    box-shadow: var(--dp-shadow) !important;
    min-height: 124px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-sizing: border-box !important;
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span,
[data-testid="stMetricLabel"] label {{
    color: var(--dp-text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    color: var(--dp-text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] div,
[data-testid="stMetricDelta"] span {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--dp-text-secondary) !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

[data-testid="stMetricDelta"] svg {{
    fill: var(--dp-text-secondary) !important;
}}

/* DataFrames & Tables */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--dp-border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background-color: var(--dp-surface) !important;
}}

[data-testid="stTable"] {{
    border: 1px solid var(--dp-border) !important;
    border-radius: 12px !important;
    color: var(--dp-text) !important;
}}

[data-testid="stTable"] th {{
    background-color: var(--dp-surface-2) !important;
    color: var(--dp-text) !important;
    border-bottom: 1px solid var(--dp-border) !important;
    font-weight: 600 !important;
}}

[data-testid="stTable"] td {{
    border-bottom: 1px solid var(--dp-border) !important;
    color: var(--dp-text-secondary) !important;
}}

/* Expanders */
[data-testid="stExpander"] {{
    border-radius: 12px !important;
    border: 1px solid var(--dp-border) !important;
    background: var(--dp-surface) !important;
    margin-bottom: 0.75rem !important;
}}

[data-testid="stExpander"] details {{
    border: none !important;
}}

[data-testid="stExpander"] summary {{
    color: var(--dp-text) !important;
    font-weight: 600 !important;
}}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {{
    color: var(--dp-text) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}}

[data-testid="stExpander"] summary svg {{
    fill: var(--dp-text-muted) !important;
}}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    border-top: 1px solid var(--dp-border) !important;
    padding: 1rem !important;
    background: var(--dp-surface) !important;
}}

/* Native Containers with Border */
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border-radius: 14px !important;
    border-color: var(--dp-border) !important;
    background-color: var(--dp-surface) !important;
}}

/* Hero Section Container Customization */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dp-hero-marker) > div {{
    background: linear-gradient(135deg, var(--dp-surface) 0%, var(--dp-surface-2) 100%) !important;
    border: 1px solid var(--dp-border) !important;
    border-radius: 18px !important;
    padding: 1.75rem 2rem !important;
    box-shadow: var(--dp-shadow) !important;
    margin-bottom: 1.25rem !important;
}}

/* Native Alert boxes */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: 1px solid var(--dp-border) !important;
    background-color: var(--dp-surface) !important;
}}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {{
    color: var(--dp-text) !important;
    font-size: 0.88rem !important;
}}

div[data-testid="stAlert"] code {{
    color: var(--dp-primary) !important;
}}

/* Disabled States */
button:disabled,
input:disabled,
select:disabled,
[aria-disabled="true"],
[disabled] {{
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    color: var(--dp-text-muted) !important;
}}

/* Reusable Container & Card Components */
.dp-card {{
    background: var(--dp-surface);
    border: 1px solid var(--dp-border);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: var(--dp-shadow);
    box-sizing: border-box;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.dp-card:hover {{
    border-color: var(--dp-border-focus);
}}

/* Hero Experience Typography & Elements */
.dp-hero-banner {{
    background: linear-gradient(135deg, var(--dp-surface) 0%, var(--dp-surface-2) 100%);
    border: 1px solid var(--dp-border);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--dp-shadow);
    box-sizing: border-box;
}}

.dp-hero-copy {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
}}

.dp-brand-title {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(1.85rem, 3.2vw, 2.75rem);
    font-weight: 800;
    line-height: 1.18;
    letter-spacing: -0.035em;
    color: var(--dp-text);
    margin: 0.5rem 0 0.75rem 0;
}}

.dp-subtitle {{
    font-size: 1.02rem;
    line-height: 1.62;
    color: var(--dp-text-secondary);
    max-width: 580px;
    margin: 0 0 1.25rem 0;
}}

.dp-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dp-primary);
}}

.dp-kicker {{
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--dp-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.dp-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.28rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
    line-height: 1;
    white-space: nowrap;
}}
.dp-badge-primary {{
    background: var(--dp-primary-soft);
    color: var(--dp-primary-text);
    border: 1px solid var(--dp-primary);
}}
.dp-badge-success {{
    background: var(--dp-success-soft);
    color: var(--dp-success);
    border: 1px solid var(--dp-success);
}}
.dp-badge-warning {{
    background: var(--dp-warning-soft);
    color: var(--dp-warning);
    border: 1px solid var(--dp-warning);
}}
.dp-badge-danger {{
    background: var(--dp-danger-soft);
    color: var(--dp-danger);
    border: 1px solid var(--dp-danger);
}}
.dp-badge-secondary {{
    background: var(--dp-secondary-soft);
    color: var(--dp-secondary);
    border: 1px solid var(--dp-secondary);
}}

/* Audit Process Pipeline Card */
.dp-pipeline-card {{
    background: var(--dp-surface);
    border: 1px solid var(--dp-border);
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    box-shadow: var(--dp-shadow);
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    transition: border-color 0.2s ease;
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    overflow: hidden;
}}

.dp-pipeline-canvas svg {{
    width: 100%;
    height: auto;
    max-height: 180px;
    display: block;
}}

@keyframes dpSignalMove {{
    0% {{ transform: translateX(0px); opacity: 0; }}
    4% {{ opacity: 1; }}
    86% {{ opacity: 1; }}
    98%, 100% {{ transform: translateX(330px); opacity: 0; }}
}}

.dp-signal-mover {{
    animation: dpSignalMove 4.6s infinite cubic-bezier(0.4, 0, 0.2, 1);
}}

.dp-pipeline-card:hover {{
    border-color: var(--dp-border-focus);
}}

.dp-pipeline-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.65rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--dp-border);
}}

.dp-pipeline-tag {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dp-text-muted);
}}

.dp-pulse-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--dp-primary);
    box-shadow: 0 0 8px var(--dp-primary);
    animation: dpPulseDot 2.4s infinite ease-in-out;
}}

@keyframes dpPulseDot {{
    0%, 100% {{ transform: scale(1); opacity: 0.8; }}
    50% {{ transform: scale(1.35); opacity: 1; }}
}}

.dp-pipeline-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--dp-primary);
    background: var(--dp-primary-soft);
    padding: 0.18rem 0.55rem;
    border-radius: 9999px;
    border: 1px solid var(--dp-primary);
}}

.dp-pipeline-canvas {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}}

@keyframes dpNodeHighlight {{
    0%, 100% {{ stroke: var(--dp-border); }}
    50% {{ stroke: var(--dp-primary); }}
}}

.dp-node-ring {{
    transition: stroke 0.3s ease;
}}

.dp-node-step1 .dp-node-ring {{
    animation: dpNodeHighlight 4.6s infinite ease-in-out 0s;
}}

.dp-node-step2 .dp-node-ring {{
    animation: dpNodeHighlight 4.6s infinite ease-in-out 1.15s;
}}

.dp-node-step3 .dp-node-ring {{
    animation: dpNodeHighlight 4.6s infinite ease-in-out 2.3s;
}}

.dp-node-ring-verified {{
    box-shadow: 0 0 10px var(--dp-primary);
}}

/* AI Chat Container */
.dp-chat-bubble-user {{
    background: var(--dp-surface-2);
    border: 1px solid var(--dp-border);
    border-radius: 14px 14px 2px 14px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.75rem;
    color: var(--dp-text);
}}
.dp-chat-bubble-ai {{
    background: var(--dp-surface);
    border: 1px solid var(--dp-border);
    border-radius: 14px 14px 14px 2px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--dp-shadow);
    color: var(--dp-text);
}}

hr {{
    border-color: var(--dp-border) !important;
    margin: 1.5rem 0 !important;
}}

/* Responsive Breakpoints */
@media (max-width: 1024px) {{
    .block-container {{
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 4.5rem !important;
    }}
}}

@media (max-width: 768px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 4.25rem !important;
    }}
    .dp-brand-title {{
        font-size: 1.7rem !important;
    }}
    .dp-hero-banner {{
        padding: 1.25rem 1.2rem !important;
        border-radius: 14px !important;
    }}
    [data-testid="stMetric"] {{
        min-height: auto !important;
        margin-bottom: 0.5rem !important;
    }}
}}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {{
    * {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }}
    .dp-pulse-dot,
    .dp-signal-pulse,
    .dp-node-group .dp-node-ring {{
        animation: none !important;
    }}
    .dp-signal-pulse {{
        display: none !important;
    }}
}}
</style>
"""


def get_audit_pipeline_visual_html(theme_name: str = "dark") -> str:
    """Generate an accessible, lightweight SVG representing DataPilot AI's 4-stage audit process.

    Visual sequence:
    1. Data enters the pipeline (Tabular Ingestion).
    2. Quality and leakage checks inspect it (Anomaly & Leakage Audit).
    3. Model evaluation compares results (Isolated Holdout Benchmarks).
    4. Evidence reaches a verified state (Grounded Prediction Readiness).

    Rendered with zero indentation to ensure Streamlit / Markdown parsers never treat it as an indented code block.
    Respects prefers-reduced-motion, uses pure inline SVG with zero remote assets.
    """
    raw = """<div class="dp-pipeline-card">
<div class="dp-pipeline-header">
<span class="dp-pipeline-tag">
<span class="dp-pulse-dot"></span>
<span>AUDIT PIPELINE VERIFICATION</span>
</span>
<span class="dp-pipeline-badge">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--dp-primary)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:3px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
<span>Deterministic Flow</span>
</span>
</div>
<div class="dp-pipeline-canvas">
<svg viewBox="0 0 440 156" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" aria-label="DataPilot AI 4-stage audit pipeline: 1 Data Ingestion, 2 Quality and Leakage Audit, 3 Holdout Model Evaluation, 4 Verified Evidence">
<defs>
<linearGradient id="dpPipelineTrack" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="var(--dp-border)" />
<stop offset="65%" stop-color="var(--dp-border)" />
<stop offset="100%" stop-color="var(--dp-primary)" />
</linearGradient>
<filter id="dpSignalGlow" x="-30%" y="-30%" width="160%" height="160%">
<feGaussianBlur stdDeviation="3" result="blur" />
<feComposite in="SourceGraphic" in2="blur" operator="over" />
</filter>
</defs>
<path d="M 55 50 L 385 50" stroke="url(#dpPipelineTrack)" stroke-width="2.5" stroke-linecap="round" fill="none" class="dp-track-base" />
<g class="dp-signal-mover">
<circle cx="55" cy="50" r="4.5" fill="var(--dp-primary)" filter="url(#dpSignalGlow)" class="dp-signal-pulse">
<animateMotion path="M 0 0 L 330 0" dur="4.6s" repeatCount="indefinite" />
</circle>
</g>
<g class="dp-node-group dp-node-step1" transform="translate(55, 50)">
<circle r="17" fill="var(--dp-surface)" stroke="var(--dp-border)" stroke-width="2" class="dp-node-ring" />
<path d="M -5 -3 a 5 1.8 0 0 1 10 0 v 6 a 5 1.8 0 0 1 -10 0 z M -5 0 a 5 1.8 0 0 0 10 0 M -5 3 a 5 1.8 0 0 0 10 0" fill="none" stroke="var(--dp-text)" stroke-width="1.3" stroke-linecap="round" />
<text y="28" text-anchor="middle" font-family="'Plus Jakarta Sans', Inter, sans-serif" font-weight="700" font-size="11" fill="var(--dp-text)">01 Ingest</text>
<text y="41" text-anchor="middle" font-family="'Inter', sans-serif" font-size="9" fill="var(--dp-text-secondary)">Tabular Input</text>
</g>
<g class="dp-node-group dp-node-step2" transform="translate(165, 50)">
<circle r="17" fill="var(--dp-surface)" stroke="var(--dp-border)" stroke-width="2" class="dp-node-ring" />
<path d="M 0 -7 L 6 -3.5 V 1 C 6 5.5 0 8 0 8 C 0 8 -6 5.5 -6 1 V -3.5 Z" fill="none" stroke="var(--dp-text)" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
<text y="28" text-anchor="middle" font-family="'Plus Jakarta Sans', Inter, sans-serif" font-weight="700" font-size="11" fill="var(--dp-text)">02 Audit</text>
<text y="41" text-anchor="middle" font-family="'Inter', sans-serif" font-size="9" fill="var(--dp-text-secondary)">Quality &amp; Leakage</text>
</g>
<g class="dp-node-group dp-node-step3" transform="translate(275, 50)">
<circle r="17" fill="var(--dp-surface)" stroke="var(--dp-border)" stroke-width="2" class="dp-node-ring" />
<path d="M -5 5 L -5 -5 M 5 5 L 5 -2 M 0 5 L 0 1.5" fill="none" stroke="var(--dp-text)" stroke-width="1.4" stroke-linecap="round" />
<text y="28" text-anchor="middle" font-family="'Plus Jakarta Sans', Inter, sans-serif" font-weight="700" font-size="11" fill="var(--dp-text)">03 Evaluate</text>
<text y="41" text-anchor="middle" font-family="'Inter', sans-serif" font-size="9" fill="var(--dp-text-secondary)">Holdout Models</text>
</g>
<g class="dp-node-group dp-node-step4" transform="translate(385, 50)">
<circle r="17" fill="var(--dp-primary-soft)" stroke="var(--dp-primary)" stroke-width="2" class="dp-node-ring dp-node-ring-verified" />
<path d="M -4.5 0 L -1 3.5 L 5 -2.5" fill="none" stroke="var(--dp-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
<text y="28" text-anchor="middle" font-family="'Plus Jakarta Sans', Inter, sans-serif" font-weight="800" font-size="11" fill="var(--dp-primary)">04 Verify</text>
<text y="41" text-anchor="middle" font-family="'Inter', sans-serif" font-size="9" fill="var(--dp-text-secondary)">Grounded Trust</text>
</g>
<g transform="translate(45, 118)">
<rect x="0" y="0" width="350" height="25" rx="6" fill="var(--dp-surface-2)" stroke="var(--dp-border)" stroke-width="1" />
<circle cx="16" cy="12.5" r="3" fill="var(--dp-primary)" />
<text x="28" y="16" font-family="'Inter', sans-serif" font-size="9.2" font-weight="600" fill="var(--dp-text-secondary)">Isolated Test Holdouts · Zero Target Leakage · Grounded Guidance</text>
</g>
</svg>
</div>
</div>"""
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def get_welcome_screen_html(theme_name: str = "dark") -> str:
    """Generate a responsive animated welcome overlay with zero third-party scripts.

    Runs on initial entry to present the DataPilot AI brand identity and data activation nodes.
    Disappears cleanly into the workspace after ~1.2s without blocking the python runtime.
    """
    palette = THEMES.get(theme_name, THEMES["dark"])

    return f"""
<div id="dp-welcome-overlay" class="dp-welcome-container">
    <div class="dp-welcome-modal">
        <div class="dp-welcome-nodes">
            <div class="dp-node dp-node-1"></div>
            <div class="dp-node dp-node-2"></div>
            <div class="dp-node dp-node-3"></div>
            <div class="dp-line dp-line-1"></div>
            <div class="dp-line dp-line-2"></div>
        </div>
        <div class="dp-welcome-brand">
            <span class="dp-welcome-logo">{render_icon("compass", size=32, color=palette['primary'])}</span>
            <span class="dp-welcome-title">DataPilot <span class="dp-welcome-accent">AI</span></span>
        </div>
        <div class="dp-welcome-tagline">Understand your data. Trust your evidence.</div>
        <div class="dp-welcome-status">
            <div class="dp-welcome-spinner"></div>
            <span>Initializing prediction audit workspace&hellip;</span>
        </div>
    </div>
</div>

<style>
.dp-welcome-container {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: {palette['bg']};
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: dpFadeOut 0.35s ease-in-out 1.2s forwards;
    pointer-events: none !important;
}}

.dp-welcome-modal {{
    text-align: center;
    max-width: 480px;
    padding: 2.5rem 2rem;
    animation: dpScaleUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

.dp-welcome-nodes {{
    position: relative;
    width: 140px;
    height: 60px;
    margin: 0 auto 1.5rem;
}}

.dp-node {{
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: {palette['primary']};
    box-shadow: 0 0 16px {palette['primary']};
    animation: dpPulse 1.6s infinite alternate ease-in-out;
}}
.dp-node-1 {{ top: 10px; left: 10px; }}
.dp-node-2 {{ top: 35px; left: 65px; animation-delay: 0.25s; }}
.dp-node-3 {{ top: 8px; left: 115px; animation-delay: 0.5s; }}

.dp-line {{
    position: absolute;
    height: 2px;
    background: linear-gradient(90deg, {palette['primary']}, {palette['secondary']});
    opacity: 0.6;
}}
.dp-line-1 {{
    width: 60px;
    top: 24px;
    left: 18px;
    transform: rotate(24deg);
    animation: dpLineGrow 0.7s ease-out forwards;
}}
.dp-line-2 {{
    width: 58px;
    top: 25px;
    left: 72px;
    transform: rotate(-26deg);
    animation: dpLineGrow 0.7s 0.15s ease-out forwards;
}}

.dp-welcome-brand {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}}

.dp-welcome-title {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: {palette['text']};
}}

.dp-welcome-accent {{
    color: {palette['primary']};
}}

.dp-welcome-tagline {{
    font-size: 0.98rem;
    color: {palette['text_secondary']};
    margin-bottom: 1.8rem;
    font-weight: 500;
}}

.dp-welcome-status {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    color: {palette['text_muted']};
    background: {palette['surface_2']};
    border: 1px solid {palette['border']};
    padding: 0.5rem 1rem;
    border-radius: 9999px;
}}

.dp-welcome-spinner {{
    width: 12px;
    height: 12px;
    border: 2px solid {palette['border']};
    border-top-color: {palette['primary']};
    border-radius: 50%;
    animation: dpSpin 0.8s linear infinite;
}}

@keyframes dpSpin {{
    to {{ transform: rotate(360deg); }}
}}
@keyframes dpPulse {{
    0% {{ transform: scale(0.9); opacity: 0.7; }}
    100% {{ transform: scale(1.2); opacity: 1; }}
}}
@keyframes dpLineGrow {{
    from {{ transform: scaleX(0); }}
    to {{ transform: scaleX(1); }}
}}
@keyframes dpScaleUp {{
    from {{ transform: translateY(12px) scale(0.96); opacity: 0; }}
    to {{ transform: translateY(0) scale(1); opacity: 1; }}
}}
@keyframes dpFadeOut {{
    0% {{ opacity: 1; visibility: visible; }}
    99% {{ opacity: 0; visibility: visible; }}
    100% {{ opacity: 0; visibility: hidden; pointer-events: none; }}
}}

@media (prefers-reduced-motion: reduce) {{
    .dp-welcome-container {{
        display: none !important;
    }}
}}
</style>
"""


def apply_chart_theme(fig, theme_name: str = "dark"):
    """Theme a Plotly figure to cleanly match the active Light or Dark system."""
    palette = THEMES.get(theme_name, THEMES["dark"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color=palette["chart_font"],
            size=12,
        ),
        margin=dict(l=10, r=10, t=24, b=10),
    )
    fig.update_xaxes(
        gridcolor=palette["chart_grid"],
        zerolinecolor=palette["chart_grid"],
        tickfont=dict(color=palette["text_muted"]),
    )
    fig.update_yaxes(
        gridcolor=palette["chart_grid"],
        zerolinecolor=palette["chart_grid"],
        tickfont=dict(color=palette["text_muted"]),
    )
    return fig
