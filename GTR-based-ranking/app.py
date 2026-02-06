"""
GTR Domain Activity & Coverage Intelligence System (DACIS)
Main Streamlit Dashboard Application - UX Optimized
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tldextract
from datetime import datetime, timedelta
import io

# Page config
st.set_page_config(
    page_title="DACIS - Domain Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for cleaner look
)

# Custom CSS for better UX
st.markdown("""
<style>
    /* Tighter spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    /* Better header styling */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
        display: inline;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    
    /* Compact dataframe */
    .stDataFrame {
        font-size: 0.85rem;
    }
    
    /* Better sidebar */
    section[data-testid="stSidebar"] {
        width: 280px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    
    /* Quick action buttons */
    .stButton > button {
        border-radius: 4px;
        font-size: 0.85rem;
    }
    
    /* Status badges */
    .status-active { color: #e74c3c; font-weight: 600; }
    .status-low { color: #f39c12; }
    .status-declining { color: #e67e22; }
    .status-inactive { color: #95a5a6; }
    
    /* Hide hamburger menu on sidebar */
    #MainMenu {visibility: hidden;}
    
    /* Tighter tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_gtr_data(use_full_dataset: bool = False):
    """Load and cache the GTR enriched data."""
    base_path = Path(__file__).parent.parent / "data" / "processed"
    sample_path = Path(__file__).parent / "sample_data.parquet"
    
    # Try full data paths first, fall back to sample for cloud deployment
    if use_full_dataset:
        data_path = base_path / "google_transparency_enriched.parquet"
    else:
        data_path = base_path / "video_piracy_clean.parquet"
    
    if not data_path.exists():
        # Fall back to sample data (for Streamlit Cloud)
        if sample_path.exists():
            data_path = sample_path
        else:
            st.error("No data file found. Please ensure data files are available.")
            st.stop()
    
    df = pd.read_parquet(data_path)
    
    # Convert date columns
    for col in ['first_major_org_date', 'last_major_org_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = enrich_domain_data(df)
    return df


def enrich_domain_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add DACIS-specific enrichment fields."""
    today = pd.Timestamp.now()
    
    # Days since last activity
    df['days_since_last_activity'] = df['last_major_org_date'].apply(
        lambda x: (today - x).days if pd.notna(x) else 9999
    )
    
    # Operational status classification
    def classify_status(row):
        if pd.isna(row['last_major_org_date']):
            return 'Unknown'
        days = row['days_since_last_activity']
        velocity = row['major_org_velocity_per_month']
        
        if days <= 90 and velocity > 10:
            return 'Active'
        elif days <= 180:
            return 'Low Activity'
        elif days <= 365:
            return 'Declining'
        else:
            return 'Inactive'
    
    df['status'] = df.apply(classify_status, axis=1)
    
    # Trend classification
    def classify_trend(row):
        last_30 = row['major_org_requests_last_30d']
        last_90 = row['major_org_requests_last_90d']
        if last_90 == 0:
            return 'No Data'
        monthly_avg = last_90 / 3
        if last_30 > monthly_avg * 1.5:
            return 'Rising'
        elif last_30 < monthly_avg * 0.5:
            return 'Declining'
        return 'Stable'
    
    df['trend'] = df.apply(classify_trend, axis=1)
    
    # Notice tier for quick filtering
    def tier(val):
        if val >= 1_000_000: return '1M+'
        if val >= 100_000: return '100K-1M'
        if val >= 10_000: return '10K-100K'
        if val >= 1_000: return '1K-10K'
        return '<1K'
    
    df['volume_tier'] = df['total_urls_removed'].apply(tier)
    
    # Handle missing columns gracefully
    if 'tranco_rank' not in df.columns:
        df['tranco_rank'] = pd.NA
    
    return df


def clean_domain(domain: str) -> str:
    """Clean domain to base domain + TLD."""
    try:
        ext = tldextract.extract(domain.lower().strip())
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return domain.lower().strip()
    except:
        return domain.lower().strip()


def format_number(n):
    """Format large numbers compactly."""
    if pd.isna(n):
        return "—"
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def format_date(d):
    """Format date nicely."""
    if pd.isna(d):
        return "—"
    return d.strftime("%b %d, %Y")


def render_quick_stats(df: pd.DataFrame, compact: bool = False):
    """Compact stats bar."""
    if compact:
        # Ultra-compact single line
        active = len(df[df['status'] == 'Active'])
        rising = len(df[df['trend'] == 'Rising'])
        st.caption(f"📊 {len(df):,} domains · {active:,} active · {rising:,} rising · {format_number(df['total_urls_removed'].sum())} removed")
        return
    
    cols = st.columns(5)
    active = len(df[df['status'] == 'Active'])
    rising = len(df[df['trend'] == 'Rising'])
    high_vol = len(df[df['total_urls_removed'] >= 100_000])
    
    with cols[0]:
        st.metric("Domains", f"{len(df):,}")
    with cols[1]:
        st.metric("Active", f"{active:,}", f"{active/len(df)*100:.1f}%" if len(df) > 0 else None)
    with cols[2]:
        st.metric("Rising", f"{rising:,}")
    with cols[3]:
        st.metric("High Volume", f"{high_vol:,}", "≥100K removals")
    with cols[4]:
        st.metric("Total Removed", format_number(df['total_urls_removed'].sum()))


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter row with dropdowns always visible."""
    
    # All filters in one row with consistent labels
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    
    with c1:
        search = st.text_input("Search", placeholder="Enter domain...", label_visibility="visible")
    with c2:
        status_filter = st.multiselect("Status", ['Active', 'Low Activity', 'Declining', 'Inactive', 'Unknown'], placeholder="All")
    with c3:
        trend_filter = st.multiselect("Trend", ['Rising', 'Stable', 'Declining', 'No Data'], placeholder="All")
    with c4:
        tier_filter = st.multiselect("Volume", ['1M+', '100K-1M', '10K-100K', '1K-10K', '<1K'], placeholder="All")
    
    # Apply filters
    filtered = df.copy()
    if search:
        filtered = filtered[filtered['Domain'].str.contains(search.lower(), case=False, na=False)]
    if status_filter:
        filtered = filtered[filtered['status'].isin(status_filter)]
    if trend_filter:
        filtered = filtered[filtered['trend'].isin(trend_filter)]
    if tier_filter:
        filtered = filtered[filtered['volume_tier'].isin(tier_filter)]
    
    return filtered


def render_domain_table(df: pd.DataFrame):
    """Domain table with pagination controls below."""
    
    # Sort control - compact
    sort_options = {
        'URLs Removed ↓': ('total_urls_removed', False),
        'URLs Removed ↑': ('total_urls_removed', True),
        'Recent Activity ↓': ('major_org_requests_last_30d', False),
        'Monthly Velocity ↓': ('major_org_velocity_per_month', False),
        'Last Active ↓': ('last_major_org_date', False),
        'Domain A-Z': ('Domain', True),
    }
    sort_col1, sort_col2 = st.columns([1, 4])
    with sort_col1:
        sort_choice = st.selectbox("Sort by", options=list(sort_options.keys()), index=0, label_visibility="collapsed")
    sort_col, ascending = sort_options[sort_choice]
    
    # Sort data
    sorted_df = df.sort_values(sort_col, ascending=ascending, na_position='last')
    
    # Get pagination state
    page_size = st.session_state.get('page_size', 50)
    page = st.session_state.get('page', 1)
    total_pages = max(1, (len(sorted_df) - 1) // page_size + 1)
    page = min(page, total_pages)  # Ensure page is valid
    
    start = (page - 1) * page_size
    end = start + page_size
    page_data = sorted_df.iloc[start:end]
    
    # Prepare display dataframe
    display_df = page_data[['Domain', 'status', 'trend', 'total_urls_removed', 'major_org_requests_last_30d', 'unique_major_studios']].copy()
    display_df.columns = ['Domain', 'Status', 'Trend', 'URLs Removed', 'Last 30d', 'Studios']
    
    # Show table
    st.dataframe(display_df, height=450)
    
    # Pagination controls BELOW table
    st.markdown("")  # Small spacer
    pc1, pc2, pc3, pc4, pc5 = st.columns([2, 1, 2, 1, 2])
    
    with pc1:
        st.caption(f"Showing {start+1}–{min(end, len(sorted_df))} of {len(sorted_df):,}")
    
    with pc2:
        if st.button("← Prev", disabled=(page <= 1), use_container_width=True):
            st.session_state['page'] = page - 1
            st.rerun()
    
    with pc3:
        st.markdown(f"<div style='text-align:center'>Page <b>{page}</b> of {total_pages}</div>", unsafe_allow_html=True)
    
    with pc4:
        if st.button("Next →", disabled=(page >= total_pages), use_container_width=True):
            st.session_state['page'] = page + 1
            st.rerun()
    
    with pc5:
        new_size = st.selectbox("Rows", options=[25, 50, 100, 200], index=[25, 50, 100, 200].index(page_size), label_visibility="collapsed")
        if new_size != page_size:
            st.session_state['page_size'] = new_size
            st.session_state['page'] = 1
            st.rerun()
    
    return sorted_df


def render_charts(df: pd.DataFrame):
    """Compact visualization section."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Status donut
        status_counts = df['status'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.5,
            marker_colors=['#e74c3c', '#f39c12', '#e67e22', '#95a5a6', '#bdc3c7']
        )])
        fig.update_layout(
            title=dict(text="By Status", font=dict(size=14)),
            margin=dict(t=40, b=20, l=20, r=20),
            height=280,
            showlegend=True,
            legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Trend distribution
        trend_counts = df['trend'].value_counts()
        colors = {'Rising': '#2ecc71', 'Stable': '#3498db', 'Declining': '#e74c3c', 'No Data': '#95a5a6'}
        fig = go.Figure(data=[go.Bar(
            x=trend_counts.index,
            y=trend_counts.values,
            marker_color=[colors.get(t, '#95a5a6') for t in trend_counts.index]
        )])
        fig.update_layout(
            title=dict(text="By Trend", font=dict(size=14)),
            margin=dict(t=40, b=20, l=20, r=20),
            height=280,
            xaxis_title=None,
            yaxis_title=None
        )
        st.plotly_chart(fig, width="stretch")
    
    with col3:
        # Volume tier
        tier_order = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M+']
        tier_counts = df['volume_tier'].value_counts().reindex(tier_order, fill_value=0)
        fig = go.Figure(data=[go.Bar(
            x=tier_counts.index,
            y=tier_counts.values,
            marker_color='#3498db'
        )])
        fig.update_layout(
            title=dict(text="By Volume", font=dict(size=14)),
            margin=dict(t=40, b=20, l=20, r=20),
            height=280,
            xaxis_title=None,
            yaxis_title=None
        )
        st.plotly_chart(fig, width="stretch")


def render_csv_upload(full_df: pd.DataFrame):
    """Streamlined CSV upload."""
    
    st.markdown("Upload a list of domains to check their status in our database.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drop CSV or TXT file",
            type=['csv', 'txt'],
            label_visibility="collapsed"
        )
    
    if uploaded_file:
        try:
            # Parse file
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                domains = [d.strip() for d in content.split('\n') if d.strip()]
                upload_df = pd.DataFrame({'domain': domains})
            else:
                upload_df = pd.read_csv(uploaded_file)
            
            # Find domain column
            domain_col = None
            for col in upload_df.columns:
                if 'domain' in col.lower() or 'url' in col.lower() or 'site' in col.lower():
                    domain_col = col
                    break
            if domain_col is None:
                domain_col = upload_df.columns[0]
            
            # Clean and match
            upload_df['_clean'] = upload_df[domain_col].apply(clean_domain)
            matched = upload_df.merge(full_df, left_on='_clean', right_on='Domain', how='left')
            
            # Summary stats
            total = len(matched)
            found = matched['Domain'].notna().sum()
            active = len(matched[matched['status'] == 'Active'])
            
            stat_cols = st.columns(4)
            stat_cols[0].metric("Uploaded", total)
            stat_cols[1].metric("Found", found, f"{found/total*100:.0f}%")
            stat_cols[2].metric("Active", active)
            stat_cols[3].metric("Not Found", total - found)
            
            # Results table
            result_df = pd.DataFrame({
                'Input': matched[domain_col],
                'Domain': matched['Domain'].fillna('—'),
                'Status': matched['status'].fillna('Not Found'),
                'Trend': matched['trend'].fillna('—'),
                'URLs Removed': matched['total_urls_removed'].apply(lambda x: format_number(x) if pd.notna(x) else '—'),
                'Last Active': matched['last_major_org_date'].apply(lambda x: format_date(x) if pd.notna(x) else '—'),
            })
            
            st.dataframe(result_df, width="stretch", height=400, hide_index=True)
            
            # Download
            csv_out = matched.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results",
                csv_out,
                file_name=f"dacis_lookup_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {e}")


def render_domain_detail(df: pd.DataFrame):
    """Clean domain detail view."""
    
    search = st.text_input("Enter domain", placeholder="rapidgator.net")
    
    if not search:
        st.info("Enter a domain above to see detailed intelligence.")
        return
    
    clean = clean_domain(search)
    matches = df[df['Domain'].str.lower() == clean.lower()]
    
    if len(matches) == 0:
        st.warning(f"'{clean}' not found in database.")
        return
    
    row = matches.iloc[0]
    
    # Status header
    status_emoji = {'Active': '🔴', 'Low Activity': '🟡', 'Declining': '🟠', 'Inactive': '⚫', 'Unknown': '⚪'}
    trend_emoji = {'Rising': '📈', 'Stable': '➡️', 'Declining': '📉', 'No Data': '—'}
    
    st.markdown(f"### {row['Domain']}")
    st.markdown(f"{status_emoji.get(row['status'], '❓')} **{row['status']}** · {trend_emoji.get(row['trend'], '—')} {row['trend']}")
    
    st.divider()
    
    # Three column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Notice Activity**")
        st.metric("URLs Removed", format_number(row['total_urls_removed']))
        st.metric("Total Requests", format_number(row['total_requests']))
        st.metric("Removal Rate", f"{row['removal_rate']*100:.1f}%")
    
    with col2:
        st.markdown("**Recent Activity**")
        st.metric("Last 30 Days", format_number(row['major_org_requests_last_30d']))
        st.metric("Last 90 Days", format_number(row['major_org_requests_last_90d']))
        st.metric("Monthly Velocity", format_number(row['major_org_velocity_per_month']))
    
    with col3:
        st.markdown("**Enforcement**")
        st.write(f"**Major Orgs:** {row['unique_major_orgs']}")
        st.write(f"**Studios:** {row['unique_major_studios']}")
        st.write(f"**First Seen:** {format_date(row['first_major_org_date'])}")
        st.write(f"**Last Active:** {format_date(row['last_major_org_date'])}")
        if row['days_since_last_activity'] < 9999:
            st.write(f"**Days Inactive:** {row['days_since_last_activity']:.0f}")


def main():
    """Main application."""
    
    # Minimal header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 🔍 Domain Activity & Coverage Intelligence System")
    with col2:
        use_full = st.checkbox("Full dataset (6M)", value=False, help="Load complete GTR data")
    
    # Load data
    with st.spinner("Loading..."):
        try:
            df = load_gtr_data(use_full_dataset=use_full)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📋 Domains", "📤 Upload", "🔎 Lookup"])
    
    with tab1:
        # Metric cards at top
        m1, m2, m3, m4 = st.columns(4)
        active = len(df[df['status'] == 'Active'])
        rising = len(df[df['trend'] == 'Rising'])
        high_vol = len(df[df['total_urls_removed'] >= 100_000])
        
        with m1:
            st.metric("Total Domains", f"{len(df):,}")
        with m2:
            st.metric("Active", f"{active:,}", f"{active/len(df)*100:.1f}%")
        with m3:
            st.metric("Rising Trend", f"{rising:,}")
        with m4:
            st.metric("High Volume", f"{high_vol:,}", "≥100K")
        
        # Filters
        filtered_df = render_filters(df)
        
        # Show filter status if filtered
        if len(filtered_df) != len(df):
            st.caption(f"Filtered: {len(filtered_df):,} of {len(df):,} domains")
        
        # Table
        sorted_df = render_domain_table(filtered_df)
        
        # Bottom row: export + charts
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button("📥 Export", filtered_df.to_csv(index=False), f"dacis_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        with st.expander("📈 Charts"):
            render_charts(filtered_df)
    
    with tab2:
        render_csv_upload(df)
    
    with tab3:
        render_domain_detail(df)
    
    # Footer
    st.divider()
    st.caption(f"Data: {'Full GTR' if use_full else 'Video Piracy Subset'} · {len(df):,} domains · Updated {datetime.now().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
