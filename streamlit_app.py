import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# -------------------- Constants --------------------
NUM_PROJECTS = 5000
STRIPE_PERCENT = 0.029
STRIPE_FIXED = 0.30
STRIPE_TRANSFER_PERCENT = 0.03
STRIPE_TRANSFER_FIXED = 0.50
PROJECT_TYPES = ["code", "docs", "image", "video"]
INFRA_COSTS = {
    "code": 0.20,
    "docs": 0.08,
    "image": 0.10,
    "video": {"free_gb": 2, "cost_per_gb": 0.10},
}

# CAC and Team Costs
CAC_PER_CUSTOMER = 10
TEAM_SALARIES = {
    "developers": {"count": 2, "monthly_salary": 8000},
    "marketer": {"count": 1, "monthly_salary": 6000},
    "support": {"count": 1, "monthly_salary": 4000}
}
PAYROLL_MULTIPLIER = 1.15  # Benefits/overhead

st.set_page_config(page_title="bl0q Platform Simulator", layout="wide", page_icon="💡")

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1f2937;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00D4FF;
        margin: 0.5rem 0;
    }
    .stMetric {
        background: #1f2937;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background: #1f2937;
        border: 1px solid #374151;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] > div {
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: #111827;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚀 bl0q Platform Financial Simulator</h1><p>Comprehensive financial modeling with AI-powered forecasting, team cost management, and advanced analytics</p></div>', unsafe_allow_html=True)

# -------------------- Sidebar Controls --------------------
st.sidebar.header("🎛️ Simulation Controls")

# Project simulation controls
with st.sidebar.expander("📊 Project Parameters", expanded=True):
    num_projects = st.slider("Number of Projects", 100, 20000, NUM_PROJECTS, 500)
    project_min = st.number_input("Min Project Value ($)", 10, 1000, 75)
    project_max = st.number_input("Max Project Value ($)", 500, 20000, 5000)
    monthly_customers = st.number_input("Monthly New Customers", 10, 1000, 100)
    churn_rate = st.slider("Monthly Churn Rate (%)", 0.0, 50.0, 5.0, 0.5)

# Revenue model controls
with st.sidebar.expander("💰 Revenue Model", expanded=True):
    fee_threshold = st.number_input("Fee Threshold ($)", 100, 500, 300)
    flat_fee = st.number_input("Flat Fee (< threshold) ($)", 5, 20, 7)
    commission_rate = st.slider("Commission Rate (≥ threshold) %", 3.0, 15.0, 6.0, 0.1)

# Infrastructure costs
with st.sidebar.expander("🏗️ Infrastructure Costs"):
    code_infra = st.number_input("Code Infra Cost ($)", 0.0, 2.0, INFRA_COSTS["code"], 0.01)
    docs_infra = st.number_input("Docs Infra Cost ($)", 0.0, 2.0, INFRA_COSTS["docs"], 0.01)
    image_infra = st.number_input("Image Infra Cost ($)", 0.0, 2.0, INFRA_COSTS["image"], 0.01)
    video_free_gb = st.number_input("Video Free GB", 0.0, 10.0, float(INFRA_COSTS["video"]["free_gb"]), 0.1)
    video_cost_per_gb = st.number_input("Video Cost per GB ($)", 0.0, 1.0, float(INFRA_COSTS["video"]["cost_per_gb"]), 0.01)

# Team costs toggle
with st.sidebar.expander("👥 Team & Operating Costs"):
    enable_salaries = st.toggle("Enable Team Salary Costs", value=False)
    
    if enable_salaries:
        dev_count = st.number_input("Developers", 1, 10, TEAM_SALARIES["developers"]["count"])
        dev_salary = st.number_input("Developer Monthly Salary ($)", 5000, 15000, TEAM_SALARIES["developers"]["monthly_salary"])
        marketer_count = st.number_input("Marketers", 0, 5, TEAM_SALARIES["marketer"]["count"])
        marketer_salary = st.number_input("Marketer Monthly Salary ($)", 4000, 12000, TEAM_SALARIES["marketer"]["monthly_salary"])
        support_count = st.number_input("Support Staff", 0, 5, TEAM_SALARIES["support"]["count"])
        support_salary = st.number_input("Support Monthly Salary ($)", 3000, 8000, TEAM_SALARIES["support"]["monthly_salary"])
        payroll_multiplier = st.slider("Payroll Multiplier (Benefits)", 1.0, 1.5, PAYROLL_MULTIPLIER, 0.05)
        
        monthly_team_cost = (
            (dev_count * dev_salary + marketer_count * marketer_salary + support_count * support_salary) 
            * payroll_multiplier
        )
    else:
        monthly_team_cost = 0

# CAC settings
with st.sidebar.expander("📈 Customer Acquisition"):
    cac_per_customer = st.number_input("CAC per Customer ($)", 0, 100, CAC_PER_CUSTOMER)

# Update costs based on user input
INFRA_COSTS["code"] = code_infra
INFRA_COSTS["docs"] = docs_infra
INFRA_COSTS["image"] = image_infra
INFRA_COSTS["video"]["free_gb"] = video_free_gb
INFRA_COSTS["video"]["cost_per_gb"] = video_cost_per_gb

# -------------------- Project Simulation --------------------
def simulate_projects(n_projects, min_val, max_val, fee_threshold, flat_fee, commission_rate):
    """Simulate project data with updated revenue model"""
    project_data = []
    
    for _ in range(n_projects):
        project_value = round(np.random.uniform(min_val, max_val), 2)
        
        # Updated revenue model
        if project_value < fee_threshold:
            platform_fee = flat_fee
        else:
            platform_fee = round(project_value * (commission_rate / 100), 2)
        
        stripe_fee = round(project_value * STRIPE_PERCENT + STRIPE_FIXED, 2)
        stripe_transfer_fee = round(project_value * STRIPE_TRANSFER_PERCENT + STRIPE_TRANSFER_FIXED, 2)  # Paid by customer
        
        project_type = np.random.choice(PROJECT_TYPES, p=[0.3, 0.2, 0.25, 0.25])
        
        if project_type == "video":
            storage_gb = round(np.random.uniform(1, 15), 2)
            if storage_gb <= INFRA_COSTS["video"]["free_gb"]:
                infra_cost = 0
            else:
                additional_gb = storage_gb - INFRA_COSTS["video"]["free_gb"]
                infra_cost = round(additional_gb * INFRA_COSTS["video"]["cost_per_gb"], 2)
        else:
            storage_gb = 0
            infra_cost = INFRA_COSTS[project_type]
        
        # Calculate CAC per project (assuming one customer per project for simplicity)
        cac_cost = cac_per_customer
        
        # Net profit calculation (transfer fees are NOT deducted from platform revenue)
        net_profit = platform_fee - stripe_fee - infra_cost - cac_cost
        freelancer_payout = project_value
        
        project_data.append({
            "Project Value": project_value,
            "Project Type": project_type,
            "Storage GB": storage_gb,
            "Platform Fee": platform_fee,
            "Stripe Fee": stripe_fee,
            "Stripe Transfer Fee": stripe_transfer_fee,
            "Infra Cost": infra_cost,
            "CAC Cost": cac_cost,
            "Net Profit": net_profit,
            "Freelancer Payout": freelancer_payout,
        })
    
    return pd.DataFrame(project_data)

# Generate project data
df = simulate_projects(num_projects, project_min, project_max, fee_threshold, flat_fee, commission_rate)
df["Project Tier"] = df["Project Value"].apply(lambda x: "Small" if x < fee_threshold else "Standard")
df["Margin (%)"] = (df["Net Profit"] / df["Platform Fee"] * 100).replace([np.inf, -np.inf], 0)

# Calculate LTV (simplified)
avg_project_value = df["Project Value"].mean()
avg_projects_per_customer = 2.5  # Assumption
customer_lifetime_months = 12  # Assumption
ltv = avg_project_value * avg_projects_per_customer * (commission_rate / 100) * customer_lifetime_months

# -------------------- Enhanced P&L Table --------------------
total_revenue = df["Platform Fee"].sum()
total_stripe_fees = df["Stripe Fee"].sum()
total_transfer_fees = df["Stripe Transfer Fee"].sum()  # Shown but not deducted from our costs
total_infra_costs = df["Infra Cost"].sum()
total_cac = df["CAC Cost"].sum()
gross_profit = total_revenue - total_stripe_fees - total_infra_costs  # Only Stripe fees, not transfer fees
monthly_projects = num_projects / 12  # Assuming annual data
annual_team_cost = monthly_team_cost * 12 if enable_salaries else 0
ebitda = gross_profit - total_cac - annual_team_cost
net_profit = ebitda  # Simplified (no taxes, interest, depreciation)

pnl = {
    "Revenue (Platform Fees)": total_revenue,
    "COGS - Stripe Payment Fees": total_stripe_fees,
    "COGS - Infrastructure Costs": total_infra_costs,
    "Gross Profit": gross_profit,
    "CAC (Customer Acquisition)": total_cac,
    "Team Salaries (Annual)": annual_team_cost,
    "EBITDA": ebitda,
    "Net Profit": net_profit,
    "--- Customer Paid Fees ---": "---",
    "Transfer Fees (Customer Paid)": total_transfer_fees,
}

# Calculate margins
pnl["Gross Margin (%)"] = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
pnl["EBITDA Margin (%)"] = (ebitda / total_revenue * 100) if total_revenue > 0 else 0
pnl["Net Margin (%)"] = (net_profit / total_revenue * 100) if total_revenue > 0 else 0

# -------------------- Dashboard Metrics --------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💰 Total Revenue", 
        value=f"${total_revenue:,.0f}", 
        delta=f"{pnl['Gross Margin (%)']:.1f}% GM",
        help="Total platform fees collected"
    )

with col2:
    st.metric(
        label="💸 Total COGS", 
        value=f"${total_stripe_fees + total_infra_costs:,.0f}",
        delta=f"Stripe: ${total_stripe_fees:,.0f}",
        help="Cost of Goods Sold (Stripe fees + Infrastructure)"
    )

with col3:
    st.metric(
        label="📈 EBITDA", 
        value=f"${ebitda:,.0f}", 
        delta=f"{pnl['EBITDA Margin (%)']:.1f}%",
        help="Earnings Before Interest, Taxes, Depreciation, Amortization"
    )

with col4:
    st.metric(
        label="🏆 Net Profit", 
        value=f"${net_profit:,.0f}", 
        delta=f"{pnl['Net Margin (%)']:.1f}%",
        help="Bottom line profit after all expenses"
    )

# Second row of metrics
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        label="📊 Avg Revenue/Project", 
        value=f"${df['Platform Fee'].mean():.2f}",
        help="Average platform fee per project"
    )

with col6:
    st.metric(
        label="🎯 CAC per Customer", 
        value=f"${cac_per_customer}",
        help="Customer Acquisition Cost"
    )

with col7:
    st.metric(
        label="💎 Customer LTV", 
        value=f"${ltv:.0f}",
        help="Customer Lifetime Value"
    )

with col8:
    st.metric(
        label="⚖️ LTV/CAC Ratio", 
        value=f"{ltv/cac_per_customer:.1f}x" if cac_per_customer > 0 else "∞",
        delta="Healthy" if (ltv/cac_per_customer > 3 if cac_per_customer > 0 else True) else "Needs Work",
        help="LTV to CAC ratio (>3x is healthy)"
    )

# Enhanced P&L Table
st.subheader("📋 Detailed P&L Statement")
pnl_df = pd.DataFrame(pnl, index=[0]).T
pnl_df.columns = ['Amount ($)']
pnl_df['Amount ($)'] = pnl_df['Amount ($)'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else str(x))
st.dataframe(pnl_df, use_container_width=True)

# Break-even analysis
if monthly_team_cost > 0:
    monthly_gross_profit = gross_profit / 12
    breakeven_months = annual_team_cost / monthly_gross_profit if monthly_gross_profit > 0 else float('inf')
    st.info(f"💡 **Break-even Analysis**: With current team costs of ${monthly_team_cost:,.0f}/month, break-even in {breakeven_months:.1f} months")

# -------------------- Modern Visualizations --------------------
modern_colors = {
    "primary": "#00D4FF",
    "secondary": "#FF6B6B", 
    "success": "#4ECDC4",
    "warning": "#FFE66D",
    "danger": "#FF8E8E",
    "purple": "#A8E6CF",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
    "orange": "#e85d04",
}

sns.set_theme(style="darkgrid", palette="bright")
plt.style.use("dark_background")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "monospace",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# 1. Project Value Distribution
st.subheader("📊 Project Value Distribution")
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
n, bins, patches = ax.hist(df["Project Value"], bins=40, alpha=0.8, color=modern_colors["primary"], edgecolor="white", linewidth=0.5)

xs = np.linspace(df["Project Value"].min(), df["Project Value"].max(), 200)
density = stats.gaussian_kde(df["Project Value"])
ax.plot(xs, density(xs) * len(df["Project Value"]) * (bins[1] - bins[0]), color=modern_colors["warning"], linewidth=3, alpha=0.9)

# Add threshold line
ax.axvline(x=fee_threshold, color=modern_colors["danger"], linestyle="--", alpha=0.8, linewidth=2, label=f"Fee Threshold (${fee_threshold})")

ax.set_title("Distribution of Project Values (Histogram with KDE)", fontsize=16, fontweight="bold", color="white", pad=20)
ax.set_xlabel("Project Value (USD)", color="white", fontweight="bold")
ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# 2. Net Profit per Project
st.subheader("💰 Net Profit per Project")
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
profit_values = df["Net Profit"]
n, bins, patches = ax.hist(profit_values, bins=40, alpha=0.8, color=modern_colors["success"], edgecolor="white", linewidth=0.5)

for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
    if bin_right <= 0:
        patch.set_facecolor(modern_colors["danger"])

xs = np.linspace(profit_values.min(), profit_values.max(), 200)
density = stats.gaussian_kde(profit_values)
ax.plot(xs, density(xs) * len(profit_values) * (bins[1] - bins[0]), color=modern_colors["warning"], linewidth=3, alpha=0.9)
ax.axvline(x=0, color="white", linestyle="--", alpha=0.7, linewidth=2)

ax.set_title("Distribution of Net Profit per Project (Including CAC)", fontsize=16, fontweight="bold", color="white", pad=20)
ax.set_xlabel("Net Profit (USD)", color="white", fontweight="bold")
ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)

# 3. Enhanced Cost Breakdown with CAC and Team Costs
st.subheader("💸 Cost Breakdown Analysis")

# Create interactive Plotly donut chart
cost_data = {
    'Category': ['Platform Revenue', 'Stripe Payment Fees', 'Infrastructure', 'Customer Acquisition', 'Team Salaries'],
    'Amount': [total_revenue, total_stripe_fees, total_infra_costs, total_cac, annual_team_cost],
    'Type': ['Revenue', 'Cost', 'Cost', 'Cost', 'Cost']
}

# Remove zero costs for cleaner visualization
cost_df = pd.DataFrame(cost_data)
cost_df_filtered = cost_df[cost_df['Amount'] > 0]

fig = px.pie(cost_df_filtered, values='Amount', names='Category', 
             title='Revenue and Cost Breakdown (Interactive)',
             hole=0.6,
             color_discrete_sequence=['#4ECDC4', '#FFE66D', '#A8E6CF', '#e85d04', '#FF6B6B'])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    title_font_size=16,
    title_x=0.5,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(color="white")
    )
)

# Add center text
fig.add_annotation(
    text=f"Net Profit<br>${net_profit:,.0f}",
    x=0.5, y=0.5,
    font_size=16,
    showarrow=False,
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

# 4. Sophisticated Scatter Plot
st.subheader("🎯 Project Value vs Net Profit")
fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
scatter = ax.scatter(df["Project Value"], df["Net Profit"], c=df["Net Profit"], s=60, alpha=0.7, cmap="RdYlBu_r", edgecolors="white", linewidth=0.5)

ax.axhline(y=0, color=modern_colors["danger"], linestyle="--", alpha=0.8, linewidth=2, label="Break-even")
ax.axvline(x=fee_threshold, color=modern_colors["warning"], linestyle="--", alpha=0.8, linewidth=2, label=f"Fee Threshold (${fee_threshold})")

z = np.polyfit(df["Project Value"], df["Net Profit"], 1)
p = np.poly1d(z)
ax.plot(df["Project Value"], p(df["Project Value"]), color=modern_colors["warning"], linewidth=3, alpha=0.8, label="Trend Line")

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Net Profit (USD)", color="white", fontweight="bold")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

ax.set_title("Project Value vs Net Profit Analysis (with Fee Threshold)", fontsize=16, fontweight="bold", color="white", pad=20)
ax.set_xlabel("Project Value (USD)", color="white", fontweight="bold")
ax.set_ylabel("Net Profit (USD)", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# 5. Modern Boxplot with Violin Plot
st.subheader("📈 Net Profit by Project Tier")
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
small_projects = df[df["Project Tier"] == "Small"]["Net Profit"].values
standard_projects = df[df["Project Tier"] == "Standard"]["Net Profit"].values

if len(small_projects) > 0 and len(standard_projects) > 0:
    parts = ax.violinplot([small_projects, standard_projects], positions=[1, 2], showmeans=True, showmedians=True)
    colors = [modern_colors["danger"], modern_colors["success"]]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    box_parts = ax.boxplot([small_projects, standard_projects], positions=[1, 2], patch_artist=True,
                          boxprops=dict(facecolor="none", edgecolor="white", linewidth=2),
                          medianprops=dict(color="white", linewidth=3),
                          whiskerprops=dict(color="white", linewidth=2),
                          capprops=dict(color="white", linewidth=2))
    
    ax.set_xticklabels(["Small Projects", "Standard Projects"], color="white", fontweight="bold")
    ax.set_title("Net Profit Distribution by Project Tier", fontsize=16, fontweight="bold", color="white", pad=20)
    ax.set_ylabel("Net Profit (USD)", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.axhline(y=0, color=modern_colors["warning"], linestyle="--", alpha=0.7, linewidth=2)
else:
    all_profits = df["Net Profit"].values
    ax.hist(all_profits, bins=30, alpha=0.8, color=modern_colors["success"], edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color=modern_colors["warning"], linestyle="--", alpha=0.7, linewidth=2)
    ax.set_title("Net Profit Distribution (All Projects)", fontsize=16, fontweight="bold", color="white", pad=20)
    ax.set_xlabel("Net Profit (USD)", color="white", fontweight="bold")
    ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
    ax.tick_params(colors="white")

plt.tight_layout()
st.pyplot(fig)

# 6. Gradient Margin Analysis
st.subheader("📊 Profit Margin Distribution")
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
n, bins, patches = ax.hist(df["Margin (%)"], bins=30, alpha=0.8, edgecolor="white", linewidth=0.5)

for i, patch in enumerate(patches):
    margin_val = (bins[i] + bins[i + 1]) / 2
    if margin_val < 0:
        patch.set_facecolor(modern_colors["danger"])
    elif margin_val < 20:
        patch.set_facecolor(modern_colors["warning"])
    else:
        patch.set_facecolor(modern_colors["success"])

margin_values = df["Margin (%)"]
xs = np.linspace(margin_values.min(), margin_values.max(), 200)
density = stats.gaussian_kde(margin_values)
ax.plot(xs, density(xs) * len(margin_values) * (bins[1] - bins[0]), color=modern_colors["primary"], linewidth=3, alpha=0.9)

mean_margin = margin_values.mean()
ax.axvline(x=mean_margin, color="white", linestyle="--", alpha=0.8, linewidth=2, label=f"Mean: {mean_margin:.1f}%")

ax.set_title("Distribution of Profit Margins", fontsize=16, fontweight="bold", color="white", pad=20)
ax.set_xlabel("Margin (%)", color="white", fontweight="bold")
ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# 7. Infrastructure Cost by Project Type
st.subheader("🏗️ Infrastructure Cost by Project Type")
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
infra_summary = df.groupby("Project Type")["Infra Cost"].sum().sort_values(ascending=False)
type_colors = [modern_colors["primary"], modern_colors["success"], modern_colors["warning"], modern_colors["danger"]]
bars = ax.bar(infra_summary.index, infra_summary.values, color=type_colors[:len(infra_summary)], alpha=0.8, edgecolor="white", linewidth=2)

for bar, value in zip(bars, infra_summary.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(infra_summary.values) * 0.01, 
            f"${value:.2f}", ha="center", va="bottom", color="white", fontweight="bold")

ax.set_title("Infrastructure Cost by Project Type", fontsize=16, fontweight="bold", color="white", pad=20)
ax.set_xlabel("Project Type", color="white", fontweight="bold")
ax.set_ylabel("Total Infrastructure Cost ($)", color="white", fontweight="bold")
ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)

# 8. Video Project Storage Distribution and Costs
video_projects = df[df["Project Type"] == "video"]
if len(video_projects) > 0:
    st.subheader("🎥 Video Project Analysis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0a0a0a")
    
    ax1.set_facecolor("#0a0a0a")
    n, bins, patches = ax1.hist(video_projects["Storage GB"], bins=20, alpha=0.8, color=modern_colors["primary"], edgecolor="white", linewidth=0.5)
    for i, patch in enumerate(patches):
        if bins[i] <= video_free_gb:
            patch.set_facecolor(modern_colors["success"])
        else:
            patch.set_facecolor(modern_colors["warning"])
    
    ax1.axvline(x=video_free_gb, color=modern_colors["danger"], linestyle="--", alpha=0.8, linewidth=3, label=f"Free Storage Limit ({video_free_gb}GB)")
    ax1.set_title("Video Project Storage Distribution", fontsize=14, fontweight="bold", color="white")
    ax1.set_xlabel("Storage Size (GB)", color="white", fontweight="bold")
    ax1.set_ylabel("Number of Projects", color="white", fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
    
    ax2.set_facecolor("#0a0a0a")
    video_costs = video_projects["Infra Cost"]
    n2, bins2, patches2 = ax2.hist(video_costs, bins=15, alpha=0.8, color=modern_colors["warning"], edgecolor="white", linewidth=0.5)
    for i, patch in enumerate(patches2):
        if bins2[i] == 0:
            patch.set_facecolor(modern_colors["success"])
    
    ax2.set_title("Video Infrastructure Costs", fontsize=14, fontweight="bold", color="white")
    ax2.set_xlabel("Infrastructure Cost ($)", color="white", fontweight="bold")
    ax2.set_ylabel("Number of Projects", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig)

# 9. Project Type Distribution Donut Chart
st.subheader("📊 Project Type Distribution")
fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
type_counts = df["Project Type"].value_counts()
type_colors_donut = [modern_colors["primary"], modern_colors["success"], modern_colors["warning"], modern_colors["danger"]][:len(type_counts)]
wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index, colors=type_colors_donut, autopct="%1.1f%%", 
                                  startangle=90, pctdistance=0.85, textprops={"color": "white", "fontweight": "bold"})

centre_circle = plt.Circle((0, 0), 0.60, fc="#0a0a0a")
ax.add_artist(centre_circle)
total_projects = len(df)
ax.text(0, 0, f"Total Projects\n{total_projects:,}", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
ax.set_title("Project Type Distribution", fontsize=18, fontweight="bold", color="white", pad=30)
plt.tight_layout()
st.pyplot(fig)

# 10. Profit Margin by Project Type Boxplot
st.subheader("📈 Net Profit by Project Type")
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
project_types = df["Project Type"].unique()
profit_data = [df[df["Project Type"] == ptype]["Net Profit"].values for ptype in project_types]
box_parts = ax.boxplot(profit_data, tick_labels=project_types, patch_artist=True,
                      boxprops=dict(facecolor="none", edgecolor="white", linewidth=2),
                      medianprops=dict(color=modern_colors["warning"], linewidth=3),
                      whiskerprops=dict(color="white", linewidth=2),
                      capprops=dict(color="white", linewidth=2))

for patch, color in zip(box_parts["boxes"], type_colors_donut):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0, color=modern_colors["danger"], linestyle="--", alpha=0.7, linewidth=2, label="Break-even")
ax.set_title("Net Profit Distribution by Project Type", fontsize=16, fontweight="bold", color="white", pad=20)
ax.set_xlabel("Project Type", color="white", fontweight="bold")
ax.set_ylabel("Net Profit ($)", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#0a0a0a", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# -------------------- NEW CHARTS --------------------

# 11. LTV vs CAC Analysis using Plotly
st.subheader("💎 LTV vs CAC Analysis")

# Create sample data for different customer segments
segments_data = {
    'Segment': ['New', 'Returning', 'Premium', 'Enterprise'],
    'LTV': [ltv * 0.5, ltv * 0.8, ltv * 1.2, ltv * 2.0],
    'CAC': [cac_per_customer * 0.8, cac_per_customer * 1.0, cac_per_customer * 1.5, cac_per_customer * 2.5]
}

ltv_cac_df = pd.DataFrame(segments_data)
ltv_cac_df['Ratio'] = ltv_cac_df['LTV'] / ltv_cac_df['CAC']

fig = go.Figure()

# Add LTV bars
fig.add_trace(go.Bar(
    name='LTV',
    x=ltv_cac_df['Segment'],
    y=ltv_cac_df['LTV'],
    marker_color='#4ECDC4',
    text=[f'${val:,.0f}' for val in ltv_cac_df['LTV']],
    textposition='auto',
))

# Add CAC bars
fig.add_trace(go.Bar(
    name='CAC',
    x=ltv_cac_df['Segment'],
    y=ltv_cac_df['CAC'],
    marker_color='#FF6B6B',
    text=[f'${val:,.0f}' for val in ltv_cac_df['CAC']],
    textposition='auto',
))

# Add ratio line
fig.add_trace(go.Scatter(
    name='LTV/CAC Ratio',
    x=ltv_cac_df['Segment'],
    y=ltv_cac_df['Ratio'] * 100,  # Scale for visibility
    mode='lines+markers+text',
    text=[f'{ratio:.1f}x' for ratio in ltv_cac_df['Ratio']],
    textposition='top center',
    yaxis='y2',
    line=dict(color='#FFE66D', width=3),
    marker=dict(size=10)
))

fig.update_layout(
    title='Customer Lifetime Value vs Customer Acquisition Cost by Segment',
    xaxis_title='Customer Segment',
    yaxis_title='Value ($)',
    yaxis2=dict(
        title='LTV/CAC Ratio',
        overlaying='y',
        side='right',
        range=[0, max(ltv_cac_df['Ratio']) * 120]
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    barmode='group',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)

# 12. Monthly Cash Flow Projection using Plotly
st.subheader("💰 Monthly Cash Flow Projection")

# Generate cash flow data with seasonality
months = range(1, 13)
base_monthly_revenue = total_revenue / 12
monthly_revenue = [base_monthly_revenue * (1 + 0.1 * (m-1) + 0.05 * np.sin(m * np.pi / 6)) for m in months]
monthly_costs = [(total_stripe_fees + total_infra_costs) / 12 + monthly_team_cost for _ in months]
monthly_cash_flow = [rev - cost for rev, cost in zip(monthly_revenue, monthly_costs)]
cumulative_cash_flow = np.cumsum(monthly_cash_flow)

# Create date range for x-axis
start_date = datetime.now()
dates = [start_date + timedelta(days=30*i) for i in range(12)]

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Monthly Cash Flow', 'Cumulative Cash Flow'),
    vertical_spacing=0.1,
    specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
)

# Monthly cash flow bars
colors = ['#4ECDC4' if cf > 0 else '#FF6B6B' for cf in monthly_cash_flow]
fig.add_trace(
    go.Bar(x=dates, y=monthly_cash_flow, marker_color=colors, name='Monthly Cash Flow', showlegend=False),
    row=1, col=1
)

# Cumulative cash flow line
fig.add_trace(
    go.Scatter(x=dates, y=cumulative_cash_flow, mode='lines+markers', 
              name='Cumulative Cash Flow', line=dict(color='#00D4FF', width=3),
              fill='tonexty', fillcolor='rgba(0, 212, 255, 0.2)'),
    row=2, col=1
)

# Add break-even line
fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=1)

fig.update_layout(
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    title_text="Cash Flow Analysis with Growth Projection",
    title_x=0.5
)

fig.update_xaxes(title_text="Month", row=2, col=1)
fig.update_yaxes(title_text="Cash Flow ($)", row=1, col=1)
fig.update_yaxes(title_text="Cumulative Cash Flow ($)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# 13. Cost Structure Analysis using Plotly
st.subheader("🏗️ Cost Structure Analysis")

cost_categories = ['Stripe Payment Fees', 'Infrastructure', 'Customer Acquisition', 'Team Salaries']
cost_values = [total_stripe_fees, total_infra_costs, total_cac, annual_team_cost]
cost_percentages = [cost/total_revenue*100 if total_revenue > 0 else 0 for cost in cost_values]

# Filter out zero costs
filtered_data = [(cat, val, pct) for cat, val, pct in zip(cost_categories, cost_values, cost_percentages) if val > 0]
if filtered_data:
    categories, values, percentages = zip(*filtered_data)
else:
    categories, values, percentages = cost_categories, cost_values, cost_percentages

fig = go.Figure()

fig.add_trace(go.Bar(
    y=categories,
    x=values,
    orientation='h',
    marker_color=['#FFE66D', '#A8E6CF', '#e85d04', '#FF6B6B'][:len(categories)],
    text=[f'${val:,.0f} ({pct:.1f}%)' for val, pct in zip(values, percentages)],
    textposition='auto',
))

fig.update_layout(
    title='Cost Structure Breakdown (% of Revenue)',
    xaxis_title='Cost ($)',
    yaxis_title='Cost Category',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# -------------------- Enhanced AI Growth & Revenue Forecasting --------------------
st.header("🤖 Enhanced AI-Powered Growth & Revenue Forecasting")
st.markdown("""
🚀 **Advanced Machine Learning Forecasting Engine**
- Multi-model ensemble prediction with confidence intervals
- Seasonal adjustment and trend analysis
- Churn-adjusted growth modeling
- Break-even scenario analysis
""")

# Enhanced Forecasting controls
col1, col2, col3 = st.columns(3)
with col1:
    forecast_months = st.slider("📅 Forecast Period (Months)", 6, 36, 12)
    growth_rate = st.slider("📈 Monthly Growth Rate (%)", 0.0, 50.0, 15.0, 1.0)

with col2:
    pred_projects = st.slider("🎯 Target Projects", 1000, 100000, num_projects * 2, 1000)
    seasonality_factor = st.slider("🌊 Seasonality Factor", 0.0, 30.0, 10.0, 1.0)

with col3:
    churn_adjustment = st.slider("📉 Churn Impact (%)", 0.0, 30.0, 10.0, 1.0)
    confidence_level = st.selectbox("📊 Confidence Level", [90, 95, 99], index=1)

# Enhanced growth simulation with multiple scenarios
def simulate_enhanced_growth(months, initial_projects, growth_rate, churn_rate, seasonality):
    """Enhanced growth simulation with seasonality and multiple scenarios"""
    scenarios = {
        'Optimistic': growth_rate * 1.5,
        'Expected': growth_rate,
        'Conservative': growth_rate * 0.7
    }
    
    results = {}
    
    for scenario_name, scenario_growth in scenarios.items():
        timeline = []
        projects_over_time = []
        revenue_over_time = []
        customers_over_time = []
        
        current_projects = initial_projects
        current_customers = initial_projects
        
        for month in range(1, months + 1):
            # Seasonal adjustment
            seasonal_multiplier = 1 + (seasonality / 100) * np.sin(2 * np.pi * month / 12)
            
            # Apply growth with seasonality
            monthly_growth = (scenario_growth / 100) * seasonal_multiplier
            current_projects *= (1 + monthly_growth)
            current_customers *= (1 + monthly_growth)
            
            # Apply churn
            current_customers *= (1 - churn_rate / 100)
            
            # Calculate revenue for this month's projects
            monthly_projects = int(current_projects / 12)
            temp_df = simulate_projects(monthly_projects, project_min, project_max, fee_threshold, flat_fee, commission_rate)
            monthly_revenue = temp_df["Platform Fee"].sum()
            
            timeline.append(month)
            projects_over_time.append(current_projects)
            revenue_over_time.append(monthly_revenue)
            customers_over_time.append(current_customers)
        
        results[scenario_name] = pd.DataFrame({
            'Month': timeline,
            'Projects': projects_over_time,
            'Revenue': revenue_over_time,
            'Customers': customers_over_time
        })
    
    return results

# Generate enhanced forecast data
forecast_scenarios = simulate_enhanced_growth(forecast_months, num_projects, growth_rate, churn_rate / 100, seasonality_factor)

# Advanced ML predictions with confidence intervals
expected_forecast = forecast_scenarios['Expected']
X_train = np.array(expected_forecast['Month']).reshape(-1, 1)
y_train = expected_forecast['Revenue'].values

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)

# Generate predictions with confidence intervals
future_months = np.array(range(forecast_months + 1, forecast_months + 7)).reshape(-1, 1)
future_revenue = poly_model.predict(future_months)

# Calculate confidence intervals
residuals = y_train - poly_model.predict(X_train)
mse = np.mean(residuals ** 2)
std_error = np.sqrt(mse)

confidence_multiplier = {90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
margin_of_error = confidence_multiplier * std_error

# Business metrics dashboard
total_expected_revenue = expected_forecast['Revenue'].sum()
total_optimistic_revenue = forecast_scenarios['Optimistic']['Revenue'].sum()
total_conservative_revenue = forecast_scenarios['Conservative']['Revenue'].sum()

avg_monthly_revenue = expected_forecast['Revenue'].mean()
revenue_growth_rate = (expected_forecast['Revenue'].iloc[-1] - expected_forecast['Revenue'].iloc[0]) / expected_forecast['Revenue'].iloc[0] * 100

# Enhanced metrics display
st.subheader("🎯 Advanced Forecast Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📈 Expected Revenue", 
        f"${total_expected_revenue:,.0f}",
        delta=f"+{revenue_growth_rate:.1f}%",
        help="Base case revenue projection"
    )

with col2:
    st.metric(
        "🚀 Optimistic Case", 
        f"${total_optimistic_revenue:,.0f}",
        delta=f"+{(total_optimistic_revenue - total_expected_revenue)/total_expected_revenue*100:.1f}%",
        help="Best case scenario (50% higher growth)"
    )

with col3:
    st.metric(
        "🛡️ Conservative Case", 
        f"${total_conservative_revenue:,.0f}",
        delta=f"{(total_conservative_revenue - total_expected_revenue)/total_expected_revenue*100:.1f}%",
        help="Conservative scenario (30% lower growth)"
    )

with col4:
    final_customers = expected_forecast['Customers'].iloc[-1]
    st.metric(
        "👥 Customer Base", 
        f"{final_customers:,.0f}",
        delta=f"CAC: ${total_cac/len(df):.0f}",
        help="Projected customer count after churn"
    )

# Advanced scenario visualization using Plotly
st.subheader("📊 Multi-Scenario Revenue Forecast")

fig = go.Figure()

# Add each scenario
colors = {'Optimistic': '#4ECDC4', 'Expected': '#00D4FF', 'Conservative': '#FFE66D'}
for scenario_name, scenario_data in forecast_scenarios.items():
    fig.add_trace(go.Scatter(
        x=scenario_data['Month'], 
        y=scenario_data['Revenue'],
        mode='lines+markers',
        name=f'{scenario_name} Case',
        line=dict(color=colors[scenario_name], width=3),
        marker=dict(size=6)
    ))

# Add confidence bands
upper_bound = expected_forecast['Revenue'] + margin_of_error
lower_bound = expected_forecast['Revenue'] - margin_of_error

fig.add_trace(go.Scatter(
    x=expected_forecast['Month'].tolist() + expected_forecast['Month'].tolist()[::-1],
    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
    fill='toself',
    fillcolor='rgba(0, 212, 255, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name=f'{confidence_level}% Confidence Interval',
    showlegend=True
))

# Add extended forecast
extended_months = list(range(forecast_months + 1, forecast_months + 7))
fig.add_trace(go.Scatter(
    x=extended_months,
    y=future_revenue,
    mode='lines+markers',
    name='Extended Forecast',
    line=dict(color='#FF6B6B', width=2, dash='dash'),
    marker=dict(size=4)
))

fig.update_layout(
    title=f'Revenue Growth Forecast - {forecast_months} Month Projection with {confidence_level}% Confidence Interval',
    xaxis_title='Month',
    yaxis_title='Monthly Revenue ($)',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)

# Customer acquisition forecast
st.subheader("👥 Customer Growth & Churn Analysis")

customer_fig = go.Figure()

for scenario_name, scenario_data in forecast_scenarios.items():
    customer_fig.add_trace(go.Scatter(
        x=scenario_data['Month'], 
        y=scenario_data['Customers'],
        mode='lines+markers',
        name=f'{scenario_name} Customers',
        line=dict(color=colors[scenario_name], width=3),
        marker=dict(size=6)
    ))

customer_fig.update_layout(
    title='Customer Base Growth (Churn-Adjusted)',
    xaxis_title='Month',
    yaxis_title='Number of Customers',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    hovermode='x unified'
)

st.plotly_chart(customer_fig, use_container_width=True)

# Enhanced break-even analysis with forecasting
if enable_salaries and monthly_team_cost > 0:
    st.subheader("⚖️ Enhanced Break-Even Analysis")
    
    # Calculate break-even point for each scenario
    breakeven_revenue_needed = monthly_team_cost + (total_stripe_fees + total_infra_costs + total_cac) / 12
    
    breakeven_results = {}
    for scenario_name, scenario_data in forecast_scenarios.items():
        breakeven_month = None
        for i, monthly_rev in enumerate(scenario_data['Revenue']):
            if monthly_rev >= breakeven_revenue_needed:
                breakeven_month = i + 1
                break
        breakeven_results[scenario_name] = breakeven_month
    
    # Break-even visualization
    breakeven_fig = go.Figure()
    
    # Add revenue lines for each scenario
    for scenario_name, scenario_data in forecast_scenarios.items():
        breakeven_fig.add_trace(go.Scatter(
            x=scenario_data['Month'], 
            y=scenario_data['Revenue'],
            mode='lines',
            name=f'{scenario_name} Revenue',
            line=dict(color=colors[scenario_name], width=3)
        ))
    
    # Add break-even line
    breakeven_fig.add_hline(
        y=breakeven_revenue_needed, 
        line_dash="dash", 
        line_color="#FF6B6B", 
        annotation_text=f"Break-even: ${breakeven_revenue_needed:,.0f}",
        annotation_position="top right"
    )
    
    # Add break-even points
    for scenario_name, breakeven_month in breakeven_results.items():
        if breakeven_month:
            breakeven_fig.add_trace(go.Scatter(
                x=[breakeven_month], 
                y=[breakeven_revenue_needed],
                mode='markers',
                name=f'{scenario_name} Break-even',
                marker=dict(
                    color=colors[scenario_name], 
                    size=12, 
                    symbol='star',
                    line=dict(color='white', width=2)
                )
            ))
    
    breakeven_fig.update_layout(
        title="Break-Even Analysis Across Scenarios",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hovermode='x unified'
    )
    
    st.plotly_chart(breakeven_fig, use_container_width=True)
    
    # Break-even summary
    col1, col2, col3 = st.columns(3)
    for i, (scenario_name, breakeven_month) in enumerate(breakeven_results.items()):
        with [col1, col2, col3][i]:
            if breakeven_month:
                st.success(f"🎉 **{scenario_name}**: Break-even Month {breakeven_month}")
            else:
                st.warning(f"⚠️ **{scenario_name}**: No break-even in forecast period")

# -------------------- Money Flow Example --------------------
st.header("💸 Money Flow Example: Project Transaction Breakdown")
st.markdown("""
### 🔍 **Detailed Transaction Flow Analysis**
Let's trace exactly where every dollar goes in a typical bl0q transaction.
""")

# Create example project
example_project_value = st.slider("📊 Example Project Value ($)", 100, 5000, 1000, 50)

# Calculate all fees and costs
if example_project_value < fee_threshold:
    example_platform_fee = flat_fee
    fee_type = f"Flat fee (< ${fee_threshold})"
else:
    example_platform_fee = round(example_project_value * (commission_rate / 100), 2)
    fee_type = f"{commission_rate}% commission (≥ ${fee_threshold})"

example_stripe_fee = round(example_project_value * STRIPE_PERCENT + STRIPE_FIXED, 2)
example_transfer_fee = round(example_project_value * STRIPE_TRANSFER_PERCENT + STRIPE_TRANSFER_FIXED, 2)
example_infra_cost = np.random.choice([INFRA_COSTS["code"], INFRA_COSTS["docs"], INFRA_COSTS["image"], 0.05])  # Random infra cost
example_cac = cac_per_customer
example_net_profit = example_platform_fee - example_stripe_fee - example_infra_cost - example_cac

# Money flow visualization
st.subheader(f"💰 Transaction Flow for ${example_project_value} Project")

# Create Sankey diagram data
sankey_fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = [
            "Client Payment",
            "Platform Revenue", 
            "Freelancer Payout",
            "Stripe Payment Fee",
            "Transfer Fee (Client)",
            "Infrastructure Cost",
            "CAC Cost",
            "Net Profit"
        ],
        color = ["#FF6B6B", "#00D4FF", "#4ECDC4", "#FFE66D", "#A8E6CF", "#e85d04", "#FF8E8E", "#4ECDC4"]
    ),
    link = dict(
        source = [0, 0, 1, 1, 1, 1, 1],  # Client Payment flows to Platform Revenue and Transfer Fee
        target = [1, 2, 3, 4, 5, 6, 7],  # Platform Revenue flows to various costs and profit
        value = [
            example_platform_fee,  # Client -> Platform Revenue
            example_project_value,  # Client -> Freelancer
            example_stripe_fee,     # Platform -> Stripe Fee
            example_transfer_fee,   # Platform -> Transfer Fee (shown but not deducted)
            example_infra_cost,     # Platform -> Infrastructure
            example_cac,           # Platform -> CAC
            max(0, example_net_profit)  # Platform -> Net Profit
        ],
        color = ["rgba(0, 212, 255, 0.3)", "rgba(76, 205, 196, 0.3)", 
                "rgba(255, 230, 109, 0.3)", "rgba(168, 230, 207, 0.3)",
                "rgba(232, 93, 4, 0.3)", "rgba(255, 142, 142, 0.3)", "rgba(76, 205, 196, 0.3)"]
    )
)])

sankey_fig.update_layout(
    title_text=f"Money Flow for ${example_project_value} Project",
    font_size=12,
    font_color="white",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=400
)

st.plotly_chart(sankey_fig, use_container_width=True)

# Detailed breakdown table
st.subheader("📊 Detailed Financial Breakdown")

breakdown_data = {
    "Item": [
        "💰 Client Pays (Total)",
        "🏢 Platform Fee",
        "👨‍💻 Freelancer Receives",
        "💳 Stripe Payment Fee (Our Cost)", 
        "🔄 Transfer Fee (Client Pays)*",
        "🏗️ Infrastructure Cost",
        "📈 Customer Acquisition Cost",
        "✅ Platform Net Profit"
    ],
    "Amount": [
        f"${example_project_value:.2f}",
        f"${example_platform_fee:.2f}",
        f"${example_project_value:.2f}",
        f"${example_stripe_fee:.2f}",
        f"${example_transfer_fee:.2f}",
        f"${example_infra_cost:.2f}",
        f"${example_cac:.2f}",
        f"${example_net_profit:.2f}"
    ],
    "Percentage": [
        "100.0%",
        f"{example_platform_fee/example_project_value*100:.1f}%",
        "100.0%",
        f"{example_stripe_fee/example_project_value*100:.1f}%",
        f"{example_transfer_fee/example_project_value*100:.1f}%",
        f"{example_infra_cost/example_project_value*100:.1f}%",
        f"{example_cac/example_project_value*100:.1f}%",
        f"{example_net_profit/example_project_value*100:.1f}%"
    ],
    "Notes": [
        "Total project value",
        fee_type,
        "Full project amount to freelancer",
        "2.9% + $0.30 (bl0q pays this)",
        "3% + $0.50 (client pays this)",
        "Server/storage costs",
        "Marketing/acquisition cost",
        "Final profit to bl0q"
    ]
}

breakdown_df = pd.DataFrame(breakdown_data)
st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

# Key insights
col1, col2 = st.columns(2)
with col1:
    st.info(f"""
    **💡 Key Insights:**
    - Platform margin: {example_net_profit/example_platform_fee*100:.1f}%
    - ROI on CAC: {example_platform_fee/example_cac:.1f}x
    - Fee structure: {fee_type}
    """)

with col2:
    st.warning(
        """
        **⚠️ Important Notes:**
        - Transfer fees are paid by CLIENT, not deducted from our revenue
        - Freelancer always gets full project amount
        - Our actual costs: Stripe fee + Infrastructure + CAC
        """
    )

# Profit margin analysis by project value
st.subheader("📈 Profit Margin Analysis by Project Value")

# Generate data for different project values
project_values = np.arange(50, 3000, 50)
margins = []
net_profits = []

for pv in project_values:
    if pv < fee_threshold:
        pf = flat_fee
    else:
        pf = pv * (commission_rate / 100)
    
    sf = pv * STRIPE_PERCENT + STRIPE_FIXED
    ic = example_infra_cost  # Use same infra cost for consistency
    cac = example_cac
    np_val = pf - sf - ic - cac
    
    margin = (np_val / pf * 100) if pf > 0 else 0
    margins.append(margin)
    net_profits.append(np_val)

margin_fig = go.Figure()

margin_fig.add_trace(go.Scatter(
    x=project_values,
    y=margins,
    mode='lines',
    name='Profit Margin %',
    line=dict(color='#00D4FF', width=3)
))

# Add threshold line
margin_fig.add_vline(
    x=fee_threshold, 
    line_dash="dash", 
    line_color="#FF6B6B",
    annotation_text=f"Fee Threshold: ${fee_threshold}",
    annotation_position="top"
)

margin_fig.update_layout(
    title='Profit Margin % vs Project Value',
    xaxis_title='Project Value ($)',
    yaxis_title='Profit Margin (%)',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(margin_fig, use_container_width=True)

total_cogs = total_stripe_fees + total_infra_costs
total_forecast_revenue = total_expected_revenue
revenue_growth = revenue_growth_rate
forecast_df = expected_forecast

# -------------------- Export & Download --------------------
st.header("📥 Export & Download")

# Prepare summary data
summary_data = {
    "Total Revenue": total_revenue,
    "Total COGS": total_cogs,
    "Gross Profit": gross_profit,
    "CAC": total_cac,
    "Team Costs": annual_team_cost,
    "EBITDA": ebitda,
    "Net Profit": net_profit,
    "Gross Margin %": pnl["Gross Margin (%)"],
    "EBITDA Margin %": pnl["EBITDA Margin (%)"],
    "Net Margin %": pnl["Net Margin (%)"],
    "LTV": ltv,
    "CAC per Customer": cac_per_customer,
    "LTV/CAC Ratio": ltv/cac_per_customer if cac_per_customer > 0 else 0,
    "Average Revenue per Project": df["Platform Fee"].mean(),
    "Average Profit per Project": df["Net Profit"].mean(),
    "Total Projects": len(df),
    "Forecast Revenue": total_forecast_revenue,
    "Revenue Growth %": revenue_growth
}

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Download Project Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"bloq_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Download Summary Report"):
        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Summary",
            data=csv,
            file_name=f"bloq_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("🚀 Download Forecast Data"):
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Forecast",
            data=csv,
            file_name=f"bloq_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# -------------------- AI-Generated Executive Summary --------------------
st.header("🤖 AI-Generated Executive Summary")

# Generate executive summary
def generate_executive_summary():
    """Generate an AI-style executive summary"""
    summary = f"""
    ## 📊 Executive Summary - bl0q Platform Analysis
    
    **Financial Performance:**
    - Total Revenue: ${total_revenue:,.0f} with {pnl['Gross Margin (%)']:.1f}% gross margin
    - Net Profit: ${net_profit:,.0f} ({pnl['Net Margin (%)']:.1f}% margin)
    - EBITDA: ${ebitda:,.0f} ({pnl['EBITDA Margin (%)']:.1f}% margin)
    
    **Customer Economics:**
    - Customer LTV: ${ltv:.0f}
    - Customer Acquisition Cost: ${cac_per_customer}
    - LTV/CAC Ratio: {ltv/cac_per_customer:.1f}x {'✅ Healthy' if ltv/cac_per_customer > 3 else '⚠️ Needs Improvement'}
    
    **Growth Projection:**
    - {forecast_months}-month revenue forecast: ${total_forecast_revenue:,.0f}
    - Projected growth rate: {revenue_growth:.1f}%
    - Expected customer base: {final_customers:,.0f}
    
    **Key Insights:**
    - Average revenue per project: ${df['Platform Fee'].mean():.2f}
    - {'- High-value projects (≥$' + str(fee_threshold) + ') represent ' + str(len(df[df['Project Tier'] == 'Standard'])) + ' projects'}
    - Infrastructure costs: ${df['Infra Cost'].sum():,.0f} ({df['Infra Cost'].sum()/total_revenue*100:.1f}% of revenue)
    
    **Recommendations:**
    {'- Strong unit economics with healthy LTV/CAC ratio' if ltv/cac_per_customer > 3 else '- Focus on improving customer acquisition efficiency'}
    {'- Profitable operations' if net_profit > 0 else '- Focus on cost optimization to achieve profitability'}
    - Consider pricing optimization for projects near the ${fee_threshold} threshold
    - Monitor infrastructure costs as video projects scale
    """
    return summary

st.markdown(generate_executive_summary())

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 2rem;'>
    <h4>🚀 bl0q Platform Financial Simulator</h4>
    <p>Built with Streamlit • Enhanced with AI/ML Forecasting • Modern Dark UI</p>
    <p>Features: Advanced P&L Analysis • LTV/CAC Optimization • Break-even Modeling • Growth Projections</p>
    <p><strong>Version 2.0</strong> | Updated with team cost management and comprehensive forecasting</p>
</div>
""", unsafe_allow_html=True)