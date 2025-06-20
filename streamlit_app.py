import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# -------------------- Constants --------------------
NUM_PROJECTS = 5000
STRIPE_PERCENT = 0.029
STRIPE_FIXED = 0.30
STRIPE_TRANSFER_PERCENT = 0.03
STRIPE_TRANSFER_FIXED = 0.50
PROJECT_TYPES = ["code", "docs", "image", "video"]
INFRA_COSTS = {
    "code": 0.25,
    "docs": 0.02,
    "image": 0.03,
    "video": {"free_gb": 2, "cost_per_gb": 0.20},
}

st.set_page_config(page_title="bl0q Platform Simulator", layout="wide", page_icon="💡")
st.title("bl0q Platform Growth & Profit Simulator")
st.markdown("""
This interactive dashboard simulates project growth, platform fees, costs, and profit margins for the bl0q platform. Adjust parameters and see the impact instantly!
""")

# Sidebar for user input
st.sidebar.header("Simulation Controls")
num_projects = st.sidebar.slider("Number of Projects", 100, 10000, NUM_PROJECTS, 100)
project_min = st.sidebar.number_input("Min Project Value ($)", 10, 1000, 75)
project_max = st.sidebar.number_input("Max Project Value ($)", 500, 10000, 5000)

# Infra costs
st.sidebar.subheader("Infrastructure Costs")
code_infra = st.sidebar.number_input(
    "Code Infra Cost ($)", 0.0, 10.0, INFRA_COSTS["code"], 0.01
)
docs_infra = st.sidebar.number_input(
    "Docs Infra Cost ($)", 0.0, 10.0, INFRA_COSTS["docs"], 0.01
)
image_infra = st.sidebar.number_input(
    "Image Infra Cost ($)", 0.0, 10.0, INFRA_COSTS["image"], 0.01
)
video_free_gb = st.sidebar.number_input(
    "Video Free GB", 0.0, 10.0, float(INFRA_COSTS["video"]["free_gb"]), 0.1
)
video_cost_per_gb = st.sidebar.number_input(
    "Video Cost per GB ($)", 0.0, 10.0, float(INFRA_COSTS["video"]["cost_per_gb"]), 0.01
)

# Update infra costs based on user input
INFRA_COSTS["code"] = code_infra
INFRA_COSTS["docs"] = docs_infra
INFRA_COSTS["image"] = image_infra
INFRA_COSTS["video"]["free_gb"] = video_free_gb
INFRA_COSTS["video"]["cost_per_gb"] = video_cost_per_gb

# -------------------- Simulation --------------------
project_data = []
for _ in range(num_projects):
    project_value = round(np.random.uniform(project_min, project_max), 2)
    platform_fee = (
        7 if project_value < 250 else round(project_value * 0.1, 2)
    )  # Ensure 10% everywhere
    stripe_fee = round(project_value * STRIPE_PERCENT + STRIPE_FIXED, 2)
    stripe_transfer_fee = round(
        project_value * STRIPE_TRANSFER_PERCENT + STRIPE_TRANSFER_FIXED, 2
    )
    project_type = np.random.choice(
        PROJECT_TYPES, p=[1 / len(PROJECT_TYPES)] * len(PROJECT_TYPES)
    )
    if project_type == "video":
        storage_gb = round(np.random.uniform(1, 10), 2)
        if storage_gb <= INFRA_COSTS["video"]["free_gb"]:
            infra_cost = 0
        else:
            additional_gb = storage_gb - INFRA_COSTS["video"]["free_gb"]
            infra_cost = round(additional_gb * INFRA_COSTS["video"]["cost_per_gb"], 2)
    else:
        storage_gb = 0
        infra_cost = INFRA_COSTS[project_type]
    net_profit = platform_fee - stripe_fee - stripe_transfer_fee - infra_cost
    freelancer_payout = project_value
    project_data.append(
        {
            "Project Value": project_value,
            "Project Type": project_type,
            "Storage GB": storage_gb,
            "Platform Fee": platform_fee,
            "Stripe Fee": stripe_fee,
            "Stripe Transfer Fee": stripe_transfer_fee,
            "Infra Cost": infra_cost,
            "Net Profit": net_profit,
            "Freelancer Payout": freelancer_payout,
        }
    )
df = pd.DataFrame(project_data)
df["Project Tier"] = df["Project Value"].apply(
    lambda x: "Small" if x < 250 else "Standard"
)
df["Margin (%)"] = df["Net Profit"] / df["Platform Fee"] * 100

# Warn if infra cost is zero
if df["Infra Cost"].sum() == 0:
    st.warning(
        "All projects are within the free infrastructure limit. Infra Costs are zero."
    )

# -------------------- P&L Table --------------------
pnl = {
    "Revenue (Platform Fee)": df["Platform Fee"].sum(),
    "COGS (Stripe Fees)": df["Stripe Fee"].sum()
    + df["Stripe Transfer Fee"].sum()
    + df["Infra Cost"].sum(),
    "Gross Profit": df["Net Profit"].sum(),
    "EBITDA": df["Net Profit"].sum(),
    "Net Profit": df["Net Profit"].sum(),
}
pnl["Gross Margin (%)"] = pnl["Gross Profit"] / pnl["Revenue (Platform Fee)"] * 100
pnl["EBITDA Margin (%)"] = pnl["EBITDA"] / pnl["Revenue (Platform Fee)"] * 100
pnl["Net Margin (%)"] = pnl["Net Profit"] / pnl["Revenue (Platform Fee)"] * 100

st.subheader("P&L Summary")
st.dataframe(pd.DataFrame(pnl, index=[0]).T, use_container_width=True)

# -------------------- Visualizations --------------------
# Modern color palette
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
plt.rcParams.update(
    {
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
    }
)

# 1. Project Value Distribution - Modern Gradient Style
st.subheader("Project Value Distribution")
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
n, bins, patches = ax.hist(
    df["Project Value"],
    bins=40,
    alpha=0.8,
    color=modern_colors["primary"],
    edgecolor="white",
    linewidth=0.5,
)

xs = np.linspace(df["Project Value"].min(), df["Project Value"].max(), 200)
density = stats.gaussian_kde(df["Project Value"])
ax.plot(
    xs,
    density(xs) * len(df["Project Value"]) * (bins[1] - bins[0]),
    color=modern_colors["warning"],
    linewidth=3,
    alpha=0.9,
)
ax.set_title(
    "Distribution of Project Values (Histogram with KDE)",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20,
)
ax.set_xlabel("Project Value (USD)", color="white", fontweight="bold")
ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)

# 2. Net Profit per Project - Elegant Design
st.subheader("Net Profit per Project")
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
profit_values = df["Net Profit"]
n, bins, patches = ax.hist(
    profit_values,
    bins=40,
    alpha=0.8,
    color=modern_colors["success"],
    edgecolor="white",
    linewidth=0.5,
)
for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
    if bin_right <= 0:
        patch.set_facecolor(modern_colors["danger"])
xs = np.linspace(profit_values.min(), profit_values.max(), 200)
density = stats.gaussian_kde(profit_values)
ax.plot(
    xs,
    density(xs) * len(profit_values) * (bins[1] - bins[0]),
    color=modern_colors["warning"],
    linewidth=3,
    alpha=0.9,
)
ax.axvline(x=0, color="white", linestyle="--", alpha=0.7, linewidth=2)
ax.set_title(
    "Distribution of Net Profit per Project (Histogram with KDE and Break-even Line)",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20,
)
ax.set_xlabel("Net Profit (USD)", color="white", fontweight="bold")
ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)

# 3. Modern Donut Chart for Cost Breakdown
st.subheader("Cost Breakdown")
costs = [
    df["Platform Fee"].sum(),
    df["Stripe Fee"].sum(),
    df["Stripe Transfer Fee"].sum(),
    df["Infra Cost"].sum(),
    df["Net Profit"].sum(),
]
labels = [
    "Platform Revenue",
    "Stripe Payment Fee",
    "Stripe Transfer Fee",
    "Infra Costs",
    "Net Profit",
]
# Ensure all wedge sizes are non-negative for the pie chart
costs_nonneg = [max(0, v) for v in costs]
if costs[-1] < 0:
    st.warning("Net Profit is negative. The donut chart will show Net Profit as zero.")
fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
wedges, texts, autotexts = ax.pie(
    costs_nonneg,
    labels=labels,
    colors=[
        modern_colors["success"],
        modern_colors["warning"],
        modern_colors["danger"],
        modern_colors["purple"],
        modern_colors["primary"],
    ],
    autopct=lambda pct: f"({pct:.2f}%)",  # Value and percent stacked
    startangle=90,
    pctdistance=0.85,
    textprops={"color": "white", "fontweight": "bold"},
)
centre_circle = plt.Circle((0, 0), 0.60, fc="#0a0a0a")
ax.add_artist(centre_circle)
ax.set_title(
    "Revenue and Cost Breakdown (Donut Chart)",
    fontsize=18,
    fontweight="bold",
    color="white",
    pad=30,
)
plt.tight_layout()
st.pyplot(fig)

# 4. Sophisticated Scatter Plot
st.subheader("Project Value vs Net Profit")
fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
scatter = ax.scatter(
    df["Project Value"],
    df["Net Profit"],
    c=df["Net Profit"],
    s=60,
    alpha=0.7,
    cmap="RdYlBu_r",
    edgecolors="white",
    linewidth=0.5,
)
ax.axhline(
    y=0,
    color=modern_colors["danger"],
    linestyle="--",
    alpha=0.8,
    linewidth=2,
    label="Break-even",
)
z = np.polyfit(df["Project Value"], df["Net Profit"], 1)
p = np.poly1d(z)
ax.plot(
    df["Project Value"],
    p(df["Project Value"]),
    color=modern_colors["warning"],
    linewidth=3,
    alpha=0.8,
    label="Trend Line",
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Net Profit (USD)", color="white", fontweight="bold")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
ax.set_title(
    "Project Value vs Net Profit Analysis (Scatter Plot with Color Mapping)",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20,
)
ax.set_xlabel("Project Value (USD)", color="white", fontweight="bold")
ax.set_ylabel("Net Profit (USD)", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# 5. Modern Boxplot with Violin Plot
st.subheader("Net Profit by Project Tier")
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
small_projects = df[df["Project Tier"] == "Small"]["Net Profit"].values
standard_projects = df[df["Project Tier"] == "Standard"]["Net Profit"].values
if len(small_projects) > 0 and len(standard_projects) > 0:
    parts = ax.violinplot(
        [small_projects, standard_projects],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
    )
    colors = [modern_colors["danger"], modern_colors["success"]]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    box_parts = ax.boxplot(
        [small_projects, standard_projects],
        positions=[1, 2],
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor="white", linewidth=2),
        medianprops=dict(color="white", linewidth=3),
        whiskerprops=dict(color="white", linewidth=2),
        capprops=dict(color="white", linewidth=2),
    )
    ax.set_xticklabels(
        ["Small Projects", "Standard Projects"], color="white", fontweight="bold"
    )
    ax.set_title(
        "Net Profit Distribution by Project Tier (Box Plot with Outliers)",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.set_ylabel("Net Profit (USD)", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.axhline(
        y=0, color=modern_colors["warning"], linestyle="--", alpha=0.7, linewidth=2
    )
else:
    all_profits = df["Net Profit"].values
    ax.hist(
        all_profits,
        bins=30,
        alpha=0.8,
        color=modern_colors["success"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(
        y=0, color=modern_colors["warning"], linestyle="--", alpha=0.7, linewidth=2
    )
    ax.set_title(
        "Net Profit Distribution (All Projects) (Histogram with Break-even Line)",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.set_xlabel("Net Profit (USD)", color="white", fontweight="bold")
    ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
    ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)

# 6. Gradient Margin Analysis
st.subheader("Profit Margin Distribution")
fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
n, bins, patches = ax.hist(
    df["Margin (%)"], bins=30, alpha=0.8, edgecolor="white", linewidth=0.5
)
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
ax.plot(
    xs,
    density(xs) * len(margin_values) * (bins[1] - bins[0]),
    color=modern_colors["primary"],
    linewidth=3,
    alpha=0.9,
)
mean_margin = margin_values.mean()
ax.axvline(
    x=mean_margin,
    color="white",
    linestyle="--",
    alpha=0.8,
    linewidth=2,
    label=f"Mean: {mean_margin:.1f}%",
)
ax.set_title(
    "Distribution of Profit Margins (Histogram with KDE and Mean Line)",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20,
)
ax.set_xlabel("Margin (%)", color="white", fontweight="bold")
ax.set_ylabel("Number of Projects", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# 7. Infrastructure Cost by Project Type
st.subheader("Infrastructure Cost by Project Type")
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
infra_summary = (
    df.groupby("Project Type")["Infra Cost"].sum().sort_values(ascending=False)
)
type_colors = [
    modern_colors["primary"],
    modern_colors["success"],
    modern_colors["warning"],
    modern_colors["danger"],
]
bars = ax.bar(
    infra_summary.index,
    infra_summary.values,
    color=type_colors[: len(infra_summary)],
    alpha=0.8,
    edgecolor="white",
    linewidth=2,
)
for bar, value in zip(bars, infra_summary.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(infra_summary.values) * 0.01,
        f"${value:.2f}",
        ha="center",
        va="bottom",
        color="white",
        fontweight="bold",
    )
ax.set_title(
    "Infrastructure Cost by Project Type (Bar Chart with Value Labels)",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20,
)
ax.set_xlabel("Project Type", color="white", fontweight="bold")
ax.set_ylabel("Total Infrastructure Cost ($)", color="white", fontweight="bold")
ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)

# 8. Video Project Storage Distribution and Costs
video_projects = df[df["Project Type"] == "video"]
if len(video_projects) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0a0a0a")
    ax1.set_facecolor("#0a0a0a")
    n, bins, patches = ax1.hist(
        video_projects["Storage GB"],
        bins=20,
        alpha=0.8,
        color=modern_colors["primary"],
        edgecolor="white",
        linewidth=0.5,
    )
    for i, patch in enumerate(patches):
        if bins[i] <= 2:
            patch.set_facecolor(modern_colors["success"])
        else:
            patch.set_facecolor(modern_colors["warning"])
    ax1.axvline(
        x=2,
        color=modern_colors["danger"],
        linestyle="--",
        alpha=0.8,
        linewidth=3,
        label="Free Storage Limit (2GB)",
    )
    ax1.set_title(
        "Video Project Storage Distribution (Histogram with Free Storage Limit)",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    ax1.set_xlabel("Storage Size (GB)", color="white", fontweight="bold")
    ax1.set_ylabel("Number of Projects", color="white", fontweight="bold")
    ax1.tick_params(colors="white")
    ax2.set_facecolor("#0a0a0a")
    video_costs = video_projects["Infra Cost"]
    n2, bins2, patches2 = ax2.hist(
        video_costs,
        bins=15,
        alpha=0.8,
        color=modern_colors["warning"],
        edgecolor="white",
        linewidth=0.5,
    )
    for i, patch in enumerate(patches2):
        if bins2[i] == 0:
            patch.set_facecolor(modern_colors["success"])
    ax2.set_title(
        "Video Infrastructure Costs (Histogram Showing Cost Distribution)",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    ax2.set_xlabel("Infrastructure Cost ($)", color="white", fontweight="bold")
    ax2.set_ylabel("Number of Projects", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig)

# 9. Project Type Distribution Donut Chart
st.subheader("Project Type Distribution")
fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
type_counts = df["Project Type"].value_counts()
type_colors_donut = [
    modern_colors["primary"],
    modern_colors["success"],
    modern_colors["warning"],
    modern_colors["danger"],
][: len(type_counts)]
wedges, texts, autotexts = ax.pie(
    type_counts.values,
    labels=type_counts.index,
    colors=type_colors_donut,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.85,
    textprops={"color": "white", "fontweight": "bold"},
)
centre_circle = plt.Circle((0, 0), 0.60, fc="#0a0a0a")
ax.add_artist(centre_circle)
total_projects = len(df)
ax.text(
    0,
    0,
    f"Total Projects\n{total_projects}",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="white",
)
ax.set_title(
    "Project Type Distribution (Donut Chart with Percentages)",
    fontsize=18,
    fontweight="bold",
    color="white",
    pad=30,
)
plt.tight_layout()
st.pyplot(fig)

# 10. Profit Margin by Project Type Boxplot
st.subheader("Net Profit by Project Type")
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
project_types = df["Project Type"].unique()
profit_data = [
    df[df["Project Type"] == ptype]["Net Profit"].values for ptype in project_types
]
box_parts = ax.boxplot(
    profit_data,
    tick_labels=project_types,
    patch_artist=True,
    boxprops=dict(facecolor="none", edgecolor="white", linewidth=2),
    medianprops=dict(color=modern_colors["warning"], linewidth=3),
    whiskerprops=dict(color="white", linewidth=2),
    capprops=dict(color="white", linewidth=2),
)
for patch, color in zip(box_parts["boxes"], type_colors_donut):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(
    y=0,
    color=modern_colors["danger"],
    linestyle="--",
    alpha=0.7,
    linewidth=2,
    label="Break-even",
)
ax.set_title(
    "Net Profit Distribution by Project Type (Box Plot Comparison)",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=20,
)
ax.set_xlabel("Project Type", color="white", fontweight="bold")
ax.set_ylabel("Net Profit ($)", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor="#0a0a0a", edgecolor="white", labelcolor="white")
plt.tight_layout()
st.pyplot(fig)

# -------------------- AI Model: Predict Growth --------------------
st.header("AI Model: Predict Platform Growth")
st.markdown("""
This section uses a linear regression to predict platform fee revenue growth based on the number of projects. You can adjust the distribution of project types below to see how it impacts revenue and model predictions.
""")

# --- Project Type Distribution: Randomized for Each Simulation ---
# Instead of user sliders, randomly generate project type weights for each run


def random_project_type_weights():
    # Generate 4 random numbers, normalize to sum to 1
    weights = np.random.dirichlet(np.ones(len(PROJECT_TYPES)), 1)[0]
    return {k: v for k, v in zip(PROJECT_TYPES, weights)}


project_type_weights = random_project_type_weights()
st.info(
    "Project type mix for this simulation: "
    + ", ".join(
        [f"{k.capitalize()}: {v * 100:.1f}%" for k, v in project_type_weights.items()]
    ),
    icon="📊",
)

# --- Simulate project data
project_data = []
for _ in range(num_projects):
    project_value = round(np.random.uniform(project_min, project_max), 2)
    platform_fee = (
        7 if project_value < 250 else round(project_value * 0.1, 2)
    )  # Ensure 10% everywhere
    stripe_fee = round(project_value * STRIPE_PERCENT + STRIPE_FIXED, 2)
    stripe_transfer_fee = round(
        project_value * STRIPE_TRANSFER_PERCENT + STRIPE_TRANSFER_FIXED, 2
    )
    project_type = np.random.choice(
        PROJECT_TYPES, p=[project_type_weights[k] for k in PROJECT_TYPES]
    )
    if project_type == "video":
        storage_gb = round(np.random.uniform(1, 10), 2)
        if storage_gb <= INFRA_COSTS["video"]["free_gb"]:
            infra_cost = 0
        else:
            additional_gb = storage_gb - INFRA_COSTS["video"]["free_gb"]
            infra_cost = round(additional_gb * INFRA_COSTS["video"]["cost_per_gb"], 2)
    else:
        storage_gb = 0
        infra_cost = INFRA_COSTS[project_type]
    net_profit = platform_fee - stripe_fee - stripe_transfer_fee - infra_cost
    freelancer_payout = project_value
    project_data.append(
        {
            "Project Value": project_value,
            "Project Type": project_type,
            "Storage GB": storage_gb,
            "Platform Fee": platform_fee,
            "Stripe Fee": stripe_fee,
            "Stripe Transfer Fee": stripe_transfer_fee,
            "Infra Cost": infra_cost,
            "Net Profit": net_profit,
            "Freelancer Payout": freelancer_payout,
        }
    )
df = pd.DataFrame(project_data)
df["Project Tier"] = df["Project Value"].apply(
    lambda x: "Small" if x < 250 else "Standard"
)
df["Margin (%)"] = df["Net Profit"] / df["Platform Fee"] * 100

# --- Simulate growth data with variable project type distribution ---
project_counts = np.arange(100, 10001, 100)
revenues = []
type_counts_matrix = []
for n in project_counts:
    temp_data = []
    type_counts = {k: 0 for k in PROJECT_TYPES}
    for _ in range(n):
        project_type = np.random.choice(
            PROJECT_TYPES, p=[project_type_weights[k] for k in PROJECT_TYPES]
        )
        type_counts[project_type] += 1
        project_value = round(np.random.uniform(project_min, project_max), 2)
        platform_fee = (
            7 if project_value < 250 else round(project_value * 0.1, 2)
        )  # Ensure 10% everywhere
        temp_data.append(platform_fee)
    revenues.append(np.sum(temp_data))
    type_counts_matrix.append([type_counts[k] for k in PROJECT_TYPES])

X = project_counts.reshape(-1, 1)
y = np.array(revenues)
model = LinearRegression().fit(X, y)
pred_projects = st.slider(
    "Predict for # Projects",
    100,
    20000,
    num_projects,
    100,
    key="predict_projects_slider",
)
pred_revenue = model.predict(np.array([[pred_projects]]))[0]

# --- Business-Focused AI Prediction Visualization ---
st.subheader("Revenue Growth Forecast: Business View")
std_err = np.std(y - model.predict(X))
interval_low = pred_revenue - 1.96 * std_err
interval_high = pred_revenue + 1.96 * std_err

# Estimate COGS, Gross Profit, EBITDA, Net Profit for the forecasted project count
sim_temp_data = []
sim_stripe_fees = []
sim_stripe_transfer_fees = []
sim_infra_costs = []
sim_net_profits = []
for _ in range(pred_projects):
    project_value = round(np.random.uniform(project_min, project_max), 2)
    platform_fee = (
        7 if project_value < 250 else round(project_value * 0.1, 2)
    )  # Ensure 10% everywhere
    stripe_fee = round(project_value * STRIPE_PERCENT + STRIPE_FIXED, 2)
    stripe_transfer_fee = round(
        project_value * STRIPE_TRANSFER_PERCENT + STRIPE_TRANSFER_FIXED, 2
    )
    project_type = np.random.choice(
        PROJECT_TYPES, p=[project_type_weights[k] for k in PROJECT_TYPES]
    )
    if project_type == "video":
        storage_gb = round(np.random.uniform(1, 10), 2)
        if storage_gb <= INFRA_COSTS["video"]["free_gb"]:
            infra_cost = 0
        else:
            additional_gb = storage_gb - INFRA_COSTS["video"]["free_gb"]
            infra_cost = round(additional_gb * INFRA_COSTS["video"]["cost_per_gb"], 2)
    else:
        infra_cost = INFRA_COSTS[project_type]
    net_profit = platform_fee - stripe_fee - stripe_transfer_fee - infra_cost
    sim_temp_data.append(platform_fee)
    sim_stripe_fees.append(stripe_fee)
    sim_stripe_transfer_fees.append(stripe_transfer_fee)
    sim_infra_costs.append(infra_cost)
    sim_net_profits.append(net_profit)

sim_platform_fee = np.sum(sim_temp_data)
sim_cogs = (
    np.sum(sim_stripe_fees) + np.sum(sim_stripe_transfer_fees) + np.sum(sim_infra_costs)
)
sim_gross_profit = np.sum(sim_net_profits)
sim_ebitda = sim_gross_profit  # No OPEX
sim_net_profit = sim_gross_profit
sim_avg_profit_per_project = sim_net_profit / pred_projects
sim_avg_revenue_per_project = sim_platform_fee / pred_projects

# Arrange metrics in columns
col1, col2 = st.columns(2)
with col1:
    st.metric("Forecasted Revenue", f"${sim_platform_fee:,.0f}")
    st.metric("Gross Profit", f"${sim_gross_profit:,.0f}")
    st.metric("Net Profit", f"${sim_net_profit:,.0f}")
    st.metric("Revenue Growth per 1,000 Projects", f"${model.coef_[0] * 1000:,.0f}")
with col2:
    st.metric("COGS (Stripe + Infra)", f"${sim_cogs:,.0f}")
    st.metric("EBITDA", f"${sim_ebitda:,.0f}")
    st.metric("Avg Revenue per Project", f"${sim_avg_revenue_per_project:,.2f}")
    st.metric("Avg Profit per Project", f"${sim_avg_profit_per_project:,.2f}")

# Add the 95% confidence interval range at the bottom spanning both columns
st.metric(
    "95% Revenue Confidence Interval", f"${interval_low:,.0f} - ${interval_high:,.0f}"
)

# Visual: Revenue Growth Curve with Highlighted Prediction (smaller font)
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
ax.plot(
    project_counts,
    y,
    label="Simulated Revenue",
    color=modern_colors["primary"],
    linewidth=2,
)
ax.scatter(
    [pred_projects],
    [pred_revenue],
    color=modern_colors["warning"],
    s=120,
    marker="o",
    label="Your Forecast",
    edgecolor="white",
    linewidth=1.5,
)
ax.fill_between(
    [pred_projects - 100, pred_projects + 100],
    [interval_low, interval_low],
    [interval_high, interval_high],
    color=modern_colors["orange"],
    alpha=0.18,
    label="95% Range",
)
ax.set_xlabel("Number of Projects", color="white", fontweight="bold", fontsize=11)
ax.set_ylabel("Platform Fee Revenue ($)", color="white", fontweight="bold", fontsize=11)
ax.set_title(
    f"Forecasted Revenue for {pred_projects:,} Projects",
    fontsize=14,
    fontweight="bold",
    color="white",
    pad=16,
)
ax.tick_params(colors="white", labelsize=10)
ax.legend(facecolor="#2d2d2d", edgecolor="white", labelcolor="white", fontsize=10)
plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Built with Streamlit · Modern UI · AI-powered growth prediction with variable project type mix"
)
