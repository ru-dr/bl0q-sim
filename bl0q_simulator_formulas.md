# bl0q Platform Simulator: Definitions & Formulas

## Key Definitions

- **Project Value**: The total value of a project, randomly simulated between user-defined min and max.
- **Platform Fee**: The fee charged by the platform, calculated as a percentage of project value. For projects below the threshold, a configurable percentage (`flat_fee_percent`) is used; for projects at or above the threshold, a different percentage (`commission_rate`) is used.
- **Stripe Fee**: Payment processing fee, calculated as a percentage of project value plus a fixed fee.
- **Stripe Platform Fee**: Additional fee on the platform fee, as a percentage.
- **International Fee**: Extra fee for international transactions, as a percentage of project value (paid by the client, not a platform cost).
- **Transfer Fee**: Fee for transferring funds, as a percentage of project value, capped at a maximum.
- **Infra Cost**: Infrastructure cost, varies by project type and storage used (for video projects).
- **Net Profit**: Profit per project after all direct platform costs (excludes CAC and team costs; does not subtract international fee).
- **Freelancer Payout**: The amount paid out to the freelancer (equals project value).
- **COGS**: Cost of Goods Sold (Stripe fees + Infrastructure costs; excludes international fee).
- **Gross Profit**: Total revenue minus COGS.
- **CAC (Customer Acquisition Cost)**: Cost to acquire a customer, set per customer.
- **Team Salaries**: Annual cost of all team members, including a payroll multiplier for benefits.
- **EBITDA**: Earnings before interest, taxes, depreciation, and amortization (Gross Profit - CAC - Team Salaries).
- **Net Profit**: EBITDA (in this simulation, as no other expenses are modeled).
- **LTV (Customer Lifetime Value)**: Gross profit per customer multiplied by average customer lifetime (in months).
- **Margin (%)**: Net profit as a percentage of platform fee.
- **Customer Paid (incl. FX)**: The total amount the client pays, including international fee: `project_value + international_fee`.

## Core Formulas

### Project Simulation
- **Project Value**: `project_value = np.random.uniform(min_val, max_val)`
- **Platform Fee**:
  - If `project_value < fee_threshold`: `platform_fee = project_value * (flat_fee_percent / 100)`
  - Else: `platform_fee = project_value * (commission_rate / 100)`
- **Stripe Fee**: `stripe_fee = project_value * STRIPE_PERCENT + STRIPE_FIXED`
- **Stripe Platform Fee**: `stripe_platform_fee = platform_fee * STRIPE_PLATFORM_PERCENT`
- **Transfer Fee**: `transfer_fee = min(project_value * TRANSFER_FEE_PERCENT, TRANSFER_FEE_MAX)`
- **International Fee**: `international_fee = project_value * INTERNATIONAL_FEE_PERCENT`  # (client paid, not platform cost)
- **Infra Cost**:
  - For video: If `storage_gb > free_gb`, `infra_cost = (storage_gb - free_gb) * cost_per_gb`, else `infra_cost = 0`
  - For others: `infra_cost = INFRA_COSTS[project_type]`
- **Net Profit**: `net_profit = platform_fee - stripe_fee - stripe_platform_fee - infra_cost`  # (no longer subtracts international_fee)
- **Freelancer Payout**: `freelancer_payout = project_value`
- **Customer Paid (incl. FX)**: `customer_paid = project_value + international_fee`

### Aggregates & Metrics
- **Total Revenue**: `total_revenue = sum(platform_fee)`
- **Total Stripe Fees**: `total_stripe_fees = sum(stripe_fee) + sum(stripe_platform_fee)`
- **Total International Fees**: `total_international_fees = sum(international_fee)`  # (client paid, not platform cost)
- **Total Infra Costs**: `total_infra_costs = sum(infra_cost)`
- **Gross Profit**: `gross_profit = total_revenue - total_stripe_fees - total_infra_costs`  # (no longer subtracts international_fee)
- **Number of Customers**: `num_customers = ceil(num_projects / avg_projects_per_customer)`
- **Total CAC**: `total_cac = num_customers * cac_per_customer`
- **Annual Team Cost**: `annual_team_cost = monthly_team_cost * 12 if enable_salaries else 0`
- **EBITDA**: `ebitda = gross_profit - total_cac - annual_team_cost`
- **Net Profit**: `net_profit = ebitda`
- **Gross Profit per Customer**: `gross_profit_per_customer = gross_profit / num_customers if num_customers > 0 else 0`
- **Average Customer Lifetime (months)**: `avg_customer_lifetime_months = 1 / (churn_rate / 100.0) if churn_rate > 0 else 12`
- **LTV**: `ltv = gross_profit_per_customer * avg_customer_lifetime_months`
- **Margin (%)**: `margin = (net_profit / platform_fee) * 100`

### Historical Data Simulation (for AI Forecast)
- **General Formula**:
  - `hist_metric = np.cumsum(np.random.normal(metric/24, abs(metric/60), 24)) + np.linspace(metric*0.5, metric, 24)`
  - For margin: `hist_margin = np.cumsum(np.random.normal(mean_margin/24, 1, 24)) + np.linspace(mean_margin*0.5, mean_margin, 24)`
  - For COGS: `hist_cogs = np.cumsum(np.random.normal((total_stripe_fees+total_infra_costs)/24, 100, 24)) + np.linspace((total_stripe_fees+total_infra_costs)*0.5, (total_stripe_fees+total_infra_costs), 24)`

### Forecasting
- **Linear Regression Forecast**: For each metric, fit a linear regression to historical data and predict the next 12 months.

### Churn-based Customer Base Decay
- **Customer Base**: `customers[n] = customers[n-1] * (1 - churn_rate/100)`

---

This document summarizes all key definitions and formulas used in the bl0q Platform Simulator for financial modeling and forecasting.
