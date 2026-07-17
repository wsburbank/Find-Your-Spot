"""
Tax estimation utilities for Find Your Spot.

Calculates estimated annual state/local taxes for a city based on user-provided
income, home value, and annual spending. Uses each city's real tax rates
(state income tax, property tax, sales tax) from the dataset.

Adjustments for local price levels:
  - Spending is scaled by the BEA Goods RPP (consumer goods price index).
  - Home value is scaled by the Housing Cost Index (home price + rent relative
    to national median). This reflects that an equivalent home costs more in
    expensive markets, so property taxes are higher.
"""

import pandas as pd
import numpy as np


def estimate_taxes(
    income: float,
    home_value: float,
    annual_expenses: float,
    state_income_tax_rate: float,
    avg_property_tax_rate: float,
    state_sales_tax_rate: float,
    goods_rpp: float = 100.0,
    housing_cost_index: float = 100.0,
) -> dict:
    """Estimate annual taxes for a single city.

    Args:
        income: User's estimated annual income.
        home_value: User's estimated home purchase price (at national avg market).
        annual_expenses: User's estimated annual taxable spending (at national avg prices).
        state_income_tax_rate: City's state income tax rate (%).
        avg_property_tax_rate: City's effective property tax rate (%).
        state_sales_tax_rate: City's combined sales tax rate (%).
        goods_rpp: BEA Goods Regional Price Parity (100 = national avg).
            Scales annual_expenses to reflect local goods prices.
        housing_cost_index: Housing Cost Index (100 = national avg).
            Scales home_value to reflect local housing prices.

    Returns:
        Dict with income_tax, property_tax, sales_tax, adjusted values, and total.
    """
    # Scale spending by local goods price level
    rpp_factor = goods_rpp / 100.0 if pd.notna(goods_rpp) else 1.0
    adjusted_spending = annual_expenses * rpp_factor

    # Scale home value by local housing cost level
    housing_factor = housing_cost_index / 100.0 if pd.notna(housing_cost_index) else 1.0
    adjusted_home_value = home_value * housing_factor

    income_tax = income * (state_income_tax_rate / 100) if pd.notna(state_income_tax_rate) else np.nan
    property_tax = adjusted_home_value * (avg_property_tax_rate / 100) if pd.notna(avg_property_tax_rate) else np.nan
    sales_tax = adjusted_spending * (state_sales_tax_rate / 100) if pd.notna(state_sales_tax_rate) else np.nan

    if any(pd.isna(v) for v in [income_tax, property_tax, sales_tax]):
        total = np.nan
    else:
        total = income_tax + property_tax + sales_tax

    return {
        "income_tax": income_tax,
        "property_tax": property_tax,
        "sales_tax": sales_tax,
        "adjusted_spending": adjusted_spending,
        "adjusted_home_value": adjusted_home_value,
        "total": total,
    }


def add_estimated_taxes_column(
    df: pd.DataFrame,
    income: float,
    home_value: float,
    annual_expenses: float,
) -> pd.DataFrame:
    """Add an ``estimated_annual_taxes`` column to a cities DataFrame.

    Spending is adjusted by the Goods RPP index and home value is adjusted by
    the Housing Cost Index to reflect local price levels.

    Args:
        df: Cities DataFrame (must contain state_income_tax_rate,
            avg_property_tax_rate, state_sales_tax_rate columns;
            optionally goods_rpp and cost_of_living_index).
        income: User's estimated annual income.
        home_value: User's estimated home purchase price.
        annual_expenses: User's estimated annual taxable spending.

    Returns:
        The same DataFrame with ``estimated_annual_taxes`` added in-place.
    """
    # Scale spending by local goods prices (RPP index, 100 = national avg)
    if "goods_rpp" in df.columns:
        rpp_factor = df["goods_rpp"].fillna(100.0) / 100.0
    else:
        rpp_factor = 1.0

    # Scale home value by local housing costs (Housing Cost Index, 100 = national avg)
    if "cost_of_living_index" in df.columns:
        housing_factor = df["cost_of_living_index"].fillna(100.0) / 100.0
    else:
        housing_factor = 1.0

    adjusted_spending = annual_expenses * rpp_factor
    adjusted_home_value = home_value * housing_factor

    df["estimated_annual_taxes"] = (
        income * df["state_income_tax_rate"].div(100)
        + adjusted_home_value * df["avg_property_tax_rate"].div(100)
        + adjusted_spending * df["state_sales_tax_rate"].div(100)
    )
    return df
