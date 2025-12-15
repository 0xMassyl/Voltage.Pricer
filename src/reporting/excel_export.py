import pandas as pd
import numpy as np
import io
from typing import Dict, Any, cast

import xlsxwriter
from xlsxwriter.workbook import Workbook
from xlsxwriter.chart import Chart


def export_pricing_to_excel(
    df_costs: pd.DataFrame,
    load_curve: pd.Series,
    volume: float,
    market_prices: Dict[str, float],
    final_price: float,
) -> io.BytesIO:
    """
    Generates a professional Excel pricing output.

    Structure:
    - Commercial summary (key figures for decision makers)
    - Cost breakdown (pricing transparency)
    - Hourly load curve with embedded chart

    Target: client-ready file, no post-processing needed.
    """
    output = io.BytesIO()

    # Use XlsxWriter directly for full formatting and chart control
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        # Explicit typing for static analyzers
        workbook = cast(Workbook, writer.book)

        # ------------------------------------------------------------------
        # Excel formats
        # ------------------------------------------------------------------
        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#0E3A5D",
                "font_color": "white",
                "border": 1,
            }
        )

        currency_format = workbook.add_format(
            {"num_format": "€#,##0.00"}
        )

        bold_currency = workbook.add_format(
            {
                "bold": True,
                "num_format": "€#,##0.00",
                "font_color": "#FF6B00",
            }
        )

        # ==============================================================
        # SHEET 1: COMMERCIAL SUMMARY
        # ==============================================================
        df_summary = pd.DataFrame(
            {
                "Key Metric": [
                    "Annual Volume (MWh)",
                    "Baseload Price (CAL)",
                    "Peakload Price (CAL)",
                    "",
                    "FINAL SALES PRICE (€/MWh)",
                    "Estimated Total Budget (€)",
                ],
                "Value": [
                    volume,
                    market_prices.get("CAL_BASE", 0.0),
                    market_prices.get("CAL_PEAK", 0.0),
                    np.nan,
                    final_price,
                    final_price * volume,
                ],
            }
        )

        df_summary.to_excel(
            writer,
            sheet_name="Commercial Summary",
            index=False,
            startrow=1,
        )

        worksheet_sum = writer.sheets["Commercial Summary"]

        worksheet_sum.write(
            0,
            0,
            "ELECTRICITY SUPPLY OFFER – EXECUTIVE SUMMARY",
            header_format,
        )

        worksheet_sum.set_column("A:A", 30)
        worksheet_sum.set_column("B:B", 20, currency_format)

        # Highlight final price and total budget
        worksheet_sum.write(6, 1, final_price, bold_currency)
        worksheet_sum.write(7, 1, final_price * volume, bold_currency)

        # ==============================================================
        # SHEET 2: COST BREAKDOWN
        # ==============================================================
        df_costs.to_excel(
            writer,
            sheet_name="Cost Breakdown",
            index=False,
            startrow=1,
        )

        worksheet_cost = writer.sheets["Cost Breakdown"]

        worksheet_cost.write(
            0,
            0,
            "PRICE STRUCTURE DETAILS (€/MWh)",
            header_format,
        )

        worksheet_cost.set_column("A:A", 35)
        worksheet_cost.set_column("B:B", 20, currency_format)

        # ==============================================================
        # SHEET 3: LOAD CURVE (DATA + CHART)
        # ==============================================================
        df_load = load_curve.to_frame(name="Power (MW)")
        df_load.index.name = "Timestamp"

        df_load.to_excel(
            writer,
            sheet_name="Hourly Load Data",
        )

        worksheet_load = writer.sheets["Hourly Load Data"]
        worksheet_load.set_column("A:A", 20)

        # Explicit typing to access chart methods cleanly
        chart = cast(
            Chart, workbook.add_chart({"type": "line"})
        )

        # Plot a representative weekly window (first 168 hours)
        chart.add_series(
            {
                "name": "=Hourly Load Data!$B$1",
                "categories": "=Hourly Load Data!$A$2:$A$169",
                "values": "=Hourly Load Data!$B$2:$B$169",
                "line": {"color": "#0E3A5D"},
            }
        )

        chart.set_title(
            {"name": "Consumption Profile (Typical Week)"}
        )
        chart.set_x_axis({"name": "Hour"})
        chart.set_y_axis({"name": "MW"})
        chart.set_size({"width": 720, "height": 400})

        worksheet_load.insert_chart("D2", chart)

    output.seek(0)
    return output
