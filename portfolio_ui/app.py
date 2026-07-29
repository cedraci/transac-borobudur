"""Entry point: streamlit run portfolio_ui/app.py"""

import streamlit as st

from portfolio_ui import sidebar
from portfolio_ui.views.analysis_page import analysis_page
from portfolio_ui.views.data_page import data_page
from portfolio_ui.views.placeholders import (
    backtest_page,
    market_data_page,
    optimization_page,
)
from portfolio_ui.views.risk_page import risk_page
from portfolio_ui.state import init_state


def main() -> None:
    st.set_page_config(page_title="Portfolio Construction", layout="wide")
    init_state(st.session_state)
    sidebar.render(st.session_state)

    pages = [
        st.Page(data_page, title="Data", default=True),
        st.Page(market_data_page, title="Market Data"),
        st.Page(analysis_page, title="Analysis"),
        st.Page(risk_page, title="Risk"),
        st.Page(optimization_page, title="Optimization"),
        st.Page(backtest_page, title="Backtest"),
    ]
    st.navigation(pages).run()


main()
