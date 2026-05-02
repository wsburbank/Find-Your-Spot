"""
Streamlit base utilities for Find Your Spot app.
"""
import streamlit as st


def initialize_session_state(page_name: str, show_header_title: bool = True):
    """Initialize page config and common session state for the app.

    Args:
        page_name: Title shown in the browser tab.
        show_header_title: If True, render the page name as a fixed header overlay.
    """
    st.set_page_config(layout="wide", page_title=page_name, page_icon="\U0001F4CD")

    header_title_html = f"""
        <script>
            document.querySelectorAll('.header-title').forEach(el => el.remove());
        </script>
        <div class="header-title">{page_name}</div>
    """ if show_header_title else ""

    st.markdown(
        f"""
        <style>
            :root {{
                color-scheme: light dark;
            }}

            .stAppHeader {{
                background-color: light-dark(#ffffff, #0e1117);
            }}

            [data-testid="stSidebarHeader"] {{
                padding: 0rem;
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }}

            .header-title {{
                position: fixed !important;
                top: 10px !important;
                left: 50%;
                transform: translateX(-50%);
                font-size: 2rem;
                font-weight: 500;
                color: light-dark(#333, #fafafa);
                z-index: 999999;
                pointer-events: none;
                background: transparent;
            }}

            .stMarkdown {{
                overflow: visible !important;
            }}
        </style>
        {header_title_html}
        """,
        unsafe_allow_html=True,
    )
