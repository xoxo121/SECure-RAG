import logging.config
import logging.handlers
import os
import streamlit as st
import yaml

st.set_page_config(
    page_title="SECure RAG", page_icon="./images/logo.png", layout="wide"
)

if not os.path.exists(".logs"):
    os.mkdir(".logs")
if "initialized" not in st.session_state or not st.session_state.initialized:
    if os.path.exists("logging.yaml"):
        logging.config.dictConfig(yaml.load(open("logging.yaml"), Loader=yaml.FullLoader))

# Page routing
finance_page = st.Page("./agent_page.py", title="SECure RAG")
explainability_page = st.Page(
    "./explainability_agent_page.py", title="Explainability Agent"
)
pg = st.navigation([finance_page, explainability_page], position="sidebar")
pg.run()
