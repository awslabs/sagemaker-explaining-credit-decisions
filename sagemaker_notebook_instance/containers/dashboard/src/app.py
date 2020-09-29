from pathlib import Path
import streamlit as st

from package import utils

from pages import local_page, global_page
from shared import list_explanation_groups


def explanation_group_selectbox():
    paths = list_explanation_groups()
    path = st.sidebar.selectbox(
        label='Select explanation group:',
        options=paths,
        format_func=lambda e: e.split('/')[-2]
    )
    return path


def explanation_scope_selectbox():
    explanation_scope = st.sidebar.selectbox(
        label='Select explanation scope:',
        options=["local", "global"],
        index=1,
        format_func=lambda e: {'local': 'Individual', 'global': 'Group'}[e]
    )
    return explanation_scope


if __name__ == "__main__":
    current_folder = utils.get_current_folder(globals())
    st.sidebar.markdown('# Explanations Dashboard')
    explanation_group_path = explanation_group_selectbox()
    explanation_scope = explanation_scope_selectbox()
    if explanation_scope == "local":
        local_page.show(explanation_group_path)
    elif explanation_scope == "global":
        global_page.show(explanation_group_path)
