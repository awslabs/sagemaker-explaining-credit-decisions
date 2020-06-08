import streamlit as st
import pandas as pd
import random
import sys
import urllib.parse

from package import visuals
from package.reports import reports

sys.path.append('..')
import session_state
from shared import load_explanation_group


state = session_state.get(sample_id=0)


def download_report_url(report_html, report_name):
    report_html = urllib.parse.quote(report_html, safe='')
    report_name = f'{report_name}.html'
    href = f'<a href="data:text/html,{report_html}" download="{report_name}">Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)
    

def show(explanation_group_path):
    explanation_group = load_explanation_group(explanation_group_path)
    sample_id_placeholder = st.sidebar.empty()

    state.sample_id = sample_id_placeholder.text_input(
        label='Select individual (by ID):',
        value=state.sample_id
    )
    
    random_sample = st.sidebar.button(
        label='Random'
    )
    
    if random_sample:
        state.sample_id = sample_id_placeholder.text_input(
                label='Select individual (by ID):',
                value=random.randint(0, len(explanation_group))
            )
        
    record = explanation_group[int(state.sample_id)]
    data = record['data']
    features = record['features']
    explanation = record['explanation']
    prediction = record['prediction']

    st.title('Individual Explanation')
    st.write('for #*{}* from *{}*.'.format(state.sample_id, explanation_group_path.split('/')[-2]))

    st.markdown('&nbsp;')
    if sum(explanation['shap_values'].values()) < 0:
        st.success('Model predicted this individual as lower risk: {:.2%} compared to {:.2%} baseline.'.format(
            prediction, visuals.log_odds_to_proba(explanation['expected_value'])
        ))
    else:
        st.error('Model predicted this individual as higher risk: {:.2%} compared to {:.2%} baseline.'.format(
            prediction, visuals.log_odds_to_proba(explanation['expected_value'])
        ))
    st.markdown('&nbsp;')

    st.subheader('Waterfall Chart')
    x_axis_label = 'Credit Default Risk Score (%)'
    names = explanation['shap_values'].keys()
    detailed_waterfall = visuals.WaterfallChart(
        baseline=explanation['expected_value'],
        shap_values=[explanation['shap_values'][n] for n in names],
        names=names,
        feature_values=[record['features'][n] for n in names],
        descriptions=[record['features'][n] for n in names],
        max_features=10,
        x_axis_label=x_axis_label
    )
    st.bokeh_chart(detailed_waterfall._figure)

    st.markdown('&nbsp;')
    st.markdown('### Data')
    st.markdown('Showing data prior to feature pre-processing. May include fields not used by the model.')
    st.json(data)

    st.markdown('### Features')
    st.markdown('Showing features after pre-processing and their associated SHAP values.')
    df_features = pd.DataFrame([features, explanation['shap_values']]).T
    df_features.columns = ['feature_value', 'shap_value']
    st.dataframe(df_features)

    st.markdown('### Export')
    report_html = reports.create_report_html(record, x_axis_label)
    report_name = f'explanation_report_for_{state.sample_id}'
    download_report_url(report_html, report_name)
