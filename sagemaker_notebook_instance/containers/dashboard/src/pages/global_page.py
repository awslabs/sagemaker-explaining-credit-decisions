import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import shap
import sys

sys.path.append('..')
from shared import load_explanation_group


def analysis_selectbox():
    analysis = st.selectbox(
        label='Select analysis:',
        options=['summary', 'dependency'],
        index=0,
        format_func=lambda e: {
            'summary': "Identify features that have the greatest effect on model predictions.",
            'dependency': 'Understand the effect a single feature has on model predictions.'
        }[e]
    )
    return analysis


def notes(explanation_group):
    st.markdown('### Notes')
    st.write('Showing', len(explanation_group), 'explanations in total.')


def show(explanation_group_path):
    explanation_group = load_explanation_group(explanation_group_path)
    st.title('Group Explanations')
    st.write('for *{}*.'.format(explanation_group_path.split('/')[-2]))
    st.markdown('&nbsp;')
    analysis = analysis_selectbox()
    if analysis == 'summary':
        summary(explanation_group)
    elif analysis == 'dependency':
        dependency(explanation_group)
    notes(explanation_group)


def num_features_number_input():
    num_features = st.number_input(
        label='Select number of features to show:',
        min_value=1,
        max_value=50,
        value=15
    )
    return num_features


def details_checkbox():
    return st.checkbox(label='Show all individuals?')


def extract_feature_names(explanation_group):
    sample = explanation_group[0]
    names = list(sample['explanation']['shap_values'].keys())
    return names


def summary_plot(explanation_group, details, num_features, plot_placeholder):
    names = extract_feature_names(explanation_group)
    shap_values = [[r['explanation']['shap_values'][e] for e in names] for r in explanation_group]
    features = [[r['features'][e] for e in names] for r in explanation_group]
    plot_type = 'dot' if details else 'bar'
    shap.summary_plot(
        shap_values=np.array(shap_values),
        features=np.array(features),
        feature_names=names,
        max_display=num_features,
        plot_type=plot_type
    )
    plot_placeholder.pyplot(bbox_inches='tight')
    plt.clf()


def summary(explanation_group):
    st.header('Summary Plot')
    plot_placeholder = st.empty()
    details = details_checkbox()
    num_features = num_features_number_input()
    summary_plot(explanation_group, details, num_features, plot_placeholder)


def interaction_checkbox():
    return st.checkbox(label='Show strongest feature interaction?')


def dependency(explanation_group):
    names = extract_feature_names(explanation_group)
    feature_idx = st.selectbox(
        label="Select feature:",
        options=range(len(names)),
        format_func=lambda e: names[e],
        index=1
    )
    plot_placeholder = st.empty()
    interaction = interaction_checkbox()
    shap_values = [[r['explanation']['shap_values'][e] for e in names] for r in explanation_group]
    features = [[r['features'][e] for e in names] for r in explanation_group]
    interaction_index = "auto" if interaction else None
    shap.dependence_plot(
        ind=feature_idx,
        shap_values=np.array(shap_values),
        features=np.array(features),
        feature_names=names,
        interaction_index=interaction_index
    )
    plot_placeholder.pyplot(bbox_inches='tight')
    plt.clf()
