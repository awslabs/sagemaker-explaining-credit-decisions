from jinja2 import Template
from bokeh.embed import components
from bokeh.resources import CDN
import pandas as pd
from pathlib import Path

from package import utils
from package import visuals


current_folder = utils.get_current_folder(globals())


def create_report_html(output, x_axis_label):
    explanation_summary = visuals.summary_explanation(output)
    summary_waterfall = visuals.WaterfallChart(
        baseline=explanation_summary['expected_value'],
        shap_values=explanation_summary['shap_values'],
        names=explanation_summary['feature_names'],
        max_features=10,
        x_axis_label=x_axis_label,
    )

    explanation = visuals.detailed_explanation(output)
    detailed_waterfall = visuals.WaterfallChart(
        baseline=explanation['expected_value'],
        shap_values=explanation['shap_values'],
        names=explanation['feature_names'],
        feature_values=explanation['feature_values'],
        max_features=10,
        x_axis_label=x_axis_label
    )

    script_summary, div_summary = components(summary_waterfall._figure)
    script_detailed, div_detailed = components(detailed_waterfall._figure)
    df = pd.DataFrame(explanation)
    df = df[['feature_names', 'feature_values', 'shap_values']]
    df = df.sort_values(by='shap_values')
    table = df.to_html(
        classes=['table', 'table-hover', 'table-sm', 'table-striped'],
        index=False, justify='left', border=0
    )

    with open(Path(current_folder, 'template', 'template.html')) as openfile:
        template_html = openfile.read()

    template = Template(template_html)
    output_html = template.render(
        bokeh_js=CDN.render(),
        script_summary=script_summary,
        div_summary=div_summary,
        script_detailed=script_detailed,
        div_detailed=div_detailed,
        table=table
    )
    return output_html
