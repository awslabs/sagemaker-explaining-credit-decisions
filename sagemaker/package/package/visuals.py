import numpy as np
import bokeh
from bokeh import models
from bokeh import plotting
import pandas as pd


def log_odds_to_proba(log_odds):
    return np.exp(log_odds) / (1 + np.exp(log_odds))


def proba_to_log_odds(proba):
    return np.log(proba / (1 - proba))


def summary_explanation(output, level=0, separator="__"):
    data = {
        'shap_values': output['explanation']['shap_values']
    }
    if 'descriptions' in output:
        data['feature_descriptions'] = output['descriptions']
    df = pd.DataFrame(data)
    df.index.name = 'feature_names'
    df = df.reset_index()
    df["feature_names"] = df["feature_names"].apply(
        lambda e: separator.join(e.split(separator)[: level + 1])
    )
    df = df.groupby("feature_names")["shap_values"].sum()
    df = df.reset_index()
    if 'descriptions' in output:
        df["feature_descriptions"] = df["feature_names"].apply(
            lambda e: "All '{}' features combined.".format(e)
        )
    explanation = {
        'shap_values': df['shap_values'].tolist(),
        'expected_value': output['explanation']['expected_value'],
        'feature_names': df['feature_names'].tolist()
    }
    if 'descriptions' in output:
        explanation['feature_descriptions'] = df['feature_descriptions'].tolist()
    return explanation


def detailed_explanation(output):
    data = {
        'feature_values': output['features'],
        'shap_values': output['explanation']['shap_values']
    }
    if 'descriptions' in output:
        data['feature_descriptions'] = output['descriptions']
    df = pd.DataFrame(data)
    df.index.name = 'feature_names'
    df = df.reset_index()
    explanation = {
        'shap_values': df['shap_values'].tolist(),
        'expected_value': output['explanation']['expected_value'],
        'feature_names': df['feature_names'].tolist(),
        'feature_values': df['feature_values'].tolist()
    }
    if 'descriptions' in output:
        explanation['feature_descriptions'] = df['feature_descriptions'].tolist()
    return explanation


class WaterfallChart():
    def __init__(
        self,
        baseline,
        shap_values,
        names,
        feature_values=None,
        descriptions=None,
        x_range_width=500,
        x_range_padding=0.25,
        x_axis_label='Score (%)',
        title=None,
        baseline_color="#808080",
        positive_color="#FF5733",
        negative_color="#69AE35",
        sort_order='absolute_descending',
        max_features=None
    ):
        self._baseline = baseline
        self._shap_values = np.array(shap_values)
        self._set_names(names, feature_values)
        self._descriptions = descriptions
        self._baseline_color = baseline_color
        self._positive_color = positive_color
        self._negative_color = negative_color

        self._sort_features(sort_order)
        self._filter_features(max_features)
        self._set_segments()  # start, end and color of each segment
        self._set_data_source()
        self._set_tooltips()
        self._set_x_range(x_range_padding)
        self._x_range_width = x_range_width
        self._set_figure()
        self._add_baseline_span()
        self._add_baseline_label()
        self._add_prediction_span()
        self._add_prediction_label()
        self._add_segments()
        self.add_x_axis_label(x_axis_label)
        self.add_title(title)

    def _set_names(self, names, feature_values):
        if feature_values:
            self._names = ['{} = {}'.format(n, v) for n, v in zip(names, feature_values)]
        else:
            self._names = names

    def _sort_features(self, sort_order):
        assert sort_order in set([
            'absolute_ascending',
            'absolute_descending',
            'ascending',
            'descending'
        ])
        # find sorted index according to sort_order
        if sort_order.startswith('absolute'):
            values = np.abs(self._shap_values)
        else:
            values = self._shap_values
        if sort_order.endswith('ascending'):
            sorted_idxs = values.argsort().tolist()
        else:
            sorted_idxs = (-1 * values).argsort().tolist()
        # update features with sorted index
        self._shap_values = self._shap_values[sorted_idxs]
        self._names = [self._names[idx] for idx in sorted_idxs]
        if self._descriptions:
            self._descriptions = [self._descriptions[idx] for idx in sorted_idxs]

    def _filter_features(self, max_features):
        if max_features and max_features < len(self._shap_values):
            shap_values_keep = self._shap_values[:max_features]
            shap_values_drop = self._shap_values[max_features:]
            shap_values_other = shap_values_drop.sum(keepdims=True)
            self._shap_values = np.concatenate((shap_values_keep, shap_values_other))
            self._names = self._names[:max_features] + ['other_features']
            if self._descriptions:
                self._descriptions = self._descriptions[:max_features] + ['All other features combined.']

    def _set_segments(self):
        self._segment_ends = self._shap_values.cumsum() + self._baseline
        self._segment_starts = self._segment_ends - self._shap_values
        segment_ends_max = self._segment_ends.max()
        segment_ends_min = self._segment_ends.min()
        segment_starts_max = self._segment_starts.max()
        segment_starts_min = self._segment_starts.min()
        self._segments_maximum = max(segment_starts_max, segment_ends_max)
        self._segments_minimum = min(segment_starts_min, segment_ends_min)
        self._segment_colors = []
        for shap_value in self._shap_values:
            if shap_value > 0:
                self._segment_colors.append(self._positive_color)
            else:
                self._segment_colors.append(self._negative_color)

    def _set_data_source(self):
        data = {
            'shap_value': self._shap_values,
            'name': self._names,
            'segment_color': self._segment_colors,
            'segment_start_proba': log_odds_to_proba(self._segment_starts),
            'segment_end_proba': log_odds_to_proba(self._segment_ends)
        }
        if self._descriptions:
            data['descriptions'] = self._descriptions
        self._data_source = bokeh.models.sources.ColumnDataSource(data=data)

    def _set_tooltips(self):
        self._tooltips = [
            ("name", "@name"),
            ("description", "@description")
        ]

    def _set_x_range(self, x_range_padding):
        center = (self._segments_minimum + self._segments_maximum) / 2
        minimum = center - ((center - self._segments_minimum) * (1 + x_range_padding))
        maximum = ((self._segments_maximum - center) * (1 + x_range_padding)) + center
        self._x_range_minimum = log_odds_to_proba(minimum)
        self._x_range_maximum = log_odds_to_proba(maximum)

    def _set_figure(self):
        figure = bokeh.plotting.figure(
            frame_width=self._x_range_width,
            frame_height=25 * len(self._names),
            x_axis_type="log",
            x_axis_location="below",
            y_range=self._names,
            y_axis_location="right",
            tooltips=self._tooltips,
            toolbar_location=None,
            tools=""
        )
        formatter = bokeh.models.formatters.NumeralTickFormatter(format="0.0 %")
        axis_label_text_font_size = "10pt"
        axis_line_color = "white"
        desired_num_ticks = 7
        # set original xaxis
        figure.x_range = bokeh.models.Range1d(
            self._x_range_minimum,
            self._x_range_maximum
        )
        figure.xaxis.formatter = formatter
        figure.xaxis.axis_label_text_font_size = axis_label_text_font_size
        figure.xaxis.axis_line_color = axis_line_color
        figure.xaxis.ticker.desired_num_ticks = desired_num_ticks
        # create a copy of xaxis above
        figure.extra_x_ranges = {"probability": figure.x_range}
        top_axis = bokeh.models.axes.LogAxis(x_range_name="probability")
        top_axis.formatter = formatter
        top_axis.axis_label_text_font_size = axis_label_text_font_size
        top_axis.axis_line_color = axis_line_color
        top_axis.ticker.desired_num_ticks = desired_num_ticks
        figure.add_layout(top_axis, 'above')
        # set yaxis
        figure.yaxis.major_label_text_font_size = axis_label_text_font_size
        figure.yaxis.axis_line_color = axis_line_color
        self._figure = figure

    def _add_baseline_span(self):
        span = bokeh.models.Span(
            location=log_odds_to_proba(self._baseline),
            dimension='height',
            line_color='grey',
            line_dash='dotted',
            line_width=1
        )
        span.level = 'underlay'
        self._figure.add_layout(span)

    def _get_x_range_offset(self, log_odd):
        minimum = proba_to_log_odds(self._x_range_minimum)
        maximum = proba_to_log_odds(self._x_range_maximum)
        x_range_length = maximum - minimum
        percentage = (log_odd - minimum) / x_range_length
        x_range_offset = percentage * self._x_range_width
        return x_range_offset

    def _add_baseline_label(self):
        baseline_proba = log_odds_to_proba(self._baseline)
        text = '{:.2%} baseline'.format(baseline_proba)
        offset = self._get_x_range_offset(self._baseline)
        label = bokeh.models.Title(
            text=text,
            align='left',
            text_font_size='10pt',
            offset=offset,
            text_font_style='italic',
            text_color='grey')
        self._figure.add_layout(label, 'below')

    def _get_prediction(self):
        return self._segment_ends[-1]

    def _get_prediction_color(self):
        prediction = self._get_prediction()
        if prediction > self._baseline:
            return self._positive_color
        else:
            return self._negative_color

    def _add_prediction_span(self):
        prediction = self._get_prediction()
        prediction_proba = log_odds_to_proba(prediction)
        prediction_color = self._get_prediction_color()
        span = bokeh.models.Span(
            location=prediction_proba,
            dimension='height',
            line_color=prediction_color,
            line_dash='dotted',
            line_width=1
        )
        span.level = 'underlay'
        self._figure.add_layout(span)

    def _add_prediction_label(self):
        prediction = self._get_prediction()
        prediction_proba = log_odds_to_proba(prediction)
        text = '{:.2%} predicted'.format(prediction_proba)
        offset = self._get_x_range_offset(prediction)
        prediction_color = self._get_prediction_color()
        label = bokeh.models.Title(
            text=text,
            align='left',
            text_font_size='10pt',
            offset=offset,
            text_font_style='italic',
            text_color=prediction_color
        )
        self._figure.add_layout(label, 'above')

    def _add_segments(self):
        self._figure.segment(
            "segment_start_proba",
            "name",
            "segment_end_proba",
            "name",
            line_width=18,
            line_color="segment_color",
            source=self._data_source
        )
        self._figure.diamond(
            y="name",
            x="segment_end_proba",
            size=16,
            fill_color="segment_color",
            line_color="segment_color",
            source=self._data_source
        )
        self._figure.diamond(
            y="name",
            x="segment_start_proba",
            size=16,
            fill_color='white',
            line_color="white",
            source=self._data_source
        )

    def add_title(self, text):
        if text:
            title = bokeh.models.Title(text=text, align='left', text_font_size='12pt')
            self._figure.add_layout(title, 'above')

    def add_x_axis_label(self, text):
        label = bokeh.models.Title(text=text, align='center', text_font_size='10pt')
        self._figure.add_layout(label, 'above')

    def show(self):
        return bokeh.plotting.show(self._figure)
