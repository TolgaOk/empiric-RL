from typing import Optional
from itertools import chain
from collections import defaultdict
import numpy as np
from scipy import stats
import optuna
import ipywidgets
import plotly.graph_objects as go


def load_study(storage_path: str, study_name: Optional[str] = None):
    study_name = study_name or storage_path.split("/")[-2][5:]
    return optuna.load_study(study_name, storage=storage_path)


def plot_trials(study: optuna.Study):
    colors = []
    sizes = []
    max_trial_numbers = []
    max_value = -np.inf
    for trial in study.trials:
        if max_value < trial.values[0]:
            max_value = trial.values[0]
            max_trial_numbers.append(trial.number)
            sizes.append(13)
            colors.append("#EF553B")
        else:
            sizes.append(11)
            colors.append("#636EFA")

    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(
        x=[trial.number for trial in study.trials],
        y=[trial.values[0] for trial in study.trials],
        marker_line_width=2,
        mode="markers",
        marker=dict(size=sizes,
                    color=colors
                    ),
        hovertemplate="%{text}",
        showlegend=False,
        text=[make_hover_text(trial) for trial in study.trials],
    ))
    fig.add_trace(go.Scatter(
        x=max_trial_numbers,
        y=[study.trials[index].values[0] for index in max_trial_numbers],
        mode="lines",
        line_color="#EF553B",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text="Trials",
            xanchor="center",
            x=0.5),
        yaxis=dict(
            title="Score"
        ),
        xaxis=dict(
            title="Trial Number"
        ),
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#282b38"
        ),
        margin=dict(
            l=20,
            r=20,
            b=10,
            t=90,
            pad=20
        ),
        width=650,
        height=500,
    )
    return fig

def make_hover_text(trial):
    return "<br>".join(["<b>{}</b>: {}".format(key, value)
                        for key, value in chain(
                            [("Trial Number", trial.number), ("score", trial.values[0])],
                            trial.params.items())])


def marginal_parameter_score(study: optuna.Study):
    return ipywidgets.GridBox(
        [plot_stat(study.trials, name) for name in study.trials[0].params.keys()],
        layout=ipywidgets.Layout(grid_template_columns="repeat(3, 450px)"))


def plot_stat(trials, name, bins=6):
    dist = trials[0].distributions[name]
    sample = [(trial.params[name], trial.values[0])
              for trial in trials]
    dist_map = dict(
        UniformDistribution=lambda: uniform(
            sample, dist.low, dist.high, name=name, bins=bins, use_log=False),
        IntUniformDistribution=lambda: uniform(
            sample, dist.low, dist.high, name=name, bins=bins, use_log=False),
        LogUniformDistribution=lambda: uniform(
            sample, dist.low, dist.high, name=name, bins=bins, use_log=True),
        CategoricalDistribution=lambda: categorical(
            sample, dist.choices, name=name),
    )
    fig = dist_map[trials[0].distributions[name].__class__.__name__]()
    fig.update_layout(
        title=dict(
            text=name,
            xanchor="center",
            x=0.5),
        margin=dict(
            l=20,
            r=20,
            b=10,
            t=50,
            pad=20
        ),
        width=450,
        height=400,
        font=dict(
            #             family="Courier New, monospace",
            size=12,
            color="#282b38"
        ),
    )
    return fig


def safe_fn(fn, values):
    if len(values) == 0:
        return 0
    return fn(values)


def half_interval(values):
    size = len(values)
    return stats.t.ppf(1-0.025, size) * np.std(values) / np.sqrt(size)


def categorical(samples, choices, name):
    stat = {choice: [] for choice in choices}
    for sample, value in samples:
        stat[sample].append(value)
    fig = go.FigureWidget()
    fig.add_trace(go.Bar(
        x=choices,
        y=[safe_fn(np.mean, stat[choice]) for choice in choices],
        error_y={"type": "data", "array": [
            safe_fn(half_interval, stat[choice]) for choice in choices]}
    ))
    fig.update_layout(
        title=name,
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        width=450
    )
    return fig


def uniform(sample, low, high, name, bins=10, use_log=False):

    if use_log:
        low = np.log10(low)
        high = np.log10(high)
        sample = [(np.log10(item[0]), item[1]) for item in sample]

    mean = stats.binned_statistic(
        *map(np.array, list(zip(*sample))),
        statistic="mean",
        bins=bins,
        range=(low, high))
    t_interval = stats.binned_statistic(
        *map(np.array, list(zip(*sample))),
        statistic=lambda seq: safe_fn(half_interval, seq),
        bins=bins,
        range=(low, high))
    edges = mean.bin_edges
    x_axis = (edges[1:] + edges[:-1])/2

    fig = go.FigureWidget()
    fig.add_trace(go.Bar(
        x=x_axis,
        y=mean.statistic,
        error_y={"type": "data", "array": t_interval.statistic}
    ))
    fig.update_layout(
        title=name,
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        width=450
    )

    if use_log:
        fig.update_xaxes(
            ticktext=["{:.6f}".format(10**val) for val in x_axis],
            tickvals=x_axis,
        )

    return fig
