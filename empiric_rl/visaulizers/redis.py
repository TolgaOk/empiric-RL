from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
from itertools import chain
from collections import defaultdict
import pickle
import json
import pandas
import numpy as np
import ipywidgets as widgets
import optuna
import ipysheet
import plotly.graph_objects as go

from empiric_rl.visaulizers.scalar import ScalarRender, MultiScalarRender


class NamedScalarRender(ScalarRender):

    def __init__(self):
        self.x_name = None
        self.y_name = None
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",)
        self.set_components()

    def set_components(self):
        self.select_yaxis = widgets.Dropdown(
            options=[],
            value=None,
            description="Y axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_yaxis.observe(self.set_y_axis)
        self.select_xaxis = widgets.Dropdown(
            options=[],
            value=None,
            description="X axis",
            disabled=False,
            # layout=Layout(width="400px")
        )
        self.select_xaxis.observe(self.set_x_axis)

    @staticmethod
    def _get_column_names(dataframes: List[pandas.DataFrame]):
        if len(dataframes) == 0:
            return set(), set()
        names = tuple(dataframes[0].columns)
        for df in dataframes:
            if sorted(tuple(df.columns)) != sorted(names):
                raise ValueError("Column names do not match")

        intersect_monotonic_names = set(names)
        for df in dataframes:
            monotonic_names = []
            for name in names:
                diff = np.diff(df[name].to_numpy())
                diff = diff[~np.isnan(diff)]
                if np.all(diff >= 0):
                    monotonic_names.append(name)
            monotonic_names = set(monotonic_names)
            intersect_monotonic_names = intersect_monotonic_names & monotonic_names

        return set(names), intersect_monotonic_names

    def fill_dropdowns(self, dataframe: List[pandas.DataFrame]):
        names, intersect_monotonic_names = self._get_column_names(dataframe)
        self.select_yaxis.options = names
        self.select_xaxis.options = intersect_monotonic_names

    def render(self, dataframes: Dict[str, pandas.DataFrame]):
        self.dataframes = dataframes
        self.fill_dropdowns(list(self.dataframes.values()))

    def render_figure(self):
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []
        for name, dataframe in self.dataframes.items():
            self.fig.add_trace(
                go.Scatter(
                    x=dataframe[self.x_name],
                    y=dataframe[self.y_name],
                    mode="lines",
                    name="Trial ID: {}".format(name),
                    line=dict(
                        # color="orange",
                        width=2,
                        shape="spline",
                        smoothing=0.7)
                )
            )
        self.fig.update_layout(
            yaxis={
                "title": self.y_name,
                "gridcolor": "gray",
            },
            xaxis={
                "title": self.x_name,
                "gridcolor": "gray",
            }
        )


class GeneralComparisonRender(MultiScalarRender):

    def __init__(self):
        self.setup()
        self.set_figure()
        self.set_components()
        self.fig.update_layout(showlegend=True)

    def render(self, dataframes: Dict[str, Union[List[pandas.DataFrame], pandas.DataFrame]]):
        self.dataframes = dataframes
        self.fill_dropdowns(
            list(chain(*[item if isinstance(item, (list, tuple)) else [item]
                         for item in self.dataframes.values()]))
        )

    def render_figure(self):
        if self.x_name is None or self.y_name is None:
            return
        self.fig.data = []
        color_size = len(self.default_colors)
        for index, (name, frames) in enumerate(self.dataframes.items()):
            if isinstance(frames, (list, tuple)):
                self._add_margin_trace(frames, self.default_colors[index % color_size], name)
            else:
                self._add_single_trace(frames, self.default_colors[index % color_size], name)

        self.fig.update_layout(
            yaxis={
                "title": self.y_name,
                "gridcolor": "gray",
            },
            xaxis={
                "title": self.x_name,
                "gridcolor": "gray",
            }
        )

    @staticmethod
    def _get_column_names(dataframes):
        if len(dataframes) == 0:
            return set(), set()
        names = set(dataframes[0].columns)
        for df in dataframes:
            names = names & set(df.columns)

        intersect_monotonic_names = set(names)
        for df in dataframes:
            monotonic_names = []
            for name in names:
                diff = np.diff(df[name].to_numpy())
                diff = diff[~np.isnan(diff)]
                if np.all(diff >= 0):
                    monotonic_names.append(name)
            monotonic_names = set(monotonic_names)
            intersect_monotonic_names = intersect_monotonic_names & monotonic_names

        return set(names), intersect_monotonic_names

    def _add_margin_trace(self, dataframes, color, name):
        super()._add_traces(dataframes, color, name)
        pass

    def _add_single_trace(self, dataframes, color, name):
        self.fig.add_trace(
            go.Scatter(
                x=dataframes[self.x_name],
                y=dataframes[self.y_name],
                mode="lines",
                name=name,
                line=dict(
                    color=color,
                    width=2,
                    shape="spline",
                    smoothing=0.7)
            )
        )


@dataclass
class BaseTableRow:
    pass


@dataclass
class TrialRow(BaseTableRow):
    trial_id: int
    number: int
    status: str
    score: float
    trial: optuna.trial.FrozenTrial
    select: widgets.Checkbox = None

    def __call__(self):
        if "meta-data" not in self.trial.user_attrs:
            seed = None
            local_ip = None
        else:
            seed = self.trial.user_attrs["meta-data"]["config"]["seed"]
            local_ip = self.trial.user_attrs["meta-data"]["local_ip_adress"]

        if self.select is None:
            self.select = widgets.Checkbox(
                value=False,
                description=str(self.trial._trial_id),
                disabled=str(self.status) == "TrialState.FAIL",
                indent=False
            )

        return {
            "select": self.select,
            "Trial ID": self.trial_id,
            "Number": self.number,
            "Score": self.score,
            "Status": str(self.status),
            "Seed": seed,
            "Local Ip": local_ip
        }


@dataclass
class StudyRow(BaseTableRow):
    study_name: str
    n_trials: int
    trial_list: List[TrialRow]

    def __call__(self):
        tab = widgets.Tab([
            self._trial_sheet(self.trial_list),
            self.render_group_trials()
        ])
        tab.set_title(0, "All Trials")
        tab.set_title(1, "Grouped Trials")

        return tab

    def group_trials(self):
        param_dict = defaultdict(list)
        for trial in self.trial_list:
            param_key = json.dumps(
                trial.trial.params, sort_keys=True, indent=4)
            param_dict[param_key].append(trial)
        return param_dict

    def render_group_trials(self):
        renders = []
        groups_info = {}
        for param_key, trial_list in self.group_trials().items():

            scalar_plot = NamedScalarRender()
            refresh_button = widgets.Button(
                description="Refresh",
                disabled=False,
                button_style="success",  # "success", "info", "warning", "danger" or ""
                tooltip=param_key,
                icon="sync-alt"  # (FontAwesome names without the `fa-` prefix)
            )
            groups_info[param_key] = {"trial_list": trial_list, "scalar_plot": scalar_plot}

            param_tab = widgets.Tab([
                self._trial_sheet(trial_list),
                widgets.Textarea(
                    value=param_key,
                    placeholder="",
                    description="",
                    disabled=True,
                    layout=widgets.Layout(height="200px")
                ),
                widgets.VBox([
                    refresh_button,
                    scalar_plot()
                ])
            ])

            def refresh_callback(button_info):
                trial_list = groups_info[button_info.tooltip]["trial_list"]
                scalar_plot = groups_info[button_info.tooltip]["scalar_plot"]

                dataframes = {
                    trial_row.trial._trial_id: pandas.DataFrame.from_records(
                        trial_row.trial.user_attrs["progress"])
                    for trial_row in trial_list if (str(trial_row.status) == "TrialState.COMPLETE" or
                                                    str(trial_row.status) == "TrialState.RUNNING")
                }
                scalar_plot.render(dataframes)

            refresh_button.on_click(refresh_callback)

            param_tab.set_title(0, "Table")
            param_tab.set_title(1, "Parameters")
            param_tab.set_title(2, "Plot")

            renders.append(param_tab)
        return widgets.VBox(renders)

    def _trial_sheet(self, trial_list):
        sheet = None
        for row_index, trial in enumerate(trial_list):
            trial_row_info = trial()
            if sheet is None:
                sheet = ipysheet.sheet(
                    rows=len(trial_list),
                    columns=len(trial_row_info),
                    column_headers=list(trial_row_info.keys()))
            for col_index, cell_value in enumerate(trial_row_info.values()):
                ipysheet.cell(row=row_index, column=col_index,
                              value=cell_value)
            ipysheet.row(row_index, list(trial_row_info.values()))
        return sheet


class StudyTableWidget():

    @staticmethod
    def render(studies: List[StudyRow]):
        display = widgets.Accordion([study() for study in studies])
        for index, study in enumerate(studies):
            display.set_title(index, study.study_name)
        return display


class RedisTableWidget():

    def __init__(self):
        self.ip_adress = widgets.Text(
            value="localhost",
            placeholder="Redis Ip Adress",
            description="IP:",
            disabled=False
        )
        self.port = widgets.IntText(
            value=6950,
            description="Port:",
            disabled=False
        )
        self.fetch_button = widgets.Button(
            description="Fetch",
            disabled=False,
            button_style="info",  # "success", "info", "warning", "danger" or ""
            tooltip="",
            icon="fa-sync-alt"  # (FontAwesome names without the `fa-` prefix)
        )
        self.study_widget = widgets.Accordion()
        self.plot_button = widgets.Button(
            description="Plot",
            disabled=False,
            button_style="info",  # "success", "info", "warning", "danger" or ""
            tooltip="",
            icon="fa-sync-alt"  # (FontAwesome names without the `fa-` prefix)
        )
        self.general_plot = GeneralComparisonRender()
        self.fetch_button.on_click(self.fetch_button_callback)
        self.plot_button.on_click(self.plot_button_callback)
        self._studies = None

    def plot_button_callback(self, _):
        if self._studies is None:
            return
        study_frames = {}
        for study in self._studies:
            grouped_frames = defaultdict(list)
            for trial_row in study.trial_list:
                name = "-".join([study.study_name, str(trial_row.trial._trial_id)])
                if trial_row.select is not None:
                    if trial_row.select.value is False:
                        continue
                    if str(trial_row.status) == "TrialState.RUNNING":
                        study_frames[name] = pandas.DataFrame.from_records(
                            trial_row.trial.user_attrs["progress"])
                    elif str(trial_row.status) == "TrialState.COMPLETE":
                        param_key = json.dumps(
                            trial_row.trial.params, sort_keys=True, indent=4)
                        grouped_frames[param_key].append({
                            str(trial_row.trial._trial_id): pandas.DataFrame.from_records(
                                trial_row.trial.user_attrs["progress"]),
                        })
                    else:
                        continue
            for _, frame_dicts in grouped_frames.items():
                trial_ids, frames = list(zip(*[list(frame_dict.items())[0]
                                         for frame_dict in frame_dicts]))
                name = "-".join([study.study_name, *trial_ids])
                study_frames[name] = frames
        self.general_plot.render(study_frames)

    def fetch_button_callback(self, _):
        studies = self.get_studies(
            "redis://{}:{}".format(
                self.ip_adress.value, self.port.value))
        self.study_widget.children = [study() for study in studies]
        self._studies = studies
        for index, study in enumerate(studies):
            self.study_widget.set_title(index, study.study_name)

    @staticmethod
    def get_studies(storage_url: str):
        storage = optuna.storages.RedisStorage(storage_url)

        studies = []
        for sid in RedisTableWidget.get_study_ids(storage):
            trials = storage.get_all_trials(sid)
            studies.append(
                StudyRow(
                    study_name=storage.get_study_name_from_id(sid),
                    n_trials=len(trials),
                    trial_list=[TrialRow(trial_id=trial._trial_id,
                                         number=trial.number,
                                         status=trial.state,
                                         score=None if trial.values is None else trial.values[0],
                                         trial=trial)
                                for trial in trials]
                )
            )
        return studies

    def set_trial_to_fail(self, trial_id: int):
        storage = optuna.storages.RedisStorage(
            "redis://{}:{}".format(
                self.ip_adress.value, self.port.value))
        storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)

    def __call__(self):
        display = widgets.Tab([
            widgets.HBox([self.ip_adress, self.port, self.fetch_button]),
            self.study_widget,
            widgets.VBox([
                self.plot_button,
                self.general_plot(),
            ])
        ])
        display.set_title(0, "Login")
        display.set_title(1, "Trials")
        display.set_title(2, "Comparison Plot")

        return display

    @staticmethod
    def get_study_ids(storage: optuna.storages.RedisStorage):
        return [pickle.loads(sid)
                for sid in storage._redis.lrange("study_list", 0, -1)]
