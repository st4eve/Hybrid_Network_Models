import json
import os
from abc import abstractmethod

import matplotlib as mpl

mpl.use("TkAgg")
mpl.interactive(True)
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 0.5


class AbstractPlotter:
    def __init__(self, name, x, y, x_label, y_label, xerr=0, yerr=0, title=None):
        self.name = name
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        # Common settings
        self.options = {}
        self.set_option("x_label", x_label)
        self.set_option("y_label", y_label)
        self.set_option("title", title)
        self.set_option("figure_size_x", 6)
        self.set_option("figure_size_y", 6)
        self.set_option("marker", "o")
        self.set_option("linestyle", "solid")
        self.set_option("markersize", 2)
        self.set_option("linewidth", 1.0)

    def plot(self):
        self.setup_plot()
        plt.show()

    @abstractmethod
    def setup_plot(self):
        pass

    def set_option(self, key, value):
        self.options[key] = value

    def get_options(self):
        return self.options

    def close(self):
        plt.close("all")

    def apply_saved_settings(self, path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if data["name"] != self.name:
                raise Exception(
                    "Attempting to apply settings from "
                    + data["name"]
                    + " to "
                    + self.name
                )
            data.pop("name", None)
            for key, val in data.items():
                self.options[key] = val
        except:
            print(
                "You tried to apply settings from "
                + data["name"]
                + " to "
                + self.name
                + ". Please ensure the plot types match."
            )

    def save(self, savename):
        self.setup_plot()
        if not os.path.isdir(savename):
            os.mkdir(savename)
        filename = savename + "/" + savename
        plt.savefig(filename + ".pdf", format="pdf", dpi=1200, bbox_inches="tight")
        with open(filename + ".json", "w") as f:
            self.options["name"] = self.name
            json.dump(self.options, f, indent=4)
            self.options.pop("name", None)


class BasicPlot(AbstractPlotter):
    def __init__(self, x, y, x_label, y_label):
        super().__init__("Basic Plot", x, y, x_label, y_label)

    def setup_plot(self):
        plt.figure(
            figsize=(
                float(self.options["figure_size_x"]),
                float(self.options["figure_size_y"]),
            )
        )
        plt.subplot(111)
        plt.grid(True, linestyle=":")
        plt.xlabel(self.options["x_label"])
        plt.ylabel(self.options["y_label"])
        plt.plot(
            self.x,
            self.y,
            marker=self.options["marker"],
            linestyle=self.options["linestyle"],
            markersize=float(self.options["markersize"]),
            linewidth=float(self.options["linewidth"]),
        )
        if (self.yerr != None):
            plt.fill_between(self.x,self.y - self.yerr,self.y + self.yerr, alpha=0.5)
        plt.tight_layout()


class MultiPlot(AbstractPlotter):
    def __init__(
        self,
        x,
        y,
        x_label,
        y_label,
        legend_name=None,
        legend_values=None,
        legend_position_x=0.2,
        legend_position_y=1.0,
        legend_formatter=None,
        yerr=0,
        title=None
    ):
        super().__init__("Multi Plot", x, y, x_label, y_label, yerr=yerr, title=title)
        self.legend_values = legend_values
        self.set_option("legend_name", legend_name)
        self.set_option("legend_position_x", legend_position_x)
        self.set_option("legend_position_y", legend_position_y)

    def setup_plot(self):
        self.fig = plt.figure(
            figsize=(
                float(self.options["figure_size_x"]),
                float(self.options["figure_size_y"]),
            )
        )
        ax = plt.subplot(111)
        self.ax = ax
        plt.grid(True, linestyle=":")
        plt.xlabel(self.options["x_label"])
        plt.ylabel(self.options["y_label"])
        for i in range(len(self.x)):
            plt.plot(
                self.x[i],
                self.y[i],
                marker=self.options["marker"],
                linestyle=self.options["linestyle"],
                markersize=float(self.options["markersize"]),
                linewidth=float(self.options["linewidth"]),
            )
            if ((self.yerr[i] != 0).any()):
                plt.fill_between(self.x[i],self.y[i] - self.yerr[i], self.y[i] + self.yerr[i], alpha=0.25, label='_nolegend_')
        if (self.legend_values==None):
            legend = self.options['legend_name']
        else:
            if self.options['legend_name'] == 'sigma':
                self.options['legend_name'] = 'Precision' 
                self.legend_values = [np.log2(2/val) for val in self.legend_values]
            if self.options['legend_name'] == 'Precision':
                legend = [f"{self.options['legend_name']}={val:.3f}" for val in self.legend_values]
            else:
                legend = [f"{self.options['legend_name']}={val}" for val in self.legend_values]
        if "trace" in self.options["y_label"] or "Trace" in self.options["y_label"]:
            plt.axhline(y=0.99, color="r", linestyle="-")
            legend.append("Trace Threshold")
        if (self.options['legend_name'] != None):
            plt.legend(
                labels=legend,
                ncol=2,
                loc="lower left",
                bbox_to_anchor=(
                    float(self.options["legend_position_x"]),
                    float(self.options["legend_position_y"]),
                ),
                borderaxespad=0.0,
                frameon=False
            )
        if (self.options['title'] != None):
            plt.title(self.options['title'])

        plt.tight_layout()

