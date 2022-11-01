from basic_plotter_cli import cli_plot
from CLI import *
from generate_database import ResultsDatabaseGenerator
from Plot import *

if __name__ == "__main__":

    data_blob = ResultsDatabaseGenerator()
    data_blob.initialize_from_folder(
        experiment_folder="Experiment_Data/cutoff_dimension/Experiment_cutoff_hybrid_with_trace",
        verify=True,
    )
    trace_plot = BasicPlot(
        data_blob.data[1]["metrics"]["epoch"],
        data_blob.data[1]["metrics"]["traces_average"],
        "Epochs",
        "Trace",
    )
    cli_plot(trace_plot)
