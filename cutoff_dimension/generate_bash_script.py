"""CV Hybrid Networks 2022"""
from sklearn.model_selection import ParameterGrid

NUM_PARALLEL_JOBS = 4

PARAMETERS = {
    "quantum_preparation_layer": [True, False],
    "regularizer_string": ["L1=0.1", "L2=0.1", "L1=0.01", "L2=0.01", "None"],
    "scale_max": [1, 3, 6, 9],
    "iteration": [i for i in range(2)]
}


def main():
    """Main function to run"""
    with open("run.sh", "w", encoding="utf-8") as write_file:
        write_file.write("#!/bin/sh\n")
        for val, parameter_combo in enumerate(list(ParameterGrid(PARAMETERS))):
            param_str = " ".join(
                [f"{key}={val}" for key, val in parameter_combo.items()]
            )
            write_str = f"python quantum_kerr.py with {param_str} &\n"
            write_file.write(write_str)
            if (val + 1) % NUM_PARALLEL_JOBS == 0:
                write_file.write("wait\n")


if __name__ == "__main__":
    main()
