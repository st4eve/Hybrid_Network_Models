"""CV Hybrid Networks 2022"""
from sklearn.model_selection import ParameterGrid

NUM_PARALLEL_JOBS = 8

PARAMETERS = {
    "precision": [2**n for n in range(10)],
}

EXPERIMENT_NAME = "hybrid_pwb.py"


def main():
    """Main function to run"""
    with open("run.sh", "w", encoding="utf-8") as write_file:
        write_file.write("#!/bin/sh\n")
        for val, parameter_combo in enumerate(list(ParameterGrid(PARAMETERS))):
            param_str = " ".join([f"{key}={val}" for key, val in parameter_combo.items()])
            write_str = f"python {EXPERIMENT_NAME} with {param_str} &\n"
            write_file.write(write_str)
            if (val + 1) % NUM_PARALLEL_JOBS == 0:
                write_file.write("wait\n")


if __name__ == "__main__":
    main()
