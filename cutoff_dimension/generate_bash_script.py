"""CV Hybrid Networks 2022"""
from sklearn.model_selection import ParameterGrid

PARAMETERS = {
    "quantum_preparation_layer": [True, False],
    "regularizer_string": ["L1=0.1", "L2=0.1", "None"],
    "scale_max": [1, 3, 6, 9],
}


def main():
    """Main function to run"""
    with open("run.sh", "w", encoding="utf-8") as write_file:
        write_file.write("#!/bin/sh\n")
        for parameter_combo in list(ParameterGrid(PARAMETERS)):
            param_str = " ".join(
                [f"{key}={val}" for key, val in parameter_combo.items()]
            )
            write_str = f"python quantum.py with {param_str} &\n"
            write_file.write(write_str)


if __name__ == "__main__":
    main()
