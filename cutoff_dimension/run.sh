#!/bin/sh
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.05 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.15 scale_max=1 &
