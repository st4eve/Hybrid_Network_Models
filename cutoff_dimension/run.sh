#!/bin/sh
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.01 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.01 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.01 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L1=0.01 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.01 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.01 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.01 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=L2=0.01 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=None scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=None scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=None scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=True regularizer_string=None scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.01 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.01 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.01 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L1=0.01 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.01 scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.01 scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.01 scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=L2=0.01 scale_max=9 &
wait
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=None scale_max=1 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=None scale_max=3 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=None scale_max=6 &
python quantum_kerr.py with iteration=0 quantum_preparation_layer=False regularizer_string=None scale_max=9 &
wait
