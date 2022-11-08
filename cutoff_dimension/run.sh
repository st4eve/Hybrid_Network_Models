#!/bin/sh
python quantum.py with quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=1 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=3 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=6 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L1=0.1 scale_max=9 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=1 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=3 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=6 &
python quantum.py with quantum_preparation_layer=True regularizer_string=L2=0.1 scale_max=9 &
wait
python quantum.py with quantum_preparation_layer=True regularizer_string=None scale_max=1 &
python quantum.py with quantum_preparation_layer=True regularizer_string=None scale_max=3 &
python quantum.py with quantum_preparation_layer=True regularizer_string=None scale_max=6 &
python quantum.py with quantum_preparation_layer=True regularizer_string=None scale_max=9 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=1 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=3 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=6 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L1=0.1 scale_max=9 &
wait
python quantum.py with quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=1 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=3 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=6 &
python quantum.py with quantum_preparation_layer=False regularizer_string=L2=0.1 scale_max=9 &
python quantum.py with quantum_preparation_layer=False regularizer_string=None scale_max=1 &
python quantum.py with quantum_preparation_layer=False regularizer_string=None scale_max=3 &
python quantum.py with quantum_preparation_layer=False regularizer_string=None scale_max=6 &
python quantum.py with quantum_preparation_layer=False regularizer_string=None scale_max=9 &
wait
