#!/bin/sh
python tf_base.py with network_type=classical_tf num_qumodes=2 &
python tf_base.py with network_type=classical_tf num_qumodes=3 &
python tf_base.py with network_type=classical_tf num_qumodes=4 &
python tf_base.py with network_type=classical_tf num_qumodes=5 &
wait
