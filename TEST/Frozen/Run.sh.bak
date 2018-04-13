#!/bin/bash
python load.py --x_val $1 --y_val $2
python2 Convert_Predictions.py
paraview p_disk_0.case &!
