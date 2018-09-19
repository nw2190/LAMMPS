#!/bin/bash
python load_soln.py --ID $1
python2 Convert_Solutions.py
paraview s_disk_0.case &!
