#!/bin/bash
python load_error.py --ID $1
python2 Convert_Errors.py
paraview e_disk_0.case &!
