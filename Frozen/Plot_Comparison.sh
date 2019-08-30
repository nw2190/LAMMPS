#!/bin/bash
coords=$(python load_soln.py --ID $1)
python2 Convert_Solutions.py
paraview s_disk_0.case &!

./Plot_Prediction.sh $coords

echo " "
echo "Coordinates:"
echo $coords
echo " "
