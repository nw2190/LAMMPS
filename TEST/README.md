LAMMPS/TensorFlow Learning Framework

    Generate Data with LAMMPS:
    $ lammps -in in.peri

    Convert LAMMPS data to array format:
    $ python Convert_Data.py

    Train TensorFlow model:
    $ python Train_Model.py

    Freeze TensorFlow model:
    $ python freeze.py

    Test model at specific point (x,y):
    $ cd Frozen
    $ python load.py --x_val x --y_val y

    Convert prediction to dump file:
    $ python Convert_Predictions.py

    View prediction in Paraview:
    $ paraview p_disk_0.case &!

    Plot true solution with id ID:
    $ python load_soln.py --ID ID
    $ python Convert_Solutions.py
    $ paraview s_disk_0.case &!
    