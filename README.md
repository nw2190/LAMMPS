# LAMMPS
TensorFlow code trained on LAMMPS peridynamics simulations


## Generating Data

```    
# Generate Data with LAMMPS:
$ ./Run_LAMMPS.sh
OR
$ OMP_NUM_THREADS=8 lmp -in in.peri
OR
$ lammps -in in.peri

# Convert LAMMPS data to array format:
$ python Convert_Data.py
```

### Parallelization

The data generation process can be sub-divided into separate LAMMPS instances by breaking up the `i` loop in the `in.peri` file.  For example, the file can be split into three files `in_0.peri`, `in_1.peri`, `in_2.peri` corresponding to a partion of the full `i` loop and executed separately by running:
    
```
$ ./Run_0_LAMMPS.sh
$ ./Run_1_LAMMPS.sh
$ ./Run_2_LAMMPS.sh        
```

where the starting count `cstart` is specified in terms of the `istart` and `imax` values.  These starting counts can be pre-computed using the `PERI_TEST/plot.py` file.

    
The array files can then be created using the Python multiprocessing module via:

```
# Convert dump files into NumPy arrays:
$ python Make_Arrays.py
```

The generated training data is created in the `./Data/` and `./Arrays/` subdirectories.  Once the arrays have been created, the large `./Data/dump_*.peri` files can be deleted; only the `./Data/indenter_*.txt` files are required for training.
     
    
## Train Model

```
# Train TensorFlow model:
$ python Train_Model.py
```


## View Results    

```
# Freeze TensorFlow model:
$ python freeze.py

# Test model at specific point (x,y):
$ cd Frozen
$ python load.py --x_val x --y_val y

# Convert prediction to dump file:
$ python Convert_Predictions.py

# View prediction in Paraview:
$ paraview p_disk_0.case &!

# Plot true solution with id ID:
$ python load_soln.py --ID ID
$ python Convert_Solutions.py
$ paraview s_disk_0.case &!
```    