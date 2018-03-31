#!/bin/bash
qsub -l walltime=00:30:00 -l nodes=1:ppn=20 job0
qsub -l walltime=00:30:00 -l nodes=1:ppn=20 job1
qsub -l walltime=00:30:00 -l nodes=1:ppn=20 job2
qsub -l walltime=00:30:00 -l nodes=1:ppn=20 job3
qsub -l walltime=00:30:00 -l nodes=1:ppn=20 job4
