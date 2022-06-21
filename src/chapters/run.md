# Run ATS model

This section provides the scripts used for executing ATS models on both local PC and high-performance computing (e.g., NERSC)

## Local PC

```bash
# using single core
ats --xml_file=input.xml

# using multiple cores
mpirun -n 4 ats --xml_file=input.xml
```

## High-performance computing
A shell script is usually used for submitting jobs on HPC. Here is a sample job script for running ATS on Cori NERSC.

```{note}
Depending on the HPC system and node architectures, the job script may be slightly different. Refer to the machine's documentation (e.g., [NERSC](https://docs.nersc.gov/jobs/examples/))
```

```bash
#!/bin/bash -l

#SBATCH -A PROJECT
#SBATCH -N 2
#SBATCH -t 14:00:00
#SBATCH -L SCRATCH
#SBATCH -J JOB_NAME
#SBATCH --qos regular
#SBATCH -C haswell

cd $SLURM_SUBMIT_DIR

srun -n 64 ats --xml_file=./input.xml
```
