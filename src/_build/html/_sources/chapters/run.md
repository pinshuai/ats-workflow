# Running ATS model

This section provides the scripts used for executing ATS models on both local PC and high-performance computing (e.g., NERSC)

## Single Job on Local PC

```bash
# using single core
ats --xml_file=input.xml

# using multiple cores
mpirun -n 4 ats --xml_file=input.xml
```

## Single Job on High-performance Computing (HPC)
A shell script is usually used for submitting jobs on HPC. Here is a sample job script for running ATS on Cori NERSC.

```{note}
Depending on the HPC system and node architectures, the job script may be slightly different. Refer to the machine's documentation (e.g., [NERSC](https://docs.nersc.gov/jobs/examples/))
```

```bash
#!/bin/bash -l

#SBATCH -A PROJECT_REPO
#SBATCH -N 2
#SBATCH -t 14:00:00
#SBATCH -L SCRATCH
#SBATCH -J JOB_NAME
#SBATCH --qos regular
#SBATCH -C haswell

cd $SLURM_SUBMIT_DIR

srun -n 64 ats --xml_file=./input.xml
```

## Batch Job

This section provides scripts for launching batch job on HPC. This is useful for ensemble runs.

### Using `for loop`

Launch job using a sbatch script. Basically write a for loop and submit entire jobs using `srun`. The example script below submits 40 small jobs through a single submission using a `for` loop. It requests a total of 160 nodes (each job uses 4 nodes) and a time limit of 22 hours.

```{admonition} Important
Use `&` between each `srun` execution, and `wait` after the entire `for loop`. 
```

```{note}
- Add `sleep` to provide additional time between each job submission. 

- `#SBATCH --no-kill` is an option to prevent batch job failure if one of the nodes it has been allocated fails. The user will assume the responsibilities for fault-tolerance should a node fail (Don't use this if you are not sure). See the [Slurm documentation](https://slurm.schedmd.com/sbatch.html#OPT_no-kill) for more information.

```

```bash
#!/bin/bash -l

#SBATCH -A PROJECT_REPO
#SBATCH -N 160
#SBATCH -t 22:00:00
#SBATCH -L SCRATCH
#SBATCH -J JOB_NAME
#SBATCH --qos regular
#SBATCH --mail-type ALL
#SBATCH --mail-user pin.shuai@pnnl.gov
#SBATCH -C haswell
#SBATCH --no-kill

module use -a /global/project/projectdirs/m3421/ats-new/modulefiles
module load ats/ecoon-land_cover/cori-haswell/intel-6.0.5-mpich-7.7.10/opt

for i in {1..40}
do
    cd /path/to/batch_job/ens$i
    srun -N 4 -n 128 -e job%J.err -o job%J.out ats --xml_file=input.xml sleep 5 &
done

sleep 1

wait
```

**Pros**:
- Simple workflow
- Finish the ensemble jobs all at once.

**Cons**:
- Long queue time at the beginning
- May not work well for relatively large ensemble runs (e.g., >200) because of the potential node failure.

### Using `job Arrays`

Job array provides a convenient way to schedule similar jobs quickly and easily. The example script below submits 30 jobs. Each job uses 1 node and has the same time limit and QOS. See [Slurm documentation](https://slurm.schedmd.com/job_array.html) on job arrays.

```bash
#!/bin/bash -l

#SBATCH -A PROJECT_REPO
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -L SCRATCH
#SBATCH --ntasks-per-node 32
#SBATCH --array=1-30
#SBATCH -J JOB_NAME
#SBATCH --qos regular
#SBATCH --mail-type ALL
#SBATCH --mail-user pin.shuai@pnnl.gov

#SBATCH --output=array_%A_%a.out

module use -a /global/project/projectdirs/m3421/ats-new/modulefiles
module load ats/ecoon-land_cover/cori-haswell/intel-6.0.5-mpich-7.7.10/opt

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

inode=${SLURM_ARRAY_TASK_ID}

cd /path/to/batch_job/ens$inode
srun -n 32 ats --xml_file=input.xml
```

**Pros**:
- Submit jobs quickly and easily.
- The queue time maybe short if running smaller jobs (e.g., short time limit).

**Cons**:
- Long queue time overall if running large ensembles because Slurm scheduler prioritizes fewer jobs requesting many nodes ahead of many jobs requesting fewer nodes (array tasks are considered individual jobs).

### Using `MATK`

Use python scripts to combine input file generation, job submission, sensitivity analysis, and model evaluation through Model Analysis ToolKit (`MATK`). MATK facilitates model analysis within the Python computational environment. See [MATK documentation](http://dharp.github.io/matk/) for more information.

```bash
#!/bin/bash -l

#SBATCH --account=PROJECT_REPO
#SBATCH -N 66
#SBATCH --tasks-per-node=32
#SBATCH -t 06:00:00
#SBATCH --job-name=sensitivity_study
#SBATCH --qos regular
#SBATCH --mail-type ALL
#SBATCH --mail-user pin.shuai@pnnl.gov
#SBATCH -C haswell
#SBATCH -o ./sensitivity_study.out
#SBATCH -e ./sensitivity_study.err

module use -a /global/project/projectdirs/m3421/ats-new/modulefiles
module load python
module load matk
module load ats/ecoon-land_cover/cori-haswell/intel-6.0.5-mpich-7.7.10/opt

python run_sensitivity.py

```

```{note}
`run_sensitivity.py` is a python script that user uses to generate model parameters for the ensemble, schedule the forward runs, and post-processing of model results.
```

**Pros**:
- Manage jobs more efficiently (e.g., keep track of finished and unfinished jobs for resubmission)
- Can leverage more functionality from `MATK` (e.g., model calibration using PEST)

**Cons**:
- Need to istall `MATK` and learn how to use it
- The total queue time may be longer than a big `for loop` submission.



```{note}
For more workflow tools, see [NERSC documentation](https://docs.nersc.gov/jobs/workflow-tools/).
```