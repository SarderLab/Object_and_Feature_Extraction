#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000mb
#SBATCH --partition=hpg-default
#SBATCH --time=24:00:00
#SBATCH --output=extraction.out
#SBATCH --job-name="Feature Extraction"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load matlab
module list

echo "Launch job"
matlab -nodisplay "Renal_quantification_master.m"
echo "All Done!"
