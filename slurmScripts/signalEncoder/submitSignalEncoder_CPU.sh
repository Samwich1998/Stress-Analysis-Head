#!/bin/bash

# Specify the module parameters
#SBATCH --time=7-00:00:00       # Total runtime. Maximum 7 days = 128 hours
#SBATCH --partition=expansion   # Specify the expansion partition (May change)
#SBATCH --ntasks=1     # number of processor cores (i.e. tasks)
#SBATCH --mem=256G
#SBATCH --nodes=1      # Total number of nodes

# Specify the job information
#SBATCH --mail-user=ssolomon@caltech.edu    # Email any results or errors

# Specify dynamic paths for output and error files
#SBATCH --output="slurmOutputs/%x-%j.out"  # Standard output file

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN,END,FAIL

# Load in your modules
module load intel-oneapi-mkl/2023.2.0-gcc-13.2.0-ohvyk7g
module load python/3.10.12-gcc-13.2.0-d2ofisd   # Load in the latest python version
module load openmpi/4.1.5-gcc-13.2.0-24q3ap2   # Load in openMPI for cross-node talk

# RUN FILE
srun signalencoderGroupAnalysis.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7"
