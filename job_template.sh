#SBATCH --account=def-sabeda
#SBATCH --time=0-08:00        # time (DD-HH:MM) |TODO to change
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32    # number of OpenMP processes
#SBATCH --mem=128G #TODO : to change
#SBATCH --mail-user=sandrine.bedard@polymtl.ca
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $SCRATCH