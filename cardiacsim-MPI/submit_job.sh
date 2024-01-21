#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=cardiacsim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cores-per-socket=16
#SBATCH --partition=shorter
#SBATCH --time=00:30:00
#SBATCH --output=out.txt
#SBATCH --qos=shorter
#SBATCH --mem-per-cpu=1G
module load openmpi/4.0.1
export PATH=/kuacc/apps/openmpi/4.0.1/bin/:$PATH
export LD_LIBRARY_PATH=/kuacc/apps/openmpi/4.0.1/lib/:$LD_LIBRARY_PATH
export PATH=/kuacc/apps/openmpi/4.0.1/include:$PATH


################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

echo "Compiling code"
make
echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."
# lscpu

# mpiexec -n 1 ./a.out

#parallel version test
# echo "Parallel version test"
# mpirun -np 3 ./cardiacsim_parallel_1 -n 1024 -t 100 -y 3

# serial version
# echo "Serial version..."
# ./cardiacsim_serial -n 1024 -t 100

# parallel version
# echo "Parallel version with 1 process"
# mpirun -np 1 ./cardiacsim_parallel_1 -n 1024 -t 100 -y 1

# echo "Parallel version with 2 processes"
# mpirun -np 2 ./cardiacsim_parallel_1 -n 1024 -t 100 -y 2 -x 1

# echo "Parallel version with 4 processes"
# mpirun -np 4 ./cardiacsim_parallel_1 -n 1024 -t 100 -y 4 -x 1

# #Different configuration of MPI+OpenMP
# #[1 + 16] [2 + 8] [4 + 4] [8 + 2] [ 1 + 16]

# export KMP_AFFINITY=verbose,compact

echo "MPI1 + OMP16"
export OMP_NUM_THREADS=16
mpirun -np 1 ./cardiacsim_parallel_2 -o 16

# echo "MPI2 + OMP8"
# export OMP_NUM_THREADS=8
# export SRUN_CPUS_PER_TASK=8
# mpirun -np 2 ./cardiacsim_parallel_2 -o 8

# echo "MPI1 + OMP16"
# export OMP_NUM_THREADS=16
# mpirun -np 1 ./cardiacsim [program arguments]

# echo "MPI2 + OMP8"
# export OMP_NUM_THREADS=8
# export SRUN_CPUS_PER_TASK=8
# mpirun -np 2 -cpus-per-proc 8 ./cardiacsim [program arguments]





