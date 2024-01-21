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
#SBATCH --output=run-prototype.out
#SBATCH --qos=shorter
#SBATCH --mem-per-cpu=1G

module load openmpi/4.0.1
export PATH=/kuacc/apps/openmpi/4.0.1/bin/:$PATH
export LD_LIBRARY_PATH=/kuacc/apps/openmpi/4.0.1/lib/:$LD_LIBRARY_PATH
export PATH=/kuacc/apps/openmpi/4.0.1/include:$PATH

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################


echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."


DIR="study1"
mkdir -p $DIR
./cardiacsim_serial -n 1024 -t 100 | tee "${DIR}/cardiacsim-n1024-t100-serial.txt"
for NUM_PROC in 1 2 4 8 16
do
    mpirun -np ${NUM_PROC} ./cardiacsim_parallel_1 -n 1024 -t 100 -y ${NUM_PROC} -x 1 | tee "${DIR}/cardiacsim-n1024-t100-y${NUM_PROC}-x1.txt"
done

DIR="study2"
mkdir -p $DIR
for NUM_PROC in 1 2 4 8 16
do
    mpirun -np ${NUM_PROC} ./cardiacsim_parallel_3 -n 1024 -t 100 -y ${NUM_PROC} -x ${NUM_PROC} | tee "${DIR}/cardiacsim-n1024-t100-y${NUM_PROC}-x${NUM_PROC}.txt"
done

DIR="study3"
mkdir -p $DIR
for Y_GEOMETRY in 1 2 4 8 16
do
    X_GEOMETRY=$((16 / Y_GEOMETRY))
    for PROBLEM_SIZE in 1024 512 256 128 64
    do
        mpirun -np 16 ./cardiacsim_parallel_3 -n ${PROBLEM_SIZE} -t 100 -y ${Y_GEOMETRY} -x ${X_GEOMETRY} -k | tee "${DIR}/cardiacsim-n${PROBLEM_SIZE}-t100-y${Y_GEOMETRY}-x${X_GEOMETRY}.txt"
    done
done

DIR="study4"
mkdir -p $DIR
for NUM_MPI_PROC in 8 4 2 1
do
    NUM_OMP_THREADS=$((16 / NUM_MPI_PROC))
    export OMP_NUM_THREADS=$NUM_OMP_THREADS
    export SRUN_CPUS_PER_TASK=$NUM_OMP_THREADS
    mpirun -np $NUM_MPI_PROC -cpus-per-proc $NUM_OMP_THREADS ./cardiacsim_parallel_2 -n 1024 -t 100 -y $NUM_MPI_PROC -x 1 | tee "${DIR}/cardiacsim-n1024-t100-mpi${NUM_MPI_PROC}-omp${NUM_OMP_THREADS}.txt"
done
