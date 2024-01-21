#include "omp.h"
#include "mpi.h"
#include "stdio.h"

int main(int argc, char const *argv[])
{
    MPI_Init(&argc, &argv);
#pragma omp parallel num_threads(2)
    {
        printf("thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    MPI_Finalize();
    return 0;
}
