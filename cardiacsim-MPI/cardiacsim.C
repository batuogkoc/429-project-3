/*
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * and reimplementation by Scott B. Baden, UCSD
 *
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
using namespace std;

// MPI variables
int mpi_rank;
int mpi_size;
MPI_Status stat;

// Utilities
//

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if (RC == -1)
    {
        cerr << "ERROR: Bad call to gettimeofday" << endl;
        return (-1);
    }

    return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));

} // end getTime()

// Allocate a 2D array
double **alloc2D(int m, int n)
{
    double **E;
    int nx = n, ny = m;
    E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
    assert(E);
    int j;
    for (j = 0; j < ny; j++)
        E[j] = (double *)(E + ny) + j * nx;
    return (E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
double stats(double **E, int m, int n, double *_mx)
{
    double mx = -1;
    double l2norm = 0;
    int i, j;
    for (j = 1; j <= m; j++)
        for (i = 1; i <= n; i++)
        {
            l2norm += E[j][i] * E[j][i];
            if (E[j][i] > mx)
                mx = E[j][i];
        }
    *_mx = mx;
    l2norm /= (double)((m) * (n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

// External functions
extern "C"
{
    void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads);

void simulate(double **E, double **E_prev, double **R,
              const double alpha, const int n, const int m, const double kk,
              const double dt, const double a, const double epsilon,
              const double M1, const double M2, const double b, const int px, const int py, bool gather)
{
    int i, j;

    int array_size_0 = m + 2;
    int array_size_1 = n + 2;

    bool is_nortmost = (mpi_rank == 0);
    bool is_southmost = (mpi_rank == py - 1);

    int south_rank = is_southmost ? -1 : mpi_rank + 1;
    int north_rank = is_nortmost ? -1 : mpi_rank - 1;

    /*
     * Copy data from boundary of the computational box
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */

    for (j = 1; j <= m; j++)
        E_prev[j][0] = E_prev[j][2];
    for (j = 1; j <= m; j++)
        E_prev[j][n + 1] = E_prev[j][n - 1];

    for (i = 1; i <= n; i++)
        E_prev[0][i] = E_prev[2][i];
    for (i = 1; i <= n; i++)
        E_prev[m + 1][i] = E_prev[m - 1][i];

    int regular_computation_size = m / py;
    int last_cell_computation_size = m - (regular_computation_size * (py - 1));
    // cout << "Regular: " << regular_computation_size << " last: " << last_cell_computation_size << " size: " << m << endl;
    MPI_Request requests[4];
    MPI_Status stats[4];
    int comm_wait_count = 0;

    // communicate boundaries
    if (!is_nortmost)
    {
        //  northern communications of ego process
        //  north outgoing
        MPI_Isend(E_prev[mpi_rank * regular_computation_size + 1], array_size_1, MPI_DOUBLE, north_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // north incoming
        MPI_Irecv(E_prev[mpi_rank * regular_computation_size], array_size_1, MPI_DOUBLE, north_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
    }

    if (!is_southmost)
    {
        // southern communications of ego process
        // south outgoing
        MPI_Isend(E_prev[(mpi_rank + 1) * regular_computation_size], array_size_1, MPI_DOUBLE, south_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // south incoming
        MPI_Irecv(E_prev[(mpi_rank + 1) * regular_computation_size + 1], array_size_1, MPI_DOUBLE, south_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // wait for boundary comms
    }
    MPI_Waitall(comm_wait_count, requests, stats);
    cout << "Comms done: " << mpi_rank << endl;

    // if (mpi_rank == 0)
    // {
    //     int curr_row_idx = 0;
    //     // master sends the data to the other cores
    //     for (int pid = 1; pid < py; pid++)
    //     {
    //         if (pid < py - 1)
    //         {
    //             MPI_Send(E_prev[pid * regular_computation_size], (regular_computation_size + 2) * array_size_1, MPI_DOUBLE, pid, 0, MPI_COMM_WORLD);
    //         }
    //         else
    //         {
    //             // last cell
    //             MPI_Send(E_prev[pid * regular_computation_size], (last_cell_computation_size + 2) * array_size_1, MPI_DOUBLE, pid, 0, MPI_COMM_WORLD);
    //         }
    //     }
    // }
    // else
    // {
    //     if (mpi_rank < py - 1)
    //     {
    //         MPI_Recv(E_prev[mpi_rank * regular_computation_size], (regular_computation_size + 2) * array_size_1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
    //     }
    //     else
    //     {
    //         // last cell
    //         MPI_Recv(E_prev[mpi_rank * regular_computation_size], (last_cell_computation_size + 2) * array_size_1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
    //     }
    // }
    // cout << "Comms done: " << mpi_rank << endl;
    int computation_size = 0;
    if (!is_southmost)
    {
        computation_size = regular_computation_size;
    }
    else
    {
        // last cell (southmost)
        computation_size = last_cell_computation_size;
    }
    // cout << "p: " << mpi_rank << " " << mpi_rank * regular_computation_size + 1 << " " << mpi_rank * regular_computation_size + computation_size + 1 << endl;
    // Solve for the excitation, the PDE
    for (j = mpi_rank * regular_computation_size + 1; j <= mpi_rank * regular_computation_size + computation_size; j++)
    {
        for (i = 1; i <= n; i++)
        {
            E[j][i] = E_prev[j][i] + alpha * (E_prev[j][i + 1] + E_prev[j][i - 1] - 4 * E_prev[j][i] + E_prev[j + 1][i] + E_prev[j - 1][i]);
        }
    }

    /*
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    for (j = mpi_rank * regular_computation_size + 1; j <= mpi_rank * regular_computation_size + computation_size; j++)
    {
        for (i = 1; i <= n; i++)
            E[j][i] = E[j][i] - dt * (kk * E[j][i] * (E[j][i] - a) * (E[j][i] - 1) + E[j][i] * R[j][i]);
    }

    for (j = mpi_rank * regular_computation_size + 1; j <= mpi_rank * regular_computation_size + computation_size; j++)
    {
        for (i = 1; i <= n; i++)
            R[j][i] = R[j][i] + dt * (epsilon + M1 * R[j][i] / (E[j][i] + M2)) * (-R[j][i] - kk * E[j][i] * (E[j][i] - b - 1));
    }
    if (gather)
    {
        int counts[py];
        int displacements[py];
        int displacement = 1;
        for (int i = 0; i < py; i++)
        {
            displacements[i] = displacement;
            displacement += regular_computation_size;
            if (i == py - 1)
            {
                counts[i] = last_cell_computation_size;
            }
            else
            {
                counts[i] = regular_computation_size;
            }
        }
        MPI_Gatherv(E[0], computation_size, MPI_DOUBLE, E[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(R[0], computation_size, MPI_DOUBLE, R[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // if (mpi_rank == 0)
    // {
    //     int curr_row_idx = 0;
    //     // master sends the data to the other cores
    //     for (int pid = 1; pid < py; pid++)
    //     {
    //         if (pid < py - 1)
    //         {
    //             MPI_Recv(E[pid * regular_computation_size + 1], (regular_computation_size)*array_size_1, MPI_DOUBLE, pid, 0, MPI_COMM_WORLD, &stat);
    //         }
    //         else
    //         {
    //             // last cell
    //             MPI_Recv(E[pid * regular_computation_size + 1], (last_cell_computation_size)*array_size_1, MPI_DOUBLE, pid, 0, MPI_COMM_WORLD, &stat);
    //         }
    //     }
    // }
    // else
    // {
    //     if (mpi_rank < py - 1)
    //     {
    //         MPI_Send(E[mpi_rank * regular_computation_size + 1], (regular_computation_size)*array_size_1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    //     }
    //     else
    //     {
    //         // last cell
    //         MPI_Send(E[mpi_rank * regular_computation_size + 1], (last_cell_computation_size)*array_size_1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    //     }
    // }
}

// Main program
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    cout << "Rank       : " << mpi_rank << endl;
    cout << "Size       : " << mpi_size << endl;
    /*
     *  Solution arrays
     *   E is the "Excitation" variable, a voltage
     *   R is the "Recovery" variable
     *   E_prev is the Excitation variable for the previous timestep,
     *      and is used in time integration
     */
    double **E, **R, **E_prev;

    // Various constants - these definitions shouldn't change
    const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

    double T = 1000.0;
    int m = 200, n = 200;
    int plot_freq = 0;
    int px = 1, py = 1;
    int no_comm = 0;
    int num_threads = 1;

    cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
    m = n;
    // Allocate contiguous memory for solution arrays
    // The computational box is defined on [1:m+1,1:n+1]
    // We pad the arrays in order to facilitate differencing on the
    // boundaries of the computation box
    E = alloc2D(m + 2, n + 2);
    E_prev = alloc2D(m + 2, n + 2);
    R = alloc2D(m + 2, n + 2);

    int i, j;
    // Initialization
    for (j = 1; j <= m; j++)
        for (i = 1; i <= n; i++)
            E_prev[j][i] = R[j][i] = 0;

    for (j = 1; j <= m; j++)
        for (i = n / 2 + 1; i <= n; i++)
            E_prev[j][i] = 1.0;

    for (j = m / 2 + 1; j <= m; j++)
        for (i = 1; i <= n; i++)
            R[j][i] = 1.0;

    double dx = 1.0 / n;

    // For time integration, these values shouldn't change
    double rp = kk * (b + 1) * (b + 1) / 4;
    double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
    double dtr = 1 / (epsilon + ((M1 / M2) * rp));
    double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
    double alpha = d * dt / (dx * dx);

    cout << "Grid Size       : " << n << endl;
    cout << "Duration of Sim : " << T << endl;
    cout << "Time step dt    : " << dt << endl;
    cout << "Process geometry: " << px << " x " << py << endl;
    if (no_comm)
        cout << "Communication   : DISABLED" << endl;

    cout << endl;

    // Start the timer
    double t0 = getTime();

    // Simulated time is different from the integer timestep number
    // Simulated time
    double t = 0.0;
    // Integer timestep number
    int niter = 0;

    while (t < T)
    {

        t += dt;
        niter++;
        bool gather = false;

        // will plot this iter, gather
        if (plot_freq)
        {
            int k = (int)(t / plot_freq);
            if ((t - k * plot_freq) < dt)
            {
                gather = true;
            }
        }
        // last iter, gather
        if (!((t + dt) < T))
        {
            gather = true;
        }

        simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, px, py, gather);

        // swap current E with previous E
        double **tmp = E;
        E = E_prev;
        E_prev = tmp;

        if (plot_freq)
        {
            int k = (int)(t / plot_freq);
            if ((t - k * plot_freq) < dt)
            {
                splot(E, t, niter, m + 2, n + 2);
            }
        }
    } // end of while loop

    double time_elapsed = getTime() - t0;

    double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
    double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

    cout << "Number of Iterations        : " << niter << endl;
    cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
    cout << "Sustained Gflops Rate       : " << Gflops << endl;
    cout << "Sustained Bandwidth (GB/sec): " << BW << endl
         << endl;

    double mx;
    double l2norm = stats(E_prev, m, n, &mx);
    cout << "Max: " << mx << " L2norm: " << l2norm << endl;

    if (plot_freq)
    {
        cout << "\n\nEnter any input to close the program and the plot..." << endl;
        getchar();
    }

    free(E);
    free(E_prev);
    free(R);

    MPI_Finalize();
    return 0;
}
