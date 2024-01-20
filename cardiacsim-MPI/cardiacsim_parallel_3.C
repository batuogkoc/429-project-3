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

void memcpy2d(double **src_array, double **dest_array, int src_x, int src_y, int dest_x, int dest_y, int size_x, int size_y)
{
    for (int row_idx = 0; row_idx < size_y; row_idx++)
    {
        memcpy(dest_array[dest_y + row_idx] + dest_x, src_array[src_y + row_idx] + src_x, sizeof(double) * size_x);
    }
}

void simulate(
    double **E, double **E_gather, double **E_gather_local,
    double **E_local, double **E_prev_local, double **R_local,
    const int local_array_in_global_idx_x, const int local_array_in_global_idx_y,
    const int p_x_idx, const int p_y_idx,
    const int regular_computation_size_x, const int regular_computation_size_y,
    const int last_cell_computation_size_x, const int last_cell_computation_size_y,
    const int computation_size_x, const int computation_size_y,
    const double alpha, const int n, const int m, const double kk,
    const double dt, const double a, const double epsilon,
    const double M1, const double M2, const double b, const int px, const int py, bool gather)
{
    int i, j;

    int array_size_0 = m + 2;
    int array_size_1 = n + 2;

    bool is_northmost = (p_y_idx == 0);
    bool is_southmost = (p_y_idx == py - 1);

    bool is_westmost = (p_x_idx == 0);
    bool is_eastmost = (p_x_idx == px - 1);

    int south_rank = is_southmost ? -1 : mpi_rank + px;
    int north_rank = is_northmost ? -1 : mpi_rank - px;

    int east_rank = is_eastmost ? -1 : mpi_rank + 1;
    int west_rank = is_westmost ? -1 : mpi_rank - 1;

    /*
     * Copy data from boundary of the computational box
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
    if (is_westmost)
    {
        for (j = 1; j <= computation_size_y; j++)
            E_prev_local[j][0] = E_prev_local[j][2];
    }

    if (is_eastmost)
    {
        // east wall
        for (j = 1; j <= computation_size_y; j++)
            E_prev_local[j][n + 1] = E_prev_local[j][n - 1];
    }

    if (is_northmost)
    {
        // north wall
        for (i = 1; i <= computation_size_x; i++)
            E_prev_local[0][i] = E_prev_local[2][i];
    }

    if (is_southmost)
    {
        // south wall
        for (i = 1; i <= n; i++)
            E_prev_local[m + 1][i] = E_prev_local[m - 1][i];
    }

    // cout << "Regular: " << regular_computation_size << " last: " << last_cell_computation_size << " size: " << m << endl;
    MPI_Request requests[4];
    MPI_Status stats[4];
    int comm_wait_count = 0;

    double west_incoming[computation_size_y];
    double east_incoming[computation_size_y];

    double west_outgoing[computation_size_y];
    double east_outgoing[computation_size_y];

    // pack west and east outgoing
    for (size_t row_idx = 0; row_idx < computation_size_y; row_idx++)
    {
        west_outgoing[row_idx] = E_prev_local[row_idx + 1][1];
        east_outgoing[row_idx] = E_prev_local[row_idx + 1][computation_size_x];
    }

    // communicate boundaries east and west
    if (!is_westmost)
    {
        // west outgoing
        MPI_Isend(west_outgoing, computation_size_y, MPI_DOUBLE, west_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // west incoming
        MPI_Irecv(west_incoming, computation_size_y, MPI_DOUBLE, west_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
    }
    if (!is_eastmost)
    {
        // east outgoing
        MPI_Isend(east_outgoing, computation_size_y, MPI_DOUBLE, east_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // east incoming
        MPI_Irecv(east_incoming, computation_size_y, MPI_DOUBLE, east_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
    }

    MPI_Waitall(comm_wait_count, requests, stats);
    comm_wait_count = 0;
    cout << "Finished w-e comms: " << mpi_rank << endl;
    // unpack west and east incoming
    for (size_t row_idx = 0; row_idx < computation_size_y; row_idx++)
    {
        E_prev_local[row_idx + 1][0] = west_incoming[row_idx];
        E_prev_local[row_idx + 1][computation_size_x + 1] = east_incoming[row_idx];
    }

    // communicate boundaries north and south
    if (!is_northmost)
    {
        //  north outgoing
        MPI_Isend(E_prev_local[1], computation_size_x + 2, MPI_DOUBLE, north_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // north incoming
        MPI_Irecv(E_prev_local[0], computation_size_x + 2, MPI_DOUBLE, north_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
    }

    if (!is_southmost)
    {
        //  south outgoing
        MPI_Isend(E_prev_local[1], computation_size_x + 2, MPI_DOUBLE, south_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
        // south incoming
        MPI_Irecv(E_prev_local[0], computation_size_x + 2, MPI_DOUBLE, south_rank, 0, MPI_COMM_WORLD, &requests[comm_wait_count]);
        comm_wait_count++;
    }
    // wait for boundary comms
    MPI_Waitall(comm_wait_count, requests, stats);
    comm_wait_count = 0;

    // Solve for the excitation, the PDE
    for (j = 1; j <= computation_size_y; j++)
    {
        for (i = 1; i <= computation_size_x; i++)
        {
            E_local[j][i] = E_prev_local[j][i] + alpha * (E_prev_local[j][i + 1] + E_prev_local[j][i - 1] - 4 * E_prev_local[j][i] + E_prev_local[j + 1][i] + E_prev_local[j - 1][i]);
        }
    }

    /*
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    for (j = 1; j <= computation_size_y; j++)
    {
        for (i = 1; i <= n; i++)
            E_local[j][i] = E_local[j][i] - dt * (kk * E_local[j][i] * (E_local[j][i] - a) * (E_local[j][i] - 1) + E_local[j][i] * R_local[j][i]);
    }

    for (j = 1; j <= computation_size_y; j++)
    {
        for (i = 1; i <= n; i++)
            R_local[j][i] = R_local[j][i] + dt * (epsilon + M1 * R_local[j][i] / (E_local[j][i] + M2)) * (-R_local[j][i] - kk * E_local[j][i] * (E_local[j][i] - b - 1));
    }
    if (gather)
    {
        int counts[py];
        int displacements[py];
        int displacement = 0;
        int idx = 0;

        for (int row_idx = 0; row_idx < py; row_idx++)
        {
            for (int col_idx = 0; col_idx < px; col_idx++)
            {
                displacements[idx] = displacement;
                int curr_square_size_x = col_idx == py - 1 ? last_cell_computation_size_x : regular_computation_size_x;
                int curr_square_size_y = row_idx == py - 1 ? last_cell_computation_size_x : regular_computation_size_x;
                int curr_square_size = curr_square_size_x * curr_square_size_y;
                displacement += curr_square_size;
                counts[idx] = curr_square_size;
                idx++;
            }
        }
        memcpy2d(E_local, E_gather_local, 1, 1, 0, 0, computation_size_x, computation_size_y);
        MPI_Gatherv(E_gather_local[0], computation_size_x * computation_size_y, MPI_DOUBLE, E_gather[0], counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0)
        {
            int offset = 0;
            for (int row_idx = 0; row_idx < py; row_idx++)
            {
                for (int col_idx = 0; col_idx < px; col_idx++)
                {
                    int curr_square_size_x = col_idx == py - 1 ? last_cell_computation_size_x : regular_computation_size_x;
                    int curr_square_size_y = row_idx == py - 1 ? last_cell_computation_size_x : regular_computation_size_x;
                    int curr_square_size = curr_square_size_x * curr_square_size_y;
                    for (int curr_square_row_idx = 0; curr_square_row_idx < curr_square_size_y; curr_square_row_idx++)
                    {
                        memcpy(&E[row_idx][col_idx + curr_square_row_idx], E_gather[0] + offset + curr_square_row_idx * curr_square_size_x, curr_square_size_x * sizeof(double));
                    }

                    offset += curr_square_size;
                }
            }
        }
    }
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

    if (mpi_rank == 0)
    {
        cout << "Grid Size       : " << n << endl;
        cout << "Duration of Sim : " << T << endl;
        cout << "Time step dt    : " << dt << endl;
        cout << "Process geometry: " << px << " x " << py << endl;
        if (no_comm)
            cout << "Communication   : DISABLED" << endl;
        cout << endl;
        cout << "Plot freq " << plot_freq << endl;
    }

    // Start the timer
    double t0 = getTime();

    // Simulated time is different from the integer timestep number
    // Simulated time
    double t = 0.0;
    // Integer timestep number
    int niter = 0;

    // user variables
    int array_size_0 = m + 2;
    int array_size_1 = n + 2;

    int p_x_idx = mpi_rank % px;
    int p_y_idx = mpi_rank / px;

    bool is_northmost = (p_y_idx == 0);
    bool is_southmost = (p_y_idx == py - 1);

    bool is_westmost = (p_x_idx == 0);
    bool is_eastmost = (p_x_idx == px - 1);

    int south_rank = is_southmost ? -1 : mpi_rank + px;
    int north_rank = is_northmost ? -1 : mpi_rank - px;

    int east_rank = is_eastmost ? -1 : mpi_rank + 1;
    int west_rank = is_westmost ? -1 : mpi_rank - 1;

    int regular_computation_size_y = m / py;
    int last_cell_computation_size_y = m - (regular_computation_size_y * (py - 1));

    int regular_computation_size_x = n / px;
    int last_cell_computation_size_x = n - (regular_computation_size_x * (px - 1));

    int computation_size_x = is_eastmost ? last_cell_computation_size_x : regular_computation_size_x;
    int computation_size_y = is_southmost ? last_cell_computation_size_y : regular_computation_size_y;

    double **E_local = alloc2D(computation_size_y + 2, computation_size_x + 2);
    double **E_prev_local = alloc2D(computation_size_y + 2, computation_size_x + 2);
    double **R_local = alloc2D(computation_size_y + 2, computation_size_x + 2);
    double **E_gather = alloc2D(m, n);
    double **E_gather_local = alloc2D(computation_size_y, computation_size_x);

    if (E_local == NULL || E_prev_local == NULL || R_local == NULL)
    {
        return 1;
    }

    int local_array_in_global_idx_x = p_x_idx * regular_computation_size_x;
    int local_array_in_global_idx_y = p_y_idx * regular_computation_size_y;

    memcpy2d(E_prev, E_prev_local, local_array_in_global_idx_x, local_array_in_global_idx_y, 0, 0, computation_size_x + 2, computation_size_y + 2);
    memcpy2d(R, R_local, local_array_in_global_idx_x, local_array_in_global_idx_y, 0, 0, computation_size_x + 2, computation_size_y + 2);

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

        simulate(E, E_gather, E_gather_local,
                 E_local, E_prev_local, R_local,
                 local_array_in_global_idx_x, local_array_in_global_idx_y,
                 p_x_idx, p_y_idx,
                 regular_computation_size_x, regular_computation_size_y,
                 last_cell_computation_size_x, last_cell_computation_size_y,
                 computation_size_x, computation_size_y,
                 alpha, n, m, kk,
                 dt, a, epsilon,
                 M1, M2, b, px, py, gather);

        // swap current E_local with  E_prev_local
        double **tmp = E_local;
        E_local = E_prev_local;
        E_prev_local = tmp;

        if (plot_freq)
        {
            int k = (int)(t / plot_freq);
            if ((t - k * plot_freq) < dt)
            {
                if (mpi_rank == 0)
                {
                    splot(E, t, niter, m + 2, n + 2);
                }
            }
        }
    } // end of while loop
    double time_elapsed = getTime() - t0;

    double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
    double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

    if (mpi_rank == 0)
    {
        cout << "Number of Iterations        : " << niter << endl;
        cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
        cout << "Sustained Gflops Rate       : " << Gflops << endl;
        cout << "Sustained Bandwidth (GB/sec): " << BW << endl
             << endl;

        double mx;
        double l2norm = stats(E, m, n, &mx);
        cout << "Max: " << mx << " L2norm: " << l2norm << endl;
    }

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
