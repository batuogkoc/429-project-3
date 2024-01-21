/* **********************************************************
 *  Author : Urvashi R.V. [04/06/2004]
 *      Modified by Didem Unat [03/23/15]
 *************************************************************/

#include <stdio.h>
#include <stdbool.h>

/* Function to plot the 2D array
 * 'gnuplot' is instantiated via a pipe and
 * the values to be plotted are passed through, along
 * with gnuplot commands */

FILE *gnu = NULL;

void splot(double **U, double T, int niter, int m, int n, bool to_png)
{
  int i, j;
  if (gnu == NULL)
    gnu = popen("gnuplot", "w");
  // gnu = fopen("plot.txt", "w");

  char filename[256];
  sprintf(filename, "plot_%d.png", niter);

  double mx = -1, mn = 32768;
  for (j = 0; j < m; j++)
  {
    for (i = 0; i < n; i++)
    {
      if (U[j][i] > mx)
        mx = U[j][i];
      if (U[j][i] < mn)
        mn = U[j][i];
    }
  }

  if (to_png)
  {
    // outputting the plot to png
    fprintf(gnu, "set terminal png\n");
    fprintf(gnu, "set output '%s'\n", filename);
  }

  fprintf(gnu, "set title \"T = %f [niter = %d]\"\n", T, niter);
  fprintf(gnu, "set size square\n");
  fprintf(gnu, "set key off\n");
  fprintf(gnu, "set pm3d map\n");
  // Various color schemes
  fprintf(gnu, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");

  //    fprintf(gnu,"set palette rgbformulae 22, 13, 31\n");
  //    fprintf(gnu,"set palette rgbformulae 30, 31, 32\n");

  fprintf(gnu, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 1, n - 1, mn, mx);
  for (j = 0; j < m; j++)
  {
    for (i = 0; i < n; i++)
    {
      fprintf(gnu, "%d %d %f\n", i, j, U[j][i]);
    }
    fprintf(gnu, "\n");
  }
  fprintf(gnu, "e\n");
  fflush(gnu);
  // fclose(gnu);
  return;
}
