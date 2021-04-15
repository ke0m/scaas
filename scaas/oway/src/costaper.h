/**
 * Applies a cosine taper at the borders
 * to N-dimensional data
 * A port of Mcostaper.c in Madagascar (written by Sergey Fomel)
 * @author: Joseph Jennings
 * @version: 2020.07.17
 */

#ifndef COSTAPER_H_
#define COSTAPER_H_

void costaper(int dim, int dim1, int n1, int n2, int *n, int *nw, int *s, float *data);
int first_index(int i, int j, int dim, const int *n, const int *s);

#endif /* COSTAPER_H_ */
