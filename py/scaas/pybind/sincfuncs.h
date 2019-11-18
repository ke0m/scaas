/* Evaluate the sinc function */
float fsinc(float x);
double dsinc(double x);

/* Toeplitz solver for calculating sinc coefficients */
void stoepd (int n, double r[], double g[], double f[], double a[]);
void stoepf (int n, float r[], float g[], float f[], float a[]);

/* Makes sinc coefficients */
void mksinc (float d, int lsinc, float sinc[]);

/* Performs the interpolation */
void intt8r (int ntable, float table[][8],
  int nxin, float dxin, float fxin, float yin[], float yinl, float yinr,
  int nxout, float xout[], float yout[]);

/* Makes coefficients and performs the interpolation */
void ints8r (int nxin, float dxin, float fxin, float yin[],
  float yinl, float yinr, int nxout, float xout[], float yout[]);


