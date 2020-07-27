/**
 * Wrapper functions for the ssr3 class
 * @author: Joseph Jennings
 * @version: 2020.07.25
 */

#ifndef SSR3WRAP_H_
#define SSR3WRAP_H_

#include <complex>

void ssr3_modshots(int nx, int ny, int nz,              // Dimensions of velocity model/image
                   float ox, float oy, float oz,        // Origins of velocity model/image
                   float dx, float dy, float dz,        // Samplings of velocity model/image
                   int nw, float ow, float dw,          // Frequency axis
                   int ntx, int nty, int px, int py,    // Tapering and padding
                   float dtmax, int nrmax,              // Reference velocities
                   float *slo,                          // Input slowness [nz,ny,nx]
                   int nexp,                            // Number of experiments
                   int *nsrc, float *srcy, float *srcx, // Number of sources per experiment and coordinates   [nexp]
                   int *nrec, float *recy, float *recx, // Number of receivers per experiment and coordinates [nexp]
                   std::complex<float>*wav,             // Input source wavelets [nsrcmax,nw]
                   float *ref,                          // Input reflectivity [nz,ny,nx]
                   std::complex<float>*dat,             // Output data [nrecmax,nw]
                   int nthrds, int verb);               // Threading and verbosity

void ssr3_migshots(int nx, int ny, int nz,              // Dimensions of velocity model/image
                   float ox, float oy, float oz,        // Origins of velocity model/image
                   float dx, float dy, float dz,        // Samplings of velocity model/image
                   int nw, float ow, float dw,          // Frequency axis
                   int ntx, int nty, int px, int py,    // Tapering and padding
                   float dtmax, int nrmax,              // Reference velocities
                   float *slo,                          // Input slowness [nz,ny,nx]
                   int nexp,                            // Number of experiments
                   int *nsrc, float *srcy, float *srcx, // Number of sources per experiment and coordinates   [nexp]
                   int *nrec, float *recy, float *recx, // Number of receivers per experiment and coordinates [nexp]
                   std::complex<float>*dat,             // Input data [nrecmax,nw]
                   std::complex<float>*wav,             // Input source wavelets [nsrcmax,nw]
                   float *img,                          // Output image [nz,ny,nx]
                   int nthrds, int verb);               // Threading and verbosity


void ssr3_migoffshots(int nx, int ny, int nz,              // Dimensions of velocity model/image
                      float ox, float oy, float oz,        // Origins of velocity model/image
                      float dx, float dy, float dz,        // Samplings of velocity model/image
                      int nw, float ow, float dw,          // Frequency axis
                      int ntx, int nty, int px, int py,    // Tapering and padding
                      float dtmax, int nrmax,              // Reference velocities
                      float *slo,                          // Input slowness [nz,ny,nx]
                      int nexp,                            // Number of experiments
                      int *nsrc, float *srcy, float *srcx, // Number of sources per experiment and coordinates   [nexp]
                      int *nrec, float *recy, float *recx, // Number of receivers per experiment and coordinates [nexp]
                      std::complex<float>*dat,             // Input data [nrecmax,nw]
                      std::complex<float>*wav,             // Input source wavelets [nsrcmax,nw]
                      int nhy, int nhx, bool sym,          // Extended image parameters
                      float *img,                          // Output image [nhy,nhx,nz,ny,nx]
                      int nthrds, int verb);               // Threading and verbosity

#endif /* SSR3WRAP_H_ */
