#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "lbfgs.h"

using namespace std;

void steepest(size_t n,float *x,float &f,float *g,float *diag,float *w,int &iflag,int *isave,float *dsave){

    /*
     STEEPEST DESCENT SOLVER 
     HUY LE
     IMPLEMENTED BASED ON LBFGS SOLVER FROM NOCEDAL

 
     This subroutine solves the unconstrained minimization problem
 
                      min F(x),    x= (x1,x2,...,xN),

      using the steepest descent method.  

      The user is required to calculate the function value F and its
      gradient G. In order to allow the user complete control over
      these computations, reverse  communication is used. The routine
      must be called repeatedly under the control of the parameter
      IFLAG. 

      The steplength is determined at each iteration by means of the
      line search routine MCVSRCH, which is a slight modification of
      the routine CSRCH written by More' and Thuente.
 
      The calling statement is 
 
          CALL STEEPEST(N,X,F,G,DIAG,W,IFLAG)
 
      where
 
     N       is an INTEGER variable that must be set by the user to the
             number of variables. It is not altered by the routine.
             Restriction: N>0.
 
     X       is a REAL array of length N. On initial entry
             it must be set by the user to the values of the initial
             estimate of the solution vector. On exit with IFLAG=0, it
             contains the values of the variables at the best point
             found (usually a solution).
 
     F       is a REAL variable. Before initial entry and on
             a re-entry with IFLAG=1, it must be set by the user to
             contain the value of the function F at the point X.
 
     G       is a REAL array of length N. Before initial
             entry and on a re-entry with IFLAG=1, it must be set by
             the user to contain the components of the gradient G at
             the point X.
 
     DIAG    is just a temporary array to store the current iterate 
             while doing line search.
 
     EPS     is a positive REAL variable that must be set by
             the user, and determines the accuracy with which the solution
             is to be found. The subroutine terminates when

                         ||G|| < EPS max(1,||X||),

             where ||.|| denotes the Euclidean norm.
 
     XTOL    is a  positive REAL variable that must be set by
             the user to an estimate of the machine precision (e.g.
             10**(-16) on a SUN station 3/60). The line search routine will
             terminate if the relative width of the interval of uncertainty
             is less than XTOL.
 
     W       is a REAL array of length N used to store the search direction, 
             which for this solver is the negative gradient.
 
     IFLAG   is an INTEGER variable that must be set to 0 on initial entry
             to the subroutine. A return with IFLAG<0 indicates an error,
             and IFLAG=0 indicates that the routine has terminated without
             detecting errors. On a return with IFLAG=1, the user must
             evaluate the function F and gradient G. On a return with
             IFLAG=2, the user must provide the diagonal matrix Hk0.
 
             The following negative values of IFLAG, detecting an error,
             are possible:
 
              IFLAG=-1  The line search routine MCSRCH failed. The
                        parameter INFO provides more detailed information
                        (see also the documentation of MCSRCH):

                       INFO = 0  IMPROPER INPUT PARAMETERS.

                       INFO = 2  RELATIVE WIDTH OF THE INTERVAL OF
                                 UNCERTAINTY IS AT MOST XTOL.

                       INFO = 3  MORE THAN 20 FUNCTION EVALUATIONS WERE
                                 REQUIRED AT THE PRESENT ITERATION.

                       INFO = 4  THE STEP IS TOO SMALL.

                       INFO = 5  THE STEP IS TOO LARGE.

                       INFO = 6  ROUNDING ERRORS PREVENT FURTHER PROGRESS. 
                                 THERE MAY NOT BE A STEP WHICH SATISFIES
                                 THE SUFFICIENT DECREASE AND CURVATURE
                                 CONDITIONS. TOLERANCES MAY BE TOO SMALL.

 
              IFLAG=-2  The i-th diagonal element of the diagonal inverse
                        Hessian approximation, given in DIAG, is not
                        positive.
           
              IFLAG=-3  Improper input parameters for LBFGS (N or M are
                        not positive).
 
    GTOL is a REAL variable with default value 0.9, which
        controls the accuracy of the line search routine MCSRCH. If the
        function and gradient evaluations are inexpensive with respect
        to the cost of the iteration (which is sometimes the case when
        solving very large problems) it may be advantageous to set GTOL
        to a small value. A typical small value is 0.1.f  Restriction:
        GTOL should be greater than 1.fD-04.
 
    STPMIN and STPMAX are non-negative REAL variables which
        specify lower and uper bounds for the step in the line search.
        Their default values are 1.fD-20 and 1.fD+20, respectively. These
        values need not be modified unless the exponents are too large
        for the machine being used, or unless the problem is extremely
        badly scaled (in which case the exponents should be increased).
 

  MACHINE DEPENDENCIES

        The only variables that are machine-dependent are XTOL,
        STPMIN and STPMAX.
     */

    float gnorm,stp,stp1,xnorm;
    int iter,nfun,info,nfev;
    bool finish=false;

    /*
     INITIALIZE
     ----------
    */
    if(iflag==0){
        iter=0;

        nfun=1;

        /*
         THE WORK VECTOR W IS STORE THE SEARCH DIRECTION, 
         WHICH FOR THIS SOLVER IS THE NEGATIVE GRADIENT.
         */
        #pragma omp parallel for num_threads(16)
        for(size_t i=0;i<n;i++) w[i]=-g[i];
        gnorm=sqrt(sdot(n,g,g));
        stp1=1.f/gnorm;

        /*
        --------------------
         MAIN ITERATION LOOP
        --------------------
        */
        iter++;
        info=0;

        /*
         OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION 
         BY USING THE LINE SEARCH ROUTINE MCSRCH
         ----------------------------------------------------
         */
        nfev=0;
        stp=stp1;
            
        //continue here on reentry with iflag=1
        mcsrch(n,x,f,g,w,stp,info,nfev,diag,isave,dsave);

        if(info==-1){
            //this is always the case at the first iteration iflag=0, i.e. first call of mcsrch always returns info=-1
            iflag=1;
            
            //save state variable before return
            isave[4]=iter;
            isave[5]=nfun;
            isave[6]=info;
            dsave[13]=stp;
            
            return;
        }
    }
    else if(iflag==1){
        //continue here on reentry with iflag=1
        //load saved variables
        iter=isave[4];
        nfun=isave[5];
        info=isave[6];
        stp=dsave[13];

        while(true){
            mcsrch(n,x,f,g,w,stp,info,nfev,diag,isave,dsave);
    
            if(info==-1){
                //this is always the case at the first iteration iflag=0, i.e. first call of mcsrch always returns info=-1
                iflag=1;
                
                //save state variable before return
                isave[4]=iter;
                isave[5]=nfun;
                isave[6]=info;
                dsave[13]=stp;
                
                return;
            }
    
            if(info!=1){
                iflag=-1;
                fprintf(stderr,"IFLAG=-1. LINE SEARCH FAILED. SEE DOCUMENTATION OF ROUTINE MCSRCH.\nERROR RETURN OF LINE SEARCH: INFO=%d. POSSIBLE CAUSES: FUNCTION OR GRADIENT ARE INCORRECT\n",info);
                return;
            }
    
            nfun+=nfev;
    
    
            /*
             TERMINATION TEST
             ----------------
            */
            gnorm=sqrt(sdot(n,g,g));
            fprintf(stderr,"FOUND A SATISFACTORY STEPLENGTH STP=%.10f\nITER=%d NFUN=%d F=%.10f GNORM=%.10f\n\n",stp,iter,nfun,f,gnorm);
            xnorm=sqrt(sdot(n,x,x));
            xnorm=max(1.f,xnorm);
            if(gnorm/xnorm<=epsilon) finish=true;
            if(finish){
                iflag=0;
                fprintf(stderr,"IFLAG=0. TERMINATION CONDITIONS PASS. STEEPEST DESCENT ENDS.\n");
                return;
            }
            
            /*
            --------------------
             MAIN ITERATION LOOP
            --------------------
            */
            iter++;
            info=0;
    
            /*
             OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION 
             BY USING THE LINE SEARCH ROUTINE MCSRCH
             ----------------------------------------------------
             */
            nfev=0;
            #pragma omp parallel for num_threads(16)
            for(size_t i=0;i<n;i++) w[i]=-g[i];
        }
    }

    return;
}
