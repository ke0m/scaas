#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "lbfgs.h"

using namespace std;

void lbfgs(size_t n,int &m,float *x,float &f,float *g,bool diagco,float *diag,float *w,int &iflag,int *isave,float *dsave){

    /*
        LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION
                          JORGE NOCEDAL
                        *** July 1990 ***

 
     This subroutine solves the unconstrained minimization problem
 
                      min F(x),    x= (x1,x2,...,xN),

      using the limited memory BFGS method. The routine is especially
      effective on problems involving a large number of variables. In
      a typical iteration of this method an approximation Hk to the
      inverse of the Hessian is obtained by applying M BFGS updates to
      a diagonal matrix Hk0, using information from the previous M steps.
      The user specifies the number M, which determines the amount of
      storage required by the routine. The user may also provide the
      diagonal matrices Hk0 if not satisfied with the default choice.
      The algorithm is described in "On the limited memory BFGS method
      for large scale optimization", by D. Liu and J. Nocedal,
      Mathematical Programming B 45 (1989) 503-528.
 
      The user is required to calculate the function value F and its
      gradient G. In order to allow the user complete control over
      these computations, reverse  communication is used. The routine
      must be called repeatedly under the control of the parameter
      IFLAG. 

      The steplength is determined at each iteration by means of the
      line search routine MCVSRCH, which is a slight modification of
      the routine CSRCH written by More' and Thuente.
 
      The calling statement is 
 
          CALL LBFGS(N,M,X,F,G,DIAGCO,DIAG,IPRINT,W,IFLAG)
 
      where
 
     N       is an INTEGER variable that must be set by the user to the
             number of variables. It is not altered by the routine.
             Restriction: N>0.
 
     M       is an INTEGER variable that must be set by the user to
             the number of corrections used in the BFGS update. It
             is not altered by the routine. Values of M less than 3 are
             not recommended; large values of M will result in excessive
             computing time. 3<= M <=7 is recommended. Restriction: M>0.
 
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
 
     DIAGCO  is a LOGICAL variable that must be set to .TRUE. if the
             user  wishes to provide the diagonal matrix Hk0 at each
             iteration. Otherwise it should be set to .FALSE., in which
             case  LBFGS will use a default value described below. If
             DIAGCO is set to .TRUE. the routine will return at each
             iteration of the algorithm with IFLAG=2, and the diagonal
              matrix Hk0  must be provided in the array DIAG.
 
 
     DIAG    is a REAL array of length N. If DIAGCO=.TRUE.,
             then on initial entry or on re-entry with IFLAG=2, DIAG
             it must be set by the user to contain the values of the 
             diagonal matrix Hk0.  Restriction: all elements of DIAG
             must be positive.
 
     IPRINT  is an INTEGER array of length two which must be set by the
             user.
 
             IPRINT(1) specifies the frequency of the output:
                IPRINT(1) < 0 : no output is generated,
                IPRINT(1) = 0 : output only at first and last iteration,
                IPRINT(1) > 0 : output every IPRINT(1) iterations.
 
             IPRINT(2) specifies the type of output generated:
                IPRINT(2) = 0 : iteration count, number of function 
                                evaluations, function value, norm of the
                                gradient, and steplength,
                IPRINT(2) = 1 : same as IPRINT(2)=0, plus vector of
                                variables and  gradient vector at the
                                initial point,
                IPRINT(2) = 2 : same as IPRINT(2)=1, plus vector of
                                variables,
                IPRINT(2) = 3 : same as IPRINT(2)=2, plus gradient vector.
 
 
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
 
     W       is a REAL array of length N(2M+1)+2M used as
             workspace for LBFGS. This array must not be altered by the
             user.
 
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

    float gnorm,stp,stp1,ys,yy,sq,yr,beta,xnorm;
    int iter,nfun,point,info,bound,cp,nfev;
    size_t npt,ispt,iypt,inmc,iycn,iscn;
    bool finish=false;

    ispt=n+2*m;
    iypt=ispt+n*m;

    /*
     INITIALIZE
     ----------
    */
    if(iflag==0){
        iter=0;

        if(n<=0 || m<=0){
            iflag=-3;
            fprintf(stderr,"IFLAG=-3. IMPROPER INPUT PARAMETERS (N OR M) ARE NOT POSITIVE\n");
            return;
        }

        nfun=1;
        point=0;

        if(diagco){
            for(size_t i=0;i<n;i++){
                if(diag[i]<=0.0f){
                    fprintf(stderr,"IFLAG=-2. THE %lu-TH DIAGONAL ELEMENT OF THE INVERSE HESSIAN APPROXIMATION IS NOT POSITIVE\n",i);
                    return;
                }
            }
        }
        else{
            #pragma omp parallel for num_threads(16)
            for(size_t i=0;i<n;i++) diag[i]=1.f;
        }

        /*
         THE WORK VECTOR W IS DIVIDED AS FOLLOWS:
         ---------------------------------------
         THE FIRST N LOCATIONS ARE USED TO STORE THE GRADIENT AND
             OTHER TEMPORARY INFORMATION.
         LOCATIONS (N+1)...(N+M) STORE THE SCALARS RHO.
         LOCATIONS (N+M+1)...(N+2M) STORE THE NUMBERS ALPHA USED
             IN THE FORMULA THAT COMPUTES H*G.
         LOCATIONS (N+2M+1)...(N+2M+NM) STORE THE LAST M SEARCH
             STEPS.
         LOCATIONS (N+2M+NM+1)...(N+2M+2NM) STORE THE LAST M
             GRADIENT DIFFERENCES.
    
         THE SEARCH STEPS AND GRADIENT DIFFERENCES ARE STORED IN A
         CIRCULAR ORDER CONTROLLED BY THE PARAMETER POINT.
         */
        #pragma omp parallel for num_threads(16)
        for(size_t i=0;i<n;i++) w[ispt+i]=-g[i]*diag[i];
        gnorm=sqrt(sdot(n,g,g));
        stp1=1.f/gnorm;

        /*
        --------------------
         MAIN ITERATION LOOP
        --------------------
        */
        iter++;
        info=0;
        bound=iter-1;

        /*
         OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION 
         BY USING THE LINE SEARCH ROUTINE MCSRCH
         ----------------------------------------------------
         */
        nfev=0;
        stp=stp1;
        memcpy(w,g,n*sizeof(float));
            
        //continue here on reentry with iflag=1
        mcsrch(n,x,f,g,w+ispt+point*n,stp,info,nfev,diag,isave,dsave);

        if(info==-1){
            //this is always the case at the first iteration iflag=0, i.e. first call of mcsrch always returns info=-1
            iflag=1;
            
            //save state variable before return
            isave[4]=iter;
            isave[5]=point;
            isave[6]=nfun;
            isave[7]=info;
            dsave[13]=stp;
            
            return;
        }
    }
    //XXX: Generally, the optimization stays here
    else if(iflag==1){
        //continue here on reentry with iflag=1
        //load saved variables
        iter=isave[4];
        point=isave[5];
        nfun=isave[6];
        info=isave[7];
        stp=dsave[13];

        while(true){
            //XXX: Generally returns an info=0 or info=-1.
            mcsrch(n,x,f,g,w+ispt+point*n,stp,info,nfev,diag,isave,dsave);
    
            if(info==-1){
                //this is always the case at the first iteration iflag=0, i.e. first call of mcsrch always returns info=-1
                iflag=1;
                
                //save state variable before return
                isave[4]=iter;
                isave[5]=point;
                isave[6]=nfun;
                isave[7]=info;
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
             COMPUTE THE NEW STEP AND GRADIENT CHANGE 
             -----------------------------------------
            */
            npt=point*n;
            #pragma omp parallel for num_threads(16)
            for(size_t i=0;i<n;i++){
                w[ispt+npt+i]*=stp;
                w[iypt+npt+i]=g[i]-w[i];
            }
            point++;
            if(point==m) point=0;
        
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
                fprintf(stderr,"IFLAG=0. TERMINATION CONDITIONS PASS. LBFGS ENDS.\n");
                return;
            }
            
            /*
            --------------------
             MAIN ITERATION LOOP
            --------------------
            */
            iter++;
            info=0;
            bound=iter-1;
            if(iter!=1){
                if(iter>m) bound=m;
                ys=sdot(n,w+iypt+npt,w+ispt+npt);
                if(!diagco){
                    yy=sdot(n,w+iypt+npt,w+iypt+npt);
                    #pragma omp parallel for num_threads(16)
                    for(size_t i=0;i<n;i++) diag[i]=ys/yy;
                }
                else{
                    iflag=2;

                    //save state variable before return
                    isave[4]=iter;
                    isave[5]=point;
                    isave[6]=nfun;
                    isave[7]=info;
                    dsave[13]=stp;
                
                    return;
                }
                
                // continue here on reentry with iflag=2
                if(diagco){
                    for(size_t i=0;i<n;i++){
                        if(diag[i]<=0.0f){
                            fprintf(stderr,"IFLAG=-2. THE %lu-TH DIAGONAL ELEMENT OF THE INVERSE HESSIAN APPROXIMATION IS NOT POSITIVE\n",i);
                            return;
                        }
                    }
                }
    
                /*
                 COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
                 "Updating quasi-Newton matrices with limited storage",
                 Mathematics of Computation, Vol.24, No.151, pp. 773-782.
                 ---------------------------------------------------------
                */
                cp=point;
                if(point==0) cp=m;
                w[n+cp-1]=1.f/ys;
                #pragma omp parallel for num_threads(16)
                for(size_t i=0;i<n;i++) w[i]=-g[i];
                cp=point;
                
                for(int i=0;i<bound;i++){
                    cp--;
                    if(cp==-1) cp=m-1;
                    sq=sdot(n,w+ispt+cp*n,w);
                    inmc=n+m+cp;
                    iycn=iypt+cp*n;
                    w[inmc]=w[n+cp]*sq;
                    saxpy(n,-w[inmc],w+iycn,w);
                }
                
                #pragma omp parallel for num_threads(16)
                for(size_t i=0;i<n;i++) w[i]=diag[i]*w[i]; 
                
                for(int i=0;i<bound;i++){
                    yr=sdot(n,w+iypt+cp*n,w);
                    beta=w[n+cp]*yr;
                    inmc=n+m+cp;
                    beta=w[inmc]-beta;
                    iscn=ispt+cp*n;
                    saxpy(n,beta,w+iscn,w);
                    cp++;
                    if(cp==m) cp=0;
                }
        
                /*
                 STORE THE NEW SEARCH DIRECTION
                 ------------------------------
                */
                memcpy(w+ispt+point*n,w,n*sizeof(float)); 
            }
    
            /*
             OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION 
             BY USING THE LINE SEARCH ROUTINE MCSRCH
             ----------------------------------------------------
             */
            nfev=0;
            memcpy(w,g,n*sizeof(float));
        }
    }
    else if(iflag==2){
        // continue here on reentry with iflag=2
        //load saved variables
        iter=isave[4];
        point=isave[5];
        nfun=isave[6];
        info=isave[7];
        stp=dsave[13];

        while(true){
            if(iter!=1){
                if(diagco){
                    for(size_t i=0;i<n;i++){
                        if(diag[i]<=0.0f){
                            fprintf(stderr,"IFLAG=-2. THE %lu-TH DIAGONAL ELEMENT OF THE INVERSE HESSIAN APPROXIMATION IS NOT POSITIVE\n",i);
                            return;
                        }
                    }
                }
    
                /*
                 COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
                 "Updating quasi-Newton matrices with limited storage",
                 Mathematics of Computation, Vol.24, No.151, pp. 773-782.
                 ---------------------------------------------------------
                */
                cp=point;
                if(point==0) cp=m;
                w[n+cp-1]=1.f/ys;
                for(size_t i=0;i<n;i++) w[i]=-g[i];
                cp=point;
                
                for(int i=0;i<bound;i++){
                    cp--;
                    if(cp==-1) cp=m-1;
                    sq=sdot(n,w+ispt+cp*n,w);
                    inmc=n+m+cp;
                    iycn=iypt+cp*n;
                    w[inmc]=w[n+cp]*sq;
                    saxpy(n,-w[inmc],w+iycn,w);
                }
                
                for(size_t i=0;i<n;i++) w[i]=diag[i]*w[i]; 
                
                for(int i=0;i<bound;i++){
                    yr=sdot(n,w+iypt+cp*n,w);
                    beta=w[n+cp]*yr;
                    inmc=n+m+cp;
                    beta=w[inmc]-beta;
                    iscn=ispt+cp*n;
                    saxpy(n,beta,w+iscn,w);
                    cp++;
                    if(cp==m) cp=0;
                }
        
                /*
                 STORE THE NEW SEARCH DIRECTION
                 ------------------------------
                */
                memcpy(w+ispt+point*n,w,n*sizeof(float)); 
            }
    
            /*
             OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION 
             BY USING THE LINE SEARCH ROUTINE MCSRCH
             ----------------------------------------------------
             */
            nfev=0;
            if(iter==1) stp=stp1;
            memcpy(w,g,n*sizeof(float));
                
            //continue here on reentry with iflag=1
            mcsrch(n,x,f,g,w+ispt+point*n,stp,info,nfev,diag,isave,dsave);
    
            if(info==-1){
                //this is always the case at the first iteration iflag=0, i.e. first call of mcsrch always returns info=-1
                iflag=1;
                
                //save state variable before return
                isave[4]=iter;
                isave[5]=point;
                isave[6]=nfun;
                isave[7]=info;
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
             COMPUTE THE NEW STEP AND GRADIENT CHANGE 
             -----------------------------------------
            */
            npt=point*n;
            for(size_t i=0;i<n;i++){
                w[ispt+npt+i]*=stp;
                w[iypt+npt+i]=g[i]-w[i];
            }
            point++;
            if(point==m) point=0;
        
            /*
             TERMINATION TEST
             ----------------
            */
            gnorm=sqrt(sdot(n,g,g));
            xnorm=sqrt(sdot(n,x,x));
            xnorm=max(1.f,xnorm);
            if(gnorm/xnorm<=epsilon) finish=true;
            if(finish){
                iflag=0;
                return;
            }
            /*
            --------------------
             MAIN ITERATION LOOP
            --------------------
            */
            iter++;
            info=0;
            bound=iter-1;
            if(iter!=1){
                if(iter>m) bound=m;
                ys=sdot(n,w+iypt+npt,w+ispt+npt);
                if(!diagco){
                    yy=sdot(n,w+iypt+npt,w+iypt+npt);
                    for(size_t i=0;i<n;i++) diag[i]=ys/yy;
                }
                else{
                    iflag=2;
    
                    //save state variable before return
                    isave[4]=iter;
                    isave[5]=point;
                    isave[6]=nfun;
                    isave[7]=info;
                    dsave[13]=stp;
                
                    return;
                }
            }            
        }
    }

    return;
}
