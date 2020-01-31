#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "lbfgs.h"

using namespace std;

void mcstep(float &stx,float &fx,float &dx,float &sty,float &fy,float &dy,float &stp,float &fp,float &dp,bool &brackt,float &stmin,float &stmax,int &info){

    /*
     SUBROUTINE MCSTEP

     THE PURPOSE OF MCSTEP IS TO COMPUTE A SAFEGUARDED STEP FOR
     A LINESEARCH AND TO UPDATE AN INTERVAL OF UNCERTAINTY FOR
     A MINIMIZER OF THE FUNCTION.

     THE PARAMETER STX CONTAINS THE STEP WITH THE LEAST FUNCTION
     VALUE. THE PARAMETER STP CONTAINS THE CURRENT STEP. IT IS
     ASSUMED THAT THE DERIVATIVE AT STX IS NEGATIVE IN THE
     DIRECTION OF THE STEP. IF BRACKT IS SET TRUE THEN A
     MINIMIZER HAS BEEN BRACKETED IN AN INTERVAL OF UNCERTAINTY
     WITH ENDPOINTS STX AND STY.

     THE SUBROUTINE STATEMENT IS

       SUBROUTINE MCSTEP(STX,FX,DX,STY,FY,DY,STP,FP,DP,BRACKT,
                        STMIN,STMAX,INFO)

     WHERE

       STX, FX, AND DX ARE VARIABLES WHICH SPECIFY THE STEP,
         THE FUNCTION, AND THE DERIVATIVE AT THE BEST STEP OBTAINED
         SO FAR. THE DERIVATIVE MUST BE NEGATIVE IN THE DIRECTION
         OF THE STEP, THAT IS, DX AND STP-STX MUST HAVE OPPOSITE
         SIGNS. ON OUTPUT THESE PARAMETERS ARE UPDATED APPROPRIATELY.

       STY, FY, AND DY ARE VARIABLES WHICH SPECIFY THE STEP,
         THE FUNCTION, AND THE DERIVATIVE AT THE OTHER ENDPOINT OF
         THE INTERVAL OF UNCERTAINTY. ON OUTPUT THESE PARAMETERS ARE
         UPDATED APPROPRIATELY.

       STP, FP, AND DP ARE VARIABLES WHICH SPECIFY THE STEP,
         THE FUNCTION, AND THE DERIVATIVE AT THE CURRENT STEP.
         IF BRACKT IS SET TRUE THEN ON INPUT STP MUST BE
         BETWEEN STX AND STY. ON OUTPUT STP IS SET TO THE NEW STEP.

       BRACKT IS A LOGICAL VARIABLE WHICH SPECIFIES IF A MINIMIZER
         HAS BEEN BRACKETED. IF THE MINIMIZER HAS NOT BEEN BRACKETED
         THEN ON INPUT BRACKT MUST BE SET FALSE. IF THE MINIMIZER
         IS BRACKETED THEN ON OUTPUT BRACKT IS SET TRUE.

       STMIN AND STMAX ARE INPUT VARIABLES WHICH SPECIFY LOWER
         AND UPPER BOUNDS FOR THE STEP.

       INFO IS AN INTEGER OUTPUT VARIABLE SET AS FOLLOWS:
         IF INFO = 1,2,3,4,5, THEN THE STEP HAS BEEN COMPUTED
         ACCORDING TO ONE OF THE FIVE CASES BELOW. OTHERWISE
         INFO = 0, AND THIS INDICATES IMPROPER INPUT PARAMETERS.
    */
    
    bool bound;
    float gamma,p,q,r,s,sgnd,stpc,stpf,stpq,theta;

    info=0;

    /*
     CHECK THE INPUT PARAMETERS FOR ERRORS.
    */
    if((brackt && (stp<=min(stx,sty) || stp>=max(stx,sty))) || dx*(stp-stx)>=0.0f || stmax <stmin) return;

    /*
     DETERMINE IF THE DERIVATIVES HAVE OPPOSITE SIGN.
    */
    sgnd=dp*(dx/fabs(dx));

    /*
     FIRST CASE. A HIGHER FUNCTION VALUE.
     THE MINIMUM IS BRACKETED. IF THE CUBIC STEP IS CLOSER
     TO STX THAN THE QUADRATIC STEP, THE CUBIC STEP IS TAKEN,
     ELSE THE AVERAGE OF THE CUBIC AND QUADRATIC STEPS IS TAKEN.
    */
    if(fp>fx){
        info=1;
        bound=true;
        theta=3.f*(fx-fp)/(stp-stx)+dx+dp;
        s=max(max(fabs(theta),fabs(dx)),fabs(dp));
        gamma=s*sqrt((theta/s)*(theta/s)-(dx/s)*(dp/s));
        if(stp<stx) gamma=-gamma;
        p=gamma-dx+theta;
        q=gamma-dx+gamma+dp;
        r=p/q;
        stpc=stx+r*(stp-stx);
        stpq=stx+((dx/((fx-fp)/(stp-stx)+dx))/2)*(stp-stx);
        if(fabs(stpc-stx)<fabs(stpq-stx)) stpf=stpc;
        else stpf=stpc+(stpq-stpc)/2;
        brackt=true;
    }
    /*
     SECOND CASE. A LOWER FUNCTION VALUE AND DERIVATIVES OF
     OPPOSITE SIGN. THE MINIMUM IS BRACKETED. IF THE CUBIC
     STEP IS CLOSER TO STX THAN THE QUADRATIC (SECANT) STEP,
     THE CUBIC STEP IS TAKEN, ELSE THE QUADRATIC STEP IS TAKEN.
    */
    else if(sgnd<0.0f){
        info=2;
        bound=false;
        theta=3.f*(fx-fp)/(stp-stx)+dx+dp;
        s=max(max(fabs(theta),fabs(dx)),fabs(dp));
        gamma=s*sqrt((theta/s)*(theta/s)-(dx/s)*(dp/s));
        if(stp>stx) gamma=-gamma;
        p=gamma-dp+theta;
        q=gamma-dp+gamma+dx;
        r=p/q;
        stpc=stp+r*(stx-stp);
        stpq=stp+(dp/(dp-dx))*(stx-stp);
        if(fabs(stpc-stp)>fabs(stpq-stp)) stpf=stpc;
        else stpf=stpq;
        brackt=true;
    }
    /*
     THIRD CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE
     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DECREASES.
     THE CUBIC STEP IS ONLY USED IF THE CUBIC TENDS TO INFINITY
     IN THE DIRECTION OF THE STEP OR IF THE MINIMUM OF THE CUBIC
     IS BEYOND STP. OTHERWISE THE CUBIC STEP IS DEFINED TO BE
     EITHER STPMIN OR STPMAX. THE QUADRATIC (SECANT) STEP IS ALSO
     COMPUTED AND IF THE MINIMUM IS BRACKETED THEN THE THE STEP
     CLOSEST TO STX IS TAKEN, ELSE THE STEP FARTHEST AWAY IS TAKEN.
    */
    else if(fabs(dp)<fabs(dx)){
        info=3;
        bound=true;
        theta=3.f*(fx-fp)/(stp-stx)+dx+dp;
        s=max(max(fabs(theta),fabs(dx)),fabs(dp));
    /*
        THE CASE GAMMA = 0 ONLY ARISES IF THE CUBIC DOES NOT TEND
        TO INFINITY IN THE DIRECTION OF THE STEP.
    */
        gamma=s*sqrt(max(0.0f,(theta/s)*(theta/s)-(dx/s)*(dp/s)));
        if(stp>stx) gamma=-gamma;
        p=gamma-dp+theta;
        q=gamma+(dx-dp)+gamma;
        r=p/q;
        if(r<0.0f && gamma!=0.0f) stpc=stp+r*(stx-stp);
        else if(stp>stx) stpc=stmax;
        else stpc=stmin;
        stpq=stp+(dp/(dp-dx))*(stx-stp);
        if(brackt){
            if(fabs(stp-stpc)<fabs(stp-stpq)) stpf=stpc;
            else stpf=stpq;
        }
        else{
            if(fabs(stp-stpc)>fabs(stp-stpq)) stpf=stpc;
            else stpf=stpq;
        }
    }
    /*
     FOURTH CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE
     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DOES
     NOT DECREASE. IF THE MINIMUM IS NOT BRACKETED, THE STEP
     IS EITHER STPMIN OR STPMAX, ELSE THE CUBIC STEP IS TAKEN.
    */
    else{
        info=4;
        bound=false;
        if(brackt){
            theta=3.f*(fp-fy)/(sty-stp)+dy+dp;
            s=max(max(fabs(theta),fabs(dy)),fabs(dp));
            gamma=s*sqrt((theta/s)*(theta/s)-(dy/s)*(dp/s));
            if(stp>sty) gamma=-gamma;
            p=gamma-dp+theta;
            q=gamma-dp+gamma+dy;
            r=p/q;
            stpc=stp+r*(sty-stp);
            stpf=stpc;
        }
        else if(stp>stx) stpf=stmax;
        else stpf=stmin;
    }

    /*
     UPDATE THE INTERVAL OF UNCERTAINTY. THIS UPDATE DOES NOT
     DEPEND ON THE NEW STEP OR THE CASE ANALYSIS ABOVE.
    */
    if(fp>fx){
        sty=stp;
        fy=fp;
        dy=dp;
    }
    else{
        if(sgnd<0.0f){
            sty=stx;
            fy=fx;
            dy=dx;
        }
        stx=stp;
        fx=fp;
        dx=dp;
    }

    /*
     COMPUTE THE NEW STEP AND SAFEGUARD IT.
    */
    stpf=min(stmax,stpf);
    stpf=max(stmin,stpf);
    stp=stpf;
    if(brackt && bound){
        if(sty>stx) stp=min(stx+0.66f*(sty-stx),stp);
        else stp=max(stx+0.66f*(sty-stx),stp);
    }
    
    return;
}

void mcsrch(size_t n,float *x,float &f,float *g,float *s,float &stp,int &info,int &nfev,float *wa,int *isave,float *dsave){

    /*
                     SUBROUTINE MCSRCH
                
     A slight modification of the subroutine CSRCH of More' and Thuente.
     The changes are to allow reverse communication, and do not affect
     the performance of the routine. 

     THE PURPOSE OF MCSRCH IS TO FIND A STEP WHICH SATISFIES
     A SUFFICIENT DECREASE CONDITION AND A CURVATURE CONDITION.

     AT EACH STAGE THE SUBROUTINE UPDATES AN INTERVAL OF
     UNCERTAINTY WITH ENDPOINTS STX AND STY. THE INTERVAL OF
     UNCERTAINTY IS INITIALLY CHOSEN SO THAT IT CONTAINS A
     MINIMIZER OF THE MODIFIED FUNCTION

          F(X+STP*S) - F(X) - FTOL*STP*(GRADF(X)'S).

     IF A STEP IS OBTAINED FOR WHICH THE MODIFIED FUNCTION
     HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE DERIVATIVE,
     THEN THE INTERVAL OF UNCERTAINTY IS CHOSEN SO THAT IT
     CONTAINS A MINIMIZER OF F(X+STP*S).

     THE ALGORITHM IS DESIGNED TO FIND A STEP WHICH SATISFIES
     THE SUFFICIENT DECREASE CONDITION

           F(X+STP*S) .LE. F(X) + FTOL*STP*(GRADF(X)'S),

     AND THE CURVATURE CONDITION

           ABS(GRADF(X+STP*S)'S)) .LE. GTOL*ABS(GRADF(X)'S).

     IF FTOL IS LESS THAN GTOL AND IF, FOR EXAMPLE, THE FUNCTION
     IS BOUNDED BELOW, THEN THERE IS ALWAYS A STEP WHICH SATISFIES
     BOTH CONDITIONS. IF NO STEP CAN BE FOUND WHICH SATISFIES BOTH
     CONDITIONS, THEN THE ALGORITHM USUALLY STOPS WHEN ROUNDING
     ERRORS PREVENT FURTHER PROGRESS. IN THIS CASE STP ONLY
     SATISFIES THE SUFFICIENT DECREASE CONDITION.

     THE SUBROUTINE STATEMENT IS

        SUBROUTINE MCSRCH(N,X,F,G,S,STP,INFO,NFEV,WA)
     WHERE

       N IS A POSITIVE INTEGER INPUT VARIABLE SET TO THE NUMBER
         OF VARIABLES.

       X IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE
         BASE POINT FOR THE LINE SEARCH. ON OUTPUT IT CONTAINS
         X + STP*S.

       F IS A VARIABLE. ON INPUT IT MUST CONTAIN THE VALUE OF F
         AT X. ON OUTPUT IT CONTAINS THE VALUE OF F AT X + STP*S.

       G IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE
         GRADIENT OF F AT X. ON OUTPUT IT CONTAINS THE GRADIENT
         OF F AT X + STP*S.

       S IS AN INPUT ARRAY OF LENGTH N WHICH SPECIFIES THE
         SEARCH DIRECTION.

       STP IS A NONNEGATIVE VARIABLE. ON INPUT STP CONTAINS AN
         INITIAL ESTIMATE OF A SATISFACTORY STEP. ON OUTPUT
         STP CONTAINS THE FINAL ESTIMATE.

       FTOL AND GTOL ARE NONNEGATIVE INPUT VARIABLES. (In this reverse
         communication implementation GTOL is defined in a COMMON
         statement.) TERMINATION OCCURS WHEN THE SUFFICIENT DECREASE
         CONDITION AND THE DIRECTIONAL DERIVATIVE CONDITION ARE
         SATISFIED.

       XTOL IS A NONNEGATIVE INPUT VARIABLE. TERMINATION OCCURS
         WHEN THE RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY
         IS AT MOST XTOL.

       STPMIN AND STPMAX ARE NONNEGATIVE INPUT VARIABLES WHICH
         SPECIFY LOWER AND UPPER BOUNDS FOR THE STEP. (In this reverse
         communication implementatin they are defined in a COMMON
         statement).

       MAXFEV IS A POSITIVE INTEGER INPUT VARIABLE. TERMINATION
         OCCURS WHEN THE NUMBER OF CALLS TO FCN IS AT LEAST
         MAXFEV BY THE END OF AN ITERATION.

       INFO IS AN INTEGER OUTPUT VARIABLE SET AS FOLLOWS:

         INFO = 0  IMPROPER INPUT PARAMETERS.

         INFO =-1  A RETURN IS MADE TO COMPUTE THE FUNCTION AND GRADIENT.

         INFO = 1  THE SUFFICIENT DECREASE CONDITION AND THE
                   DIRECTIONAL DERIVATIVE CONDITION HOLD.

         INFO = 2  RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY
                   IS AT MOST XTOL.

         INFO = 3  NUMBER OF CALLS TO FCN HAS REACHED MAXFEV.

         INFO = 4  THE STEP IS AT THE LOWER BOUND STPMIN.

         INFO = 5  THE STEP IS AT THE UPPER BOUND STPMAX.

         INFO = 6  ROUNDING ERRORS PREVENT FURTHER PROGRESS.
                   THERE MAY NOT BE A STEP WHICH SATISFIES THE
                   SUFFICIENT DECREASE AND CURVATURE CONDITIONS.
                   TOLERANCES MAY BE TOO SMALL.

       NFEV IS AN INTEGER OUTPUT VARIABLE SET TO THE NUMBER OF
         CALLS TO FCN.

       WA IS A WORK ARRAY OF LENGTH N.
       
       isave is integer array of length 4 and dsave is float array of length 13 to remember state variables on reentry.
    */

    int infoc;
    bool brackt,stage1;
    float dg,dgm,dginit,dgtest,dgx,dgxm,dgy,dgym,finit,ftest1,fm,fx,fxm,fy,fym,stx,sty,stmin,stmax,width,width1;

    if(info==-1){
//        fprintf(stderr,"in mcsrch with info=-1 f=%.10f\n",f);
//        if(!isfinite(f)) fprintf(stderr,"objfunc is not finite\n");
//        else fprintf(stderr,"objfunc is finite\n");
        /*
         * Load variabels from saved memory
         */
        infoc=isave[0];
        brackt=isave[1];
        stage1=isave[2];
        nfev=isave[3];
        dginit=dsave[0];
        finit=dsave[1];
        dgtest=dsave[2];
        width=dsave[3];
        width1=dsave[4];
        stx=dsave[5];
        fx=dsave[6];
        dgx=dsave[7];
        sty=dsave[8];
        fy=dsave[9];
        dgy=dsave[10];
        stmin=dsave[11];
        stmax=dsave[12];

        info=0;
        nfev++;
        dg=sdot(n,g,s);
        ftest1=finit+stp*dgtest;

        if(f>1e10){
            if(isinf(f)) fprintf(stderr,"objfunc is infinity\n");
            else if(isnan(f)) fprintf(stderr,"objfunc is nan\n");
            else if (!isfinite(f)) fprintf(stderr,"objfunc is not finite\n");
            else fprintf(stderr,"f is too large\n");

            fprintf(stderr,"try reducing step length by half\n");
            stp=stp/2;

            /*
            EVALUATE THE FUNCTION AND GRADIENT AT STP
            AND COMPUTE THE DIRECTIONAL DERIVATIVE.
            We return to main program to obtain F and G.
            */
            saxpyz(n,stp,s,wa,x);
            info=-1;
            fprintf(stderr,"trial steplength stp=%.10f\n",stp);
        
            /*
             * Save local variable to remember on reentry
             */
            isave[0]=infoc;
            isave[1]=brackt;
            isave[2]=stage1;
            isave[3]=nfev;
            dsave[0]=dginit;
            dsave[1]=finit;
            dsave[2]=dgtest;
            dsave[3]=width;
            dsave[4]=width1;
            dsave[5]=stx;
            dsave[6]=fx;
            dsave[7]=dgx;
            dsave[8]=sty;
            dsave[9]=fy;
            dsave[10]=dgy;
            dsave[11]=stmin;
            dsave[12]=stmax;
            
            return;
        }

        /*
        TEST FOR CONVERGENCE.
        */
//        fprintf(stderr,"TEST FOR CONVERGENCE OF LINE SEARCH\n");
//        fprintf(stderr,"f %.10f ftest1 %.10f dg %.10f dginit %.10f\n",f,ftest1,dg,dginit);
        if((brackt && (stp<=stmin || stp>=stmax)) || infoc==0){
            info=6;
            fprintf(stderr,"INFO=6. ROUNDING ERRORS PREVENT FURTHER PROGRESS.\nTHERE MAY NOT BE A STEP WHICH SATISFIES THE SUFFICIENT DECREASE AND CURVATURE CONDITIONS.\nTOLERANCES MAY BE TOO SMALL.\n");
        }
        else if(stp==stpmax && f<=ftest1 && dg<=dgtest){
            info=5;
            fprintf(stderr,"INFO=5. THE STEP IS AT THE UPPER BOUND STPMAX.\n");
        }
        else if(stp==stpmin && (f>ftest1 || dg>=dgtest)){
            info=4;
            fprintf(stderr,"INFO=4. THE STEP IS AT THE LOWER BOUND STPMIN.\n");
        }
        else if(nfev>=maxfev){
            info=3;
            fprintf(stderr,"INFO=3. NUMBER OF CALLS TO FUNCTION HAS REACHED MAXFEV OF %d.\n",maxfev);
        }
        else if(brackt && stmax-stmin<=xtol*stmax){
            info=2;
            fprintf(stderr,"INFO=2. RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY IS AT MOST XTOL.\n");
        }
        else if(f<=ftest1 && fabs(dg)<=gtol*(-dginit)) info=1;

        /*
        CHECK FOR TERMINATION.
        */
        if(info!=0) return;

        /*
        IN THE FIRST STAGE WE SEEK A STEP FOR WHICH THE MODIFIED
        FUNCTION HAS A NONPOSITIVE VALUE AND NONNEGATIVE DERIVATIVE.
        */
        if(stage1 && f<=ftest1 && dg>=min(ftol,gtol)*dginit) stage1=false;

        /*
        A MODIFIED FUNCTION IS USED TO PREDICT THE STEP ONLY IF
        WE HAVE NOT OBTAINED A STEP FOR WHICH THE MODIFIED
        FUNCTION HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE
        DERIVATIVE, AND IF A LOWER FUNCTION VALUE HAS BEEN
        OBTAINED BUT THE DECREASE IS NOT SUFFICIENT.
        */
        if(stage1 && f<=fx && f>ftest1){
            /*
            DEFINE THE MODIFIED FUNCTION AND DERIVATIVE VALUES.
            */
            fm=f-stp*dgtest;
            fxm=fx-stx*dgtest;
            fym=fy-sty*dgtest;
            dgm=dg-dgtest;
            dgxm=dgx-dgtest;
            dgym=dgy-dgtest;

            /*
            CALL CSTEP TO UPDATE THE INTERVAL OF UNCERTAINTY
            AND TO COMPUTE THE NEW STEP.
            */
            mcstep(stx,fxm,dgxm,sty,fym,dgym,stp,fm,dgm,brackt,stmin,stmax,infoc);

            /*
            RESET THE FUNCTION AND GRADIENT VALUES FOR F.
            */
            fx=fxm+stx*dgtest;
            fy=fym+sty*dgtest;
            dgx=dgxm+dgtest;
            dgy=dgym+dgtest;
        }
        else{
            /*
            CALL MCSTEP TO UPDATE THE INTERVAL OF UNCERTAINTY
            AND TO COMPUTE THE NEW STEP.
            */
            mcstep(stx,fx,dgx,sty,fy,dgy,stp,f,dg,brackt,stmin,stmax,infoc);
        }

        /*
        FORCE A SUFFICIENT DECREASE IN THE SIZE OF THE
        INTERVAL OF UNCERTAINTY.
        */
        if(brackt){
            if(fabs(sty-stx)>=0.66f*width1) stp=stx+0.5f*(sty-stx);
            width1=width;
            width=fabs(sty-stx);
        }
    }
    else{
        infoc=1;
    
        /*
         CHECK THE INPUT PARAMETERS FOR ERRORS.
        */
        if(n<=0 || stp<=0.0f || ftol<0.0f || gtol<0.0f || xtol<0.0f || stpmin<0.0f || stpmax<stpmin || maxfev<=0) return;
    
        /*
         COMPUTE THE INITIAL GRADIENT IN THE SEARCH DIRECTION
         AND CHECK THAT S IS A DESCENT DIRECTION.
        */
        dginit=sdot(n,g,s);
        if(dginit>=0.0f){
            fprintf(stderr,"THE SEARCH DIRECTION IS NOT A DESCENT DIRECTION\n");
            return;
        }
    
        /*
         INITIALIZE LOCAL VARIABLES.
        */
        brackt=false;
        stage1=true;
        nfev=0;
        finit=f;
        dgtest=ftol*dginit;
        width=stpmax-stpmin;
        width1=2*width;
        memcpy(wa,x,n*sizeof(float));
    
        /*
         THE VARIABLES STX, FX, DGX CONTAIN THE VALUES OF THE STEP,
         FUNCTION, AND DIRECTIONAL DERIVATIVE AT THE BEST STEP.
         THE VARIABLES STY, FY, DGY CONTAIN THE VALUE OF THE STEP,
         FUNCTION, AND DERIVATIVE AT THE OTHER ENDPOINT OF
         THE INTERVAL OF UNCERTAINTY.
         THE VARIABLES STP, F, DG CONTAIN THE VALUES OF THE STEP,
         FUNCTION, AND DERIVATIVE AT THE CURRENT STEP.
        */
        stx=0.0f;
        fx=finit;
        dgx=dginit;
        sty=0.0f;
        fy=finit;
        dgy=dginit;
    }
    
    /*
     START OF ITERATION.
    */
    /*
    SET THE MINIMUM AND MAXIMUM STEPS TO CORRESPOND
    TO THE PRESENT INTERVAL OF UNCERTAINTY.
    */
    if(brackt){
        stmin=min(stx,sty);
        stmax=max(stx,sty);
    }
    else{
        stmin=stx;
        stmax=stp+xtrapf*(stp-stx);
    }

    /*
    FORCE THE STEP TO BE WITHIN THE BOUNDS STPMAX AND STPMIN.
    */
    stp=max(stp,stpmin);
    stp=min(stp,stpmax);

    /*
    IF AN UNUSUAL TERMINATION IS TO OCCUR THEN LET
    STP BE THE LOWEST POINT OBTAINED SO FAR.
    */
    if((brackt && (stp<=stmin || stp>=stmax)) || nfev>=maxfev-1 || infoc==0 || (brackt && stmax-stmin<=xtol*stmax)) stp=stx;

    /*
    EVALUATE THE FUNCTION AND GRADIENT AT STP
    AND COMPUTE THE DIRECTIONAL DERIVATIVE.
    We return to main program to obtain F and G.
    */
    saxpyz(n,stp,s,wa,x);
    info=-1;
    fprintf(stderr,"trial steplength stp=%.10f\n",stp);

    /*
     * Save local variable to remember on reentry
     */
    isave[0]=infoc;
    isave[1]=brackt;
    isave[2]=stage1;
    isave[3]=nfev;
    dsave[0]=dginit;
    dsave[1]=finit;
    dsave[2]=dgtest;
    dsave[3]=width;
    dsave[4]=width1;
    dsave[5]=stx;
    dsave[6]=fx;
    dsave[7]=dgx;
    dsave[8]=sty;
    dsave[9]=fy;
    dsave[10]=dgy;
    dsave[11]=stmin;
    dsave[12]=stmax;
    
    return;
}

