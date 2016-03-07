#ifdef __cplusplus
extern "C" {
#endif



#include <math.h>
#ifdef MKL

#include <mkl.h>
#include <mkl_cblas.h>
#define LAPACK_DPOTRF LAPACKE_dpotrf
#define LAPACK_DPOTRI LAPACKE_dpotri
#else

#include "clapack.h"
#include "cblas.h"
// atlas version of clapack uses other version compared to reg mac
#ifdef ATL_INT

#define LAPACK_DPOTRF clapack_dpotrf
#define LAPACK_DPOTRI clapack_dpotri

#else

#define LAPACK_DPOTRF dpotrf
#define LAPACK_DPOTRI dpotri

#endif
#endif

#include <stdlib.h> 
#include <string.h>

int sigma_update(double *Qstar,const double* Q,const double* X,const long *z_index,const long* z_count, const int k, const int d,const long int n)
{
    int i,j;
    int n_count = z_count[k];   
    int index, dim;
    double *X_temp = (double*) malloc(d*n_count * sizeof(double));

     for( i=0; i < d ;i++){
        for( j= i ; j <  d  ;j++){
            Qstar[d * i + j] = Q[d * i + j];
        }
    }  
    for(i = 0; i < n_count; i++)
    {   
        index = z_index[i + k*n];
        for(dim = 0; dim < d ; dim++)
            X_temp[d*i + dim] = X[d*index + dim];
    }

   // cblas_dgemm(CblasRowMajor, CblasTrans,CblasNoTrans, d, d, n_count, 1., X_temp, d,
   // 				X_temp,d,1.,Qstar,d);
    cblas_dsyrk(CblasRowMajor, CblasUpper,CblasTrans, d, n_count, 1, X_temp, d,1,Qstar,d);
    for( i=0; i < d ;i++){
	for( j=0; j <  i  ;j++)
			Qstar[j + d * i] = Qstar[d * j + i];
	}
    free(X_temp);
 
/*
    for( i=0; i < d ;i++){
	for( j=0; j <  i  ;j++)
			Qstar[i + d * j] = Qstar[d * i + j];
	}
*/
    //cblas_dspr(CblasColMajor,CblasUpper,d,1.,X,1,Qstar);
    return n_count;
}

void mult_c(double *res,  const double *sigma_inv,const double* theta,const int d, const double beta)
{
/*

void cblas_ssymv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY);
*/

    cblas_dsymv(CblasColMajor,CblasUpper, d, 1., sigma_inv, d, theta, 1, beta, res, 1);

}

void sample_muc(double *sigma_inv, const double *sigma_mu_inv, double* mu_c,  double* r, const double n_d, const long int d)
{
    int i,j;
    for( i=0; i < d ;i++){
        for( j=0; j < (i + 1) ;j++){
            sigma_inv[d * i + j] *= n_d;
            sigma_inv[d * i + j] += sigma_mu_inv[d * i + j];
        }
    }
#ifdef MKL
    LAPACK_DPOTRF(LAPACK_COL_MAJOR, 'U',d, sigma_inv,d);
#else
    LAPACK_DPOTRF( CblasColMajor, CblasUpper,d, sigma_inv,d);
#endif

    cblas_dtrsm(CblasColMajor,CblasLeft,CblasUpper,CblasTrans,CblasNonUnit, d,1, 1. ,sigma_inv,d,mu_c,d);
    cblas_dtrsm(CblasColMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit, d,1, 1. ,sigma_inv,d,mu_c,d);
}

void inv_sigma_c( double *sigma_inv, const double *sigma, const int d)
{
    int i,j;
    for( i=0; i < d ;i++){
	for( j=0; j < (i + 1) ;j++)
			sigma_inv[d * i + j] = sigma[d * i + j];
	}
#ifdef MKL
    LAPACK_DPOTRF(LAPACK_COL_MAJOR, 'U',d, sigma_inv,d);
    LAPACK_DPOTRI(LAPACK_COL_MAJOR, 'U',d, sigma_inv,d);
#else
    LAPACK_DPOTRF( CblasColMajor, CblasUpper,d, sigma_inv,d);
    LAPACK_DPOTRI( CblasColMajor, CblasUpper,d, sigma_inv,d);
#endif

}


void draw_x_c(long* x,long* x_index, long* x_count,const double* P,const double* U,const long int N, const int K) {

    long int i;
    for(i = 0; i < K; i++)
        x_count[i] = 0;
    for (i = 0; i < N; i++) {
        int iK = i*K;
        double p = *(P + iK);
        int val = 0;
        while(U[i] > p)
            p += *(P + iK + ++val);

        x[i] = val;
        x_index[N*val + x_count[val]++] = i;
    }
}

void calc_lik_c(double* lik ,const double* X,const double* Rp, const long int N, const int D, const double log_detQ_div) {


    int i,j,k;
    for (i = 0; i < N; i++) {
        lik[i] = log_detQ_div - D * log(2. * M_PI);
        int count = 0;
        for(j = 0; j < D; j++){
             double Qx = 0;
            for(k = j; k < D; k++ )
                Qx += Rp[count++] * X[D*i + k];
                
            lik[i] -= Qx*Qx;
        }
        lik[i] /= 2;
    }
}



void calc_exp_normalize_c(double* P,const double* prior, const long int N, const int K, const int K_act, const long int* act_index)
{
    int i,k;
    int k_act;
    //printf("K = %d\n",K);
    for( i = 0; i < N; i++)
    {
        int Ki = K*i;
        k_act = act_index[0];
        //printf("k_act = %d\n",k_act);
        P[Ki + k_act] += log(prior[k_act]);
        double max_P = P[Ki + k_act] ;

        
        for(k = 1; k < K_act ; k++){
        	k_act = act_index[k];
        	//printf("k_act = %d\n",k_act);
            //printf("P[Ki + k_act] = %.2f\n",P[Ki + k_act]);
			P[Ki + k_act] += log(prior[k_act]);
			if(P[Ki + k_act] > max_P)
				max_P = P[Ki + k_act];
        }
        double sum_P = 0;
        for(k = 0; k < K_act; k++){
        	k_act = act_index[k];
        	P[Ki + k_act] = exp(P[Ki + k_act] - max_P);
        	sum_P += P[Ki + k_act];
        }
        double sum_P_div = 1/sum_P;
        for(k = 0; k < K_act; k++){
        	k_act = act_index[k];
        	P[Ki + k_act] *= sum_P_div;
        }
        
        //self.prob_X += np.log(self.p)
        //self.prob_X -= np.reshape(np.max(self.prob_X,1),(self.prob_X.shape[0],1))
        //self.prob_X = np.exp(self.prob_X)
        //self.prob_X /= np.reshape(np.sum(self.prob_X,1),(self.prob_X.shape[0],1)) 
    }
}




/*
 *  Cholesky factorisation
 *  of X stored in R!
 *
 */
void chol_c( double *R, const double *X, const int d)
{
    int i,j;
    for( i=0; i < d ;i++){
	for( j=0; j < (i + 1) ;j++)
			R[d * j + i] = X[d * j + i];

	}
#ifdef MKL
    LAPACK_DPOTRF(LAPACK_ROW_MAJOR, 'U',d, R,d);
#else
    LAPACK_DPOTRF( CblasRowMajor, CblasUpper,d, R,d);
#endif

}

/*
	solving an Y = A^-1 X
	R - (dxd) cholesky of A
	X - (dxn) stores Y in X


*/
void solve_R_c(const double *R, double *X, const int d, const int n)
{
	cblas_dtrsm(CblasRowMajor,CblasLeft,CblasUpper,CblasTrans,CblasNonUnit, d, n, 1. ,R,d,X,  n);
	cblas_dtrsm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit, d, n, 1. ,R,d,X,n);
}
//cblas_dtrsm(CblasColMajor,CblasLeft,CblasUpper,CblasTrans,CblasNonUnit, d,1, 1. ,L,d,X,d);

#ifdef __cplusplus
}
#endif
