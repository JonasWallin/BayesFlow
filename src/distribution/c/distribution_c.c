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
#define LAPACK_DPOTRF clapack_dpotrf
#define LAPACK_DPOTRI clapack_dpotri
#endif
#include <stdlib.h> 
#include <string.h>


/*
 *
 *  Takes the outer product of a vector Y and then sums it up
 * Y   - (nxd)
 * YYt - (dxd)
 * n
 * d
 */
void outer_Y(double* YYt, const double* Y, const int n, const int d)
{
	// d(N), n(K), 1(alpha), Y, d(leading dimension of Y), 0 (beta), d(leading dimension of YYt)
	cblas_dsyrk(CblasRowMajor, CblasUpper,CblasTrans, d, n, 1, Y, d,0,YYt ,d);
	int i,ii;
	for( i = 0; i < d; i++)
	{
		for( ii = 0; ii < i ; ii++)
			YYt[d * i + ii] = YYt[d * ii + i];
	}
}

//cblas_dsyrk(CblasRowMajor, CblasUpper,CblasTrans, d, n_count, 1, X_temp, d,1,Qstar,d);

void sum_Y(double* sumY,const double* Y, const int n, const int d) {

    int i, ii;
    for(i = 0; i < d; i++)
    	sumY[i] = 0;
    for(i = 0; i < n; i++)
    {
    	for(ii = 0 ; ii < d ; ii++)
    		sumY[ii] += Y[i*d + ii];

    }
}

void update_mu_Q_sample(double* mu_sample, double *Q_sample, const double* Q_pmu_p, const double* Q,
		                const double* Q_p, const double* sumY, const long int n, const int d)
{
	int i,ii;
	double Q_ii;
	for( i = 0; i < d; i++)
	{
		mu_sample[i] = Q_pmu_p[i];
		for( ii = 0; ii <= i; ii++){
			Q_ii = Q[i*d + ii];
			Q_sample[i * d + ii] = Q_p[i*d + ii] + n *Q_ii;
			mu_sample[i] += Q_ii * sumY[ii];
		}
		for( ii = i+1; ii < d ; ii++){
			Q_ii = Q[i*d + ii];
			mu_sample[i] += Q_ii * sumY[ii];
		}
	}

#ifdef MKL
	LAPACK_DPOTRF(LAPACK_ROW_MAJOR, 'L',d, Q_sample,d);
#else
	LAPACK_DPOTRF( CblasRowMajor, CblasLower,d, Q_sample,d);
#endif
	// ..., d (order of Q), Q, d (lda leading dimension), X , increment X
	cblas_dtrsv(CblasRowMajor,CblasLower,CblasNoTrans,CblasNonUnit, d,Q_sample,d,mu_sample,1);

	//DPPTRF
	//cblas_dtrsm
}

void Lt_XpZ(const double* L, double* X, const double* Z, const int d)
{
	int i;
	for( i = 0; i < d; i++)
		X[i] += Z[i];
	cblas_dtrsv(CblasRowMajor,CblasLower,CblasTrans,CblasNonUnit, d,L,d,X,1);
}

void wishartrand(double* phi, const int d , double* X_rand, double* X_out){
	/*
	*
	*	X- rand is dxd matrix with N(0,1) on off diagonal and chisquare(nu - i) on diagonal
	*
	*  result is stored in X_out
	*/

	int i,ii;
	/*
	printf("Q  = [\n");
	for(i = 0; i < d; i++)
	{
		for(ii = 0; ii <d ; ii++)
			printf(" %.6f ",phi[ d * i + ii ]);
		printf("\n");
	}
	printf("]\n");
	*/
	//choleksy
	//triu(R)
#ifdef MKL
	LAPACK_DPOTRF(LAPACK_ROW_MAJOR, 'U',d, phi,d);
#else
	LAPACK_DPOTRF( CblasRowMajor, CblasUpper,d, phi,d);
#endif
		// R = chol(phi)
	// R'* X_rand
	// M(d) N(d), alpha (1), A (phi), leading dim of A (d),B (X_rand), leading dim of B (d)
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,CblasTrans,CblasNonUnit, d,d,1.,phi,d,X_rand,d);

	// (triu(X_rand) * R' ) (triu(X_rand) * R')^T
	cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, d, d, 1, X_rand, d,0,X_out ,d);

	for( i = 0; i < d; i++)
	{
		for( ii = 0; ii < i ; ii++)
			X_out[d * i + ii] = X_out[d * ii + i];
	}

}

/*
 *
 *
 * Inverse of symmetric pos def matrix
 *
 */
void inv_c( double *X_inv, const double *X, const int d)
{
    int i,j;
    for( i=0; i < d ;i++){
	for( j=0; j < (i + 1) ;j++)
			X_inv[d * i + j] = X[d * i + j];

	}
#ifdef MKL
    LAPACK_DPOTRF(LAPACK_ROW_MAJOR, 'L',d, X_inv,d);
    LAPACK_DPOTRI(LAPACK_ROW_MAJOR, 'L',d, X_inv,d);
#else
    LAPACK_DPOTRF( CblasRowMajor, CblasLower,d, X_inv,d);
    LAPACK_DPOTRI( CblasRowMajor, CblasLower,d, X_inv,d);
#endif

	for( i = 0; i < d; i++)
	{
		for( j = 0; j < i ; j++)
			X_inv[d * j + i] = X_inv[d * i + j];
	}
	/*
	printf("X = [\n");
		for( i = 0; i < d; i++)
		{
			for( j = 0; j < d ; j++)
			{
				printf(" %.4f ",X_inv[i*d + j]);
			}
			printf("\n");
		}
		printf("]\n");
		*/
}



#ifdef __cplusplus
}
#endif
