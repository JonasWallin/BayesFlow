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
// http://blue.for.msu.edu/comp-notes/
#define LAPACK_DPOTRF dpotrf_
#define LAPACK_DPOTRI dpotri_

#endif
#endif
#include <stdlib.h> 
#include <string.h>
#include <stdio.h>


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



void update_mu_Q_sample2(double* mu_sample, double *Q_sample, const double* Q_pmu_p, const double* Q_i,
        const double* Q_p, const double* Y_i, const double* B_i, const long int n, const int d, const int m)

{	/*
	* Internal function for improving speed in sampling posterior distribution
	* computes the Cholesky factor L = (Q_p + \sum B_i^T Q_i B_i)
	* also computes  L\ ( Q_pmu_p + \sum B_i^T Q_i * Y_i )
	*
	*	Q_j ( n * d * d x 1) -  the data matrix   from a (d x d x n) numpy vector c ordering
	*	Y_j ( n * m     x 1) -  the observations, from a (n x m x 1) numpy vector c ordering
	*	B_j ( n * m * d x 1) -  the covariates,   from a (m x d x n) numpy vector c ordering
	*/

	// loop over n
	// compute B_i^T Q_i
	// compute Q as in update_mu_Q_sample using (B_i^T Q_i) B_i
	// compute (B_i^T Q_i) Y_i

}


void update_mu_Q_sample(double* mu_sample, double *Q_sample, const double* Q_pmu_p, const double* Q,
		                const double* Q_p, const double* sumY, const long int n, const int d)
{
	/*
	* Internal function for improving speed in sampling posterior distribution
	* computes the Cholesky factor L = (Q_p + n * Q)
	* also computes  L\ ( Q_pmu_p + Q * sumY)
	*
	*/
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
#elif ATL_INT
	LAPACK_DPOTRF( CblasRowMajor, CblasLower,d, Q_sample,d);
#else
    char lower[] = "L";
    int  lda = d, d_ = d;
    int info_;
    /* arguments:
	 storing upper lower
	 order of matrix (?)
	 d x d matrix
	 leading dimension (?)
	 info
    */
    LAPACK_DPOTRF( lower, &d_, Q_sample, &lda, &info_);

#endif
	// ..., d (order of Q), Q, d (lda leading dimension), X , increment X
	cblas_dtrsv(CblasRowMajor,CblasLower,CblasNoTrans,CblasNonUnit, d,Q_sample,d,mu_sample,1);

	//DPPTRF
	//cblas_dtrsm
}

void Lt_XpZ(const double* L, double* X, const double* Z, const int d)
{
	/*
	 * Computes L'\(X + Z)
	 * L - (d x d) RowMajor lower triangular cholesky decomp
	 * X - (d x 1) vector
	 * Z - (d x 1) vector
	 */
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

	//choleksy
	//triu(R)
#ifdef MKL
	LAPACK_DPOTRF(LAPACK_ROW_MAJOR, 'U',d, phi,d);
#elif ATL_INT
	LAPACK_DPOTRF( CblasRowMajor, CblasUpper,d, phi,d);
#else
	// using Lower since this corresponds to Upper with colum mayor!!
    char lower[] = "L";
    int  lda = d, d_ = d;
    int info_;
    /* arguments:
	 storing upper lower
	 order of matrix (?)
	 d x d matrix
	 leading dimension (?)
	 info
    */
    LAPACK_DPOTRF( lower, &d_, phi, &lda, &info_);
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
void inv_c( double *X_inv, const double *X,const int d)
{
    int i,j;



    for( i=0; i < d ;i++){
    	for( j=0; j < (i+1) ;j++)
			X_inv[d * i + j] = X[d * i + j];

	}


#ifdef MKL
    LAPACK_DPOTRF(LAPACK_ROW_MAJOR, 'L',d, X_inv,d);
    LAPACK_DPOTRI(LAPACK_ROW_MAJOR, 'L',d, X_inv,d);
#elif ATL_INT
    LAPACK_DPOTRF( CblasRowMajor, CblasLower,d, X_inv, d);
    LAPACK_DPOTRI( CblasRowMajor, CblasLower,d, X_inv, d);
#else

	// using Upper since this corresponds to Lower with colum mayor!!
    char lower[] = "U";
    int  lda = d, d_ = d;
    int info_;
    /* arguments:
	 storing upper lower
	 order of matrix (?)
	 d x d matrix
	 leading dimension (?)
	 info
    */
	dpotrf_( lower, &d_, X_inv, &lda, &info_);
    dpotri_( lower, &d_, X_inv, &lda, &info_);

#endif



	for( i = 0; i < d; i++)
	{
		for( j = 0; j < i ; j++)
			X_inv[d * j + i] = X_inv[d * i + j];
	}


}



#ifdef __cplusplus
}
#endif
