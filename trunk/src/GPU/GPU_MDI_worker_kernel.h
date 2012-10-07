/*
 * GPU_MDI_worker_kernel.h
 *
 *  Created on: Apr 15, 2010
 *      Author: msuchard
 */

#ifndef GPU_MDI_WORKER_KERNEL_H_
#define GPU_MDI_WORKER_KERNEL_H_

typedef float REAL;
typedef unsigned int uint;

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define USE_CONSTANT

#define COMPUTE_ETA_DATA_BLOCK_SIZE	128
#define COMPUTE_ETA_CHOICE_BLOCK_SIZE 2 // TODO Determine at run-time

#define SAMPLE_U_DATA_BLOCK_SIZE	64
#define SAMPLE_U_CHOICE_BLOCK_SIZE	2 // TODO Determine at run-time

#define REDUCE_ETA_BLOCK_SIZE	128

#define REDUCE_XTWX_THREADS	128
#define REDUCE_XWU_THREADS	128

#define REDUCE_XTWX_NEW_THREADS	64
#define REDUCE_XTWU_NEW_THREADS	128

cudaError_t gpuComputeEta(
		REAL* eta,
		REAL* X,
		REAL* beta,
		uint nData,
		uint nChoices,
		uint nPredictors);

cudaError_t gpuComputeEta_new(
		cublasHandle_t handle,
		REAL* eta,
		REAL* X,
		REAL* beta,
		uint nData,
		uint nBeta);

cudaError_t gpuReduceEta(
		REAL* reduced,
		REAL* eta,
		REAL* rng,
		uint nData,
		uint nChoices);

cudaError_t gpuReduceEta_new(
		REAL* reduced,
		REAL* eta,
		REAL* rng,
		uint nData,
		uint nChoices);

cudaError_t gpuSampleAllU(
		REAL* U,
		REAL* weight,
		uint* Y,
		REAL* eta,
		REAL* logZMin,
		REAL* rng,
		REAL* mu,
		REAL* sigmaSqInv,
		REAL* logPriorWeight,
		uint nData,
		uint nChoices,
		uint nMixtures);

cudaError_t gpuSampleAllU_new(
		REAL* U,
		REAL* weight,
		uint* Y,
		REAL* eta,
		REAL* logZMin,
		REAL* rng,
		REAL* mu,
		REAL* sigmaSqInv,
		REAL* logPriorWeight,
		uint nData,
		uint nDataNotPadded,
		uint nChoices,
		uint nMixtures);

cudaError_t gpuReduceXtWX(
		REAL* XtWX,
		REAL* X,
		REAL* W,
		uint px,
		uint py,
		uint choice,
		uint nData,
		uint nPredictors);

cudaError_t gpuReduceXtWX_new(
		REAL* XtWX,
		REAL* X,
		REAL* W,
		uint choice,
		uint nData,
		uint nPredictors,
		uint lda);

cudaError_t gpuReduceXWU(
		REAL* dXWU,
		REAL* dX,
		REAL* dU,
		REAL* dWeight,
		uint nXWUReducedRows,
		uint nNonZeroChoices,
		uint paddedDataChuckSize,
		uint nPredictors);

cudaError_t gpuReduceXtWU_new(
		REAL* dXWU,
		REAL* dX,
		REAL* dU,
		REAL* dWeight,
		uint nChoices,
		uint paddedDataChuckSize,
		uint nPredictors);


#ifdef USE_CONSTANT
cudaError_t gpuLoadConstantMemory(
		REAL* hMu,
		REAL* hPrec,
		REAL* hLogPrior,
		size_t length);
#endif

void initMTRef(const char *fname);

#endif /* GPU_MDI_WORKER_KERNEL_H_ */
