/*
 * GPU_MDI_worker_kernel.cu
 *
 *  Created on: Apr 15, 2010
 *      Author: msuchard
 */

#include "GPU_MDI_worker_kernel.h"

#define ACCURATE_TIMING

__global__
void kernelComputeEta(
		REAL* eta,
		REAL* X,
		REAL* beta,
		uint nData,
		uint nChoices,
		uint nPredictors) {

	__shared__ REAL sX[COMPUTE_ETA_DATA_BLOCK_SIZE];
	__shared__ REAL sBeta[COMPUTE_ETA_DATA_BLOCK_SIZE];

	uint offsetData = blockIdx.x * COMPUTE_ETA_DATA_BLOCK_SIZE + threadIdx.x;
	uint offsetChoice = blockIdx.y * COMPUTE_ETA_CHOICE_BLOCK_SIZE + threadIdx.y;

	if (threadIdx.x < nPredictors * nChoices) {
		sBeta[threadIdx.x] = beta[threadIdx.x]; // TODO Rewrite for larger #s of predictors
	}

	REAL sum = 0;

	for (uint i = 0; i < nPredictors; i++) {

		if (threadIdx.y == 0) {
			sX[threadIdx.x] = X[nData * i + offsetData];
		}
		__syncthreads();

		sum += sX[threadIdx.x] * sBeta[offsetChoice * nPredictors + i];
	}

	if (offsetData < nData && offsetChoice < nChoices) {
		eta[offsetChoice * nData + offsetData] = sum;
	}
}

cudaError_t gpuComputeEta(
		REAL* eta,
		REAL* X,
		REAL* beta,
		uint nData,
		uint nChoices,
		uint nPredictors) {

	dim3 gridComputeEta(nData / COMPUTE_ETA_DATA_BLOCK_SIZE,
			nChoices / COMPUTE_ETA_CHOICE_BLOCK_SIZE);
	if (nData % COMPUTE_ETA_DATA_BLOCK_SIZE != 0) {
		gridComputeEta.x += 1;
	}
	if (nChoices % COMPUTE_ETA_CHOICE_BLOCK_SIZE != 0) {
		gridComputeEta.y += 1;
	}
	dim3 blockComputeEta(COMPUTE_ETA_DATA_BLOCK_SIZE, COMPUTE_ETA_CHOICE_BLOCK_SIZE);
	kernelComputeEta<<<gridComputeEta, blockComputeEta>>>(eta, X, beta, nData, nChoices, nPredictors);

#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif

	return cudaSuccess;
}

cudaError_t gpuComputeEta_new(
		cublasHandle_t handle,
		REAL* eta,
		REAL* X,
		REAL* beta,
		uint nData,
		uint nBeta) {

//     cublasSgemv(handle, 
    exit(-1);

#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif		
    return cudaSuccess;		
}		

__global__
void kernelReduceEta(
		REAL* reduced,
		REAL* eta,
		REAL* rng,
		uint nData,
		uint nChoices) {

	uint offsetData = blockIdx.x * REDUCE_ETA_BLOCK_SIZE + threadIdx.x;
	REAL sum = 1.0;
	for (uint i = 0; i < nChoices; i++) {
		sum += exp(eta[i * nData + offsetData]);
	}

	if (offsetData < nData) {
//		reduced[offsetData] = log(-log(rng[offsetData])) - log(sum); // TODO Minimize logs
		reduced[offsetData] = -log(rng[offsetData]) / sum;
//		reduced[offsetData] = sum;

	}
}

__global__
void kernelReduceEta_new(
		REAL* reduced,
		REAL* eta,
		REAL* rng,
		uint nData,
		uint nChoices) {

	uint offsetData = blockIdx.x * REDUCE_ETA_BLOCK_SIZE + threadIdx.x;
	REAL sum = 0.0;
	for (uint i = 0; i < nChoices; i++) {
		sum += exp(eta[i * nData + offsetData]);
	}

	if (offsetData < nData) {
 		reduced[offsetData] = -log(rng[offsetData]) / sum;
//		reduced[offsetData] = sum; // Used for debugging
	}
}

cudaError_t gpuReduceEta(
		REAL* reduced,
		REAL* eta,
		REAL* rng,
		uint nData,
		uint nChoices) {

	dim3 gridReduceEta(nData / REDUCE_ETA_BLOCK_SIZE);
	if (nData % REDUCE_ETA_BLOCK_SIZE != 0) {
		gridReduceEta.x += 1;
	}
	dim3 blockReduceEta(REDUCE_ETA_BLOCK_SIZE);
	kernelReduceEta<<<gridReduceEta, blockReduceEta>>>(reduced, eta, rng, nData, nChoices);

#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif

	return cudaSuccess;
}

cudaError_t gpuReduceEta_new(
		REAL* reduced,
		REAL* eta,
		REAL* rng,
		uint nData,
		uint nChoices) {

	dim3 gridReduceEta(nData / REDUCE_ETA_BLOCK_SIZE);
	if (nData % REDUCE_ETA_BLOCK_SIZE != 0) {
		gridReduceEta.x += 1;
	}
	dim3 blockReduceEta(REDUCE_ETA_BLOCK_SIZE);
 	kernelReduceEta_new<<<gridReduceEta, blockReduceEta>>>(reduced, eta, rng, nData, nChoices);

#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif

	return cudaSuccess;
}

__device__
inline REAL kernelLogDF(
		REAL x,
		REAL mu,
		REAL prec) {
	REAL delta = x - mu;
	return 0.5 * (log(prec) - prec * delta * delta);
}

//#define NO_DIVERGENCE1
//#define NO_DIVERGENCE2

__device__
uint kernelOneU(
		REAL x,
		REAL unif,
		REAL* mu,
		REAL* prec,
		REAL* logP,
		uint nMixture) {

	REAL sum = 0;
	for (uint k = 0; k < nMixture; k++) {
		sum += exp(logP[k] + kernelLogDF(x, mu[k], prec[k]));
	}
	sum *= unif;
	int K = -1;
#ifdef NO_DIVERGENCE1
	for (int k = 0; k < nMixture; k++) {
		if (sum > 0) {
			K++;
		}
		sum -= exp(logP[k] + kernelLogDF(x, mu[k], prec[k]));
	}
#else
	while (sum > 0) {
		K++;
		sum -= exp(logP[K] + kernelLogDF(x, mu[K], prec[K]));
	}
#endif
	return (uint) K;
}

#ifdef USE_CONSTANT
#define MU_PTR			cMu
#define PREC_PTR		cPrec
#define LOGPRIOR_PTR	cLogPrior
#else
#define MU_PTR 			sMu
#define PREC_PTR		sPrec
#define LOGPRIOR_PTR	sLogPrior
#endif

#define MU(x)		MU_PTR[x]
#define PREC(x)		PREC_PTR[x]
#define LOGPRIOR(x)	LOGPRIOR_PTR[x]

#ifdef USE_CONSTANT
	__constant__ REAL cMu[16];
	__constant__ REAL cPrec[16];
	__constant__ REAL cLogPrior[16];

	cudaError_t gpuLoadConstantMemory(
			REAL* hMu,
			REAL* hPrec,
			REAL* hLogPrior,
			size_t length) {
		cudaMemcpyToSymbol(cMu, hMu, length, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cLogPrior, hLogPrior, length, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cPrec, hPrec, length, 0, cudaMemcpyHostToDevice);

		return cudaSuccess;
	}
#endif

#define HAND_UNROLL

#ifdef HAND_UNROLL
#define COMPUTE(k)	REAL v##k = exp(LOGPRIOR(k) + kernelLogDF(x, MU(k), PREC(k)));
#endif

__global__
void kernelSampleAllU(
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
		uint nMixtures) {

	uint offsetData = blockIdx.x * SAMPLE_U_DATA_BLOCK_SIZE + threadIdx.x;
	uint offsetChoice = blockIdx.y * SAMPLE_U_CHOICE_BLOCK_SIZE + threadIdx.y;

#ifndef USE_CONSTANT
	__shared__ REAL sMu[16]; // Determine at run-time
	__shared__ REAL sPrec[16]; // Determine at run-time
	__shared__ REAL sLogPrior[16]; // Determine at run-time
#endif
	__shared__ REAL sZMin[SAMPLE_U_DATA_BLOCK_SIZE];
	__shared__ uint sY[SAMPLE_U_DATA_BLOCK_SIZE];

	if (threadIdx.y == 0) { // Load once per choice
#ifndef USE_CONSTANT
		if (threadIdx.x < nMixtures) {
			sMu[threadIdx.x] = mu[threadIdx.x];
			sPrec[threadIdx.x] = sigmaSqInv[threadIdx.x];
			sLogPrior[threadIdx.x] = logPriorWeight[threadIdx.x];
		}
#endif
		sY[threadIdx.x] = Y[offsetData];
		sZMin[threadIdx.x] = logZMin[offsetData];
	}
	__syncthreads();

	if (offsetData < nData) {
		REAL z = sZMin[threadIdx.x];
		REAL thisEta = eta[offsetChoice * nData + offsetData];
		REAL thisRng = rng[offsetData + nData * (2 * (offsetChoice + 1) + 1)];
#ifdef NO_DIVERGENCE2
		int indy = (sY[threadIdx.x] != (offsetChoice + 1));
		z += indy * -log(thisRng) / exp(thisEta);
#else
		if (sY[threadIdx.x] != (offsetChoice + 1)) {
			z += -log(thisRng) / exp(thisEta);
		}
#endif
		REAL minusLogZ = -log(z);
		REAL unif = rng[offsetData + nData * (2 * (offsetChoice + 1) + 2)];

#ifdef USE_CONSTANT
#ifdef HAND_UNROLL
		REAL x = minusLogZ - thisEta;
		REAL sum = 0;
		COMPUTE(0); sum += v0;
		COMPUTE(1); sum += v1;
		COMPUTE(2); sum += v2;
		COMPUTE(3); sum += v3;
		COMPUTE(4); sum += v4;
		COMPUTE(5); sum += v5;
		COMPUTE(6); sum += v6;
		COMPUTE(7); sum += v7;
		COMPUTE(8); sum += v8;
		COMPUTE(9); sum += v9;

		sum *= unif;
		int K = -1;

		if (sum > 0) { K++; }; sum -= v0;
		if (sum > 0) { K++; }; sum -= v1;
		if (sum > 0) { K++; }; sum -= v2;
		if (sum > 0) { K++; }; sum -= v3;
		if (sum > 0) { K++; }; sum -= v4;
		if (sum > 0) { K++; }; sum -= v5;
		if (sum > 0) { K++; }; sum -= v6;
		if (sum > 0) { K++; }; sum -= v7;
		if (sum > 0) { K++; }; sum -= v8;
		if (sum > 0) { K++; }; sum -= v9;

//		for (int k = 0; k < nMixture; k++) {
//			if (sum > 0) {
//				K++;
//			}
//			sum -= exp(logP[k] + kernelLogDF(x, mu[k], prec[k]));
//		}
#else
		REAL mySum[16];
		REAL sum = 0;
		REAL x = minusLogZ - thisEta;
		for (uint k = 0; k < nMixtures; k++) {
			REAL v = exp(LOGPRIOR(k) + kernelLogDF(x, MU(k), PREC(k)));
			mySum[k] = v;
			sum += v;
		}
		sum *= unif;
		int K = -1;
		while (sum > 0) {
			K++;
//			sum -= exp(LOGPRIOR(K) + kernelLogDF(x, MU(K), PREC(K)));
			sum -= mySum[K];
		}
#endif // HAND_UNROLL
#else
		uint K = kernelOneU(minusLogZ - thisEta, unif, MU_PTR, PREC_PTR, LOGPRIOR_PTR,
				nMixtures);
#endif

		U[offsetChoice * nData + offsetData] = minusLogZ - MU(K);
		weight[offsetChoice * nData + offsetData] = PREC(K);
	}
}

__global__
void kernelSampleAllU_new(
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
		uint nMixtures) {

	uint offsetData = blockIdx.x * SAMPLE_U_DATA_BLOCK_SIZE + threadIdx.x;
	uint offsetChoice = blockIdx.y * SAMPLE_U_CHOICE_BLOCK_SIZE + threadIdx.y;

#ifndef USE_CONSTANT
	__shared__ REAL sMu[16]; // Determine at run-time
	__shared__ REAL sPrec[16]; // Determine at run-time
	__shared__ REAL sLogPrior[16]; // Determine at run-time
#endif
	__shared__ REAL sZMin[SAMPLE_U_DATA_BLOCK_SIZE];
	__shared__ uint sY[SAMPLE_U_DATA_BLOCK_SIZE];

	if (threadIdx.y == 0) { // Load once per choice
#ifndef USE_CONSTANT
		if (threadIdx.x < nMixtures) {
			sMu[threadIdx.x] = mu[threadIdx.x];
			sPrec[threadIdx.x] = sigmaSqInv[threadIdx.x];
			sLogPrior[threadIdx.x] = logPriorWeight[threadIdx.x];
		}
#endif
		sY[threadIdx.x] = Y[offsetData];
		sZMin[threadIdx.x] = logZMin[offsetData];
	}
	__syncthreads();

	if (offsetData < nDataNotPadded && offsetChoice < nChoices) {
		REAL z = sZMin[threadIdx.x];
		REAL thisEta = eta[offsetChoice * nData + offsetData];
		REAL thisRng = rng[offsetData + nData * (2 * offsetChoice + 1)];
#ifdef NO_DIVERGENCE2
		int indy = (sY[threadIdx.x] != offsetChoice);
		z += indy * -log(thisRng) / exp(thisEta);
#else
		if (sY[threadIdx.x] != offsetChoice) {
			z += -log(thisRng) / exp(thisEta);
		}
#endif
		REAL minusLogZ = -log(z);
		REAL unif = rng[offsetData + nData * (2 * offsetChoice + 2)];

#ifdef USE_CONSTANT
#ifdef HAND_UNROLL
		REAL x = minusLogZ - thisEta;
		REAL sum = 0;
		COMPUTE(0); sum += v0;
		COMPUTE(1); sum += v1;
		COMPUTE(2); sum += v2;
		COMPUTE(3); sum += v3;
		COMPUTE(4); sum += v4;
		COMPUTE(5); sum += v5;
		COMPUTE(6); sum += v6;
		COMPUTE(7); sum += v7;
		COMPUTE(8); sum += v8;
		COMPUTE(9); sum += v9;

		sum *= unif;
		int K = -1;

		if (sum > 0) { K++; }; sum -= v0;
		if (sum > 0) { K++; }; sum -= v1;
		if (sum > 0) { K++; }; sum -= v2;
		if (sum > 0) { K++; }; sum -= v3;
		if (sum > 0) { K++; }; sum -= v4;
		if (sum > 0) { K++; }; sum -= v5;
		if (sum > 0) { K++; }; sum -= v6;
		if (sum > 0) { K++; }; sum -= v7;
		if (sum > 0) { K++; }; sum -= v8;
		if (sum > 0) { K++; }; sum -= v9;

//		for (int k = 0; k < nMixture; k++) {
//			if (sum > 0) {
//				K++;
//			}
//			sum -= exp(logP[k] + kernelLogDF(x, mu[k], prec[k]));
//		}
#else
		REAL mySum[16];
		REAL sum = 0;
		REAL x = minusLogZ - thisEta;
		for (uint k = 0; k < nMixtures; k++) {
			REAL v = exp(LOGPRIOR(k) + kernelLogDF(x, MU(k), PREC(k)));
			mySum[k] = v;
			sum += v;
		}
		sum *= unif;
		int K = -1;
		while (sum > 0) {
			K++;
//			sum -= exp(LOGPRIOR(K) + kernelLogDF(x, MU(K), PREC(K)));
			sum -= mySum[K];
		}
#endif // HAND_UNROLL
#else
		uint K = kernelOneU(minusLogZ - thisEta, unif, MU_PTR, PREC_PTR, LOGPRIOR_PTR,
				nMixtures);
#endif

		U[offsetChoice * nData + offsetData] = minusLogZ - MU(K);
		weight[offsetChoice * nData + offsetData] = PREC(K);
	} 
	
	else if (offsetData < nData) {
		U[offsetChoice * nData + offsetData] = 0;
		weight[offsetChoice * nData + offsetData] = 0;	
	}
}


cudaError_t gpuSampleAllU(
		REAL* dU,
		REAL* dWeight,
		uint* dY,
		REAL* dEta,
		REAL* dLogZMin,
		REAL* dRng,
		REAL* dMu,
		REAL* dSigmaSqInv,
		REAL* dLogPriorWeight,
		uint nData,
		uint nChoices,
		uint nMixtures) {

	dim3 gridSampleU(nData / SAMPLE_U_DATA_BLOCK_SIZE,
			nChoices / SAMPLE_U_CHOICE_BLOCK_SIZE);
	if (nData % SAMPLE_U_DATA_BLOCK_SIZE != 0) {
		gridSampleU.x += 1;
	}
	if (nChoices % SAMPLE_U_CHOICE_BLOCK_SIZE != 0) {
		gridSampleU.y += 1;
	}
	dim3 blockSampleU(SAMPLE_U_DATA_BLOCK_SIZE, SAMPLE_U_CHOICE_BLOCK_SIZE);
	kernelSampleAllU<<<gridSampleU, blockSampleU>>>(dU, dWeight, dY, dEta, dLogZMin, dRng, dMu, dSigmaSqInv, dLogPriorWeight, nData, nChoices, nMixtures);

#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif

	return cudaSuccess;
}

cudaError_t gpuSampleAllU_new(
		REAL* dU,
		REAL* dWeight,
		uint* dY,
		REAL* dEta,
		REAL* dLogZMin,
		REAL* dRng,
		REAL* dMu,
		REAL* dSigmaSqInv,
		REAL* dLogPriorWeight,
		uint nData,
		uint nDataNotPadded,
		uint nChoices,
		uint nMixtures) {

	dim3 gridSampleU(nData / SAMPLE_U_DATA_BLOCK_SIZE,
			nChoices / SAMPLE_U_CHOICE_BLOCK_SIZE);
	if (nData % SAMPLE_U_DATA_BLOCK_SIZE != 0) {
		gridSampleU.x += 1;
	}
	if (nChoices % SAMPLE_U_CHOICE_BLOCK_SIZE != 0) {
		gridSampleU.y += 1;
	}
	dim3 blockSampleU(SAMPLE_U_DATA_BLOCK_SIZE, SAMPLE_U_CHOICE_BLOCK_SIZE);
	kernelSampleAllU_new<<<gridSampleU, blockSampleU>>>(dU, dWeight, dY, dEta, dLogZMin, dRng, dMu, dSigmaSqInv, dLogPriorWeight, nData, nDataNotPadded, nChoices, nMixtures);

#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif

	return cudaSuccess;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

#define SDATA_XTWX(x,z,y)	sdata[y]

template <unsigned int blockSize>
__global__
void kernelReduceXtWX(
		REAL* xtwx,
		REAL* x,
		REAL* w,
		const uint totalP,
		const uint totalN) {

	const uint tid = threadIdx.x;
	const uint nPred = totalP;

    uint row = 0;
    uint delta = nPred - 1;
    uint tmp;
    for( tmp = delta; tmp < blockIdx.x; tmp += delta-- ){
        row++;
    }
    const uint pred1 = row;
    const uint pred2 = nPred + blockIdx.x - tmp - 1;

	const uint choice = blockIdx.y;
	uint i = tid;

	REAL mySum = 0;

	while (i < totalN) {

		mySum += x[pred1 * totalN + i] * x[pred2 * totalN + i] * w[choice * totalN + i];

		i += blockSize;
	}

	REAL *sdata = SharedMemory<REAL>();
	SDATA_XTWX(pred, ch, tid) = mySum;
	__syncthreads();

    if (blockSize >= 512) { if (tid < 256) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid + 256); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid + 128); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid +  64); } __syncthreads(); }

    if (tid < 32) {
    	if (blockSize >=  64) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid + 32); }
    	if (blockSize >=  32) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid + 16); }
    	if (blockSize >=  16) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid +  8); }
    	if (blockSize >=   8) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid +  4); }
    	if (blockSize >=   4) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid +  2); }
    	if (blockSize >=   2) { SDATA_XTWX(pred, ch, tid) = mySum = mySum + SDATA_XTWX(pred, ch, tid +  1); }
    }

	if (tid == 0) {
		xtwx[//choice * nPred * nPred + pred1 * nPred + pred2
		     choice * gridDim.x + blockIdx.x
		     ] = SDATA_XTWX(pred, ch, 0);
	}
}

template <unsigned int blockSize>
__global__
void kernelReduceXtWX_new(
		REAL* xtwx,
		const REAL* x,
		const REAL* w,		
		const uint totalN,
		const uint totalP,
		const uint lda) {

	const uint tid = threadIdx.x;
	const uint nPred = totalP;
	
    uint row = 0;
    uint delta = nPred - 1;
    uint tmp;
    for( tmp = delta; tmp < blockIdx.x; tmp += delta-- ){
        row++;
    }
    const uint pred1 = row;
    const uint pred2 = nPred + blockIdx.x - tmp - 1;

	// const uint choice = blockIdx.y;
	uint i = tid;

    const REAL* thisX1 = x + pred1 * totalN;
    const REAL* thisX2 = x + pred2 * totalN;
    
	REAL mySum = 0;

	while (i < totalN) {

		mySum += thisX1[i] * thisX2[i] * w[i];
		
//#define DOUBLE
#ifdef DOUBLE
		if (i + blockSize < totalN) {
		    mySum += thisX1[i + blockSize] * thisX2[i + blockSize] * w[i + blockSize];
		}
		i += blockSize * 2;
#else
        i += blockSize;
#endif
	}

	REAL *sdata = SharedMemory<REAL>();
	SDATA_XTWX(pred, ch, tid) = mySum;
	__syncthreads();	

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        volatile REAL* smem = sdata;
    	if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
    	if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
    	if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
    	if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
    	if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
    	if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

	if (tid == 0) {
	    uint addr;
//   	    addr = blockIdx.x;
	    addr = pred1 * lda + pred2;
	    xtwx[addr] = sdata[0];    
// 	    xtwx[blockIdx.x] = sdata[0];  
	}
}

cudaError_t gpuReduceXtWX(
		REAL* XtWX,
		REAL* X,
		REAL* W,
		uint px,
		uint nRows,
		uint nChoices,
		uint nData,
		uint nPredictors) {

	dim3 grid(nPredictors * (nPredictors + 1) / 2, nChoices);
	dim3 block(REDUCE_XTWX_THREADS);
	kernelReduceXtWX<REDUCE_XTWX_THREADS><<<grid, block, sizeof(REAL) * REDUCE_XTWX_THREADS>>>
	(XtWX, X, W, nPredictors, nData);
	
#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif	

	return cudaSuccess;
}

cudaError_t gpuReduceXtWX_new(
		REAL* XtWX,
		REAL* X,
		REAL* W,
		uint nChoices,
		uint nData,
		uint nPredictors,
		uint lda) {

	dim3 grid(nPredictors * (nPredictors + 1) / 2);
	dim3 block(REDUCE_XTWX_NEW_THREADS);
	kernelReduceXtWX_new<REDUCE_XTWX_NEW_THREADS><<<grid, block, sizeof(REAL) * REDUCE_XTWX_NEW_THREADS>>>
	(XtWX, X, W, nChoices * nData, nPredictors, lda);
	
#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif	

	return cudaSuccess;
}



#define SDATA_XWU(x,z,y)	sdata[y]

template <unsigned int blockSize>
__global__
void kernelReduceXWU(
		REAL* xwu,
		REAL* x,
		REAL* u,
		REAL* w,
		uint totalP,
		uint totalN) {

	const uint tid = threadIdx.x;
	const uint nPred = gridDim.x;
	const uint pred = blockIdx.x;
	const uint choice = blockIdx.y;
	uint i = tid;

	REAL mySum = 0;

	while (i < totalN) {

		mySum += x[pred * totalN + i] * u[choice * totalN + i] * w[choice * totalN + i];

		i += blockSize;
	}

	REAL *sdata = SharedMemory<REAL>();
	SDATA_XWU(pred, ch, tid) = mySum;
	__syncthreads();

    if (blockSize >= 512) { if (tid < 256) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 256); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 128); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  64); } __syncthreads(); }

    if (tid < 32) {
    	if (blockSize >=  64) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 32); }
    	if (blockSize >=  32) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 16); }
    	if (blockSize >=  16) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  8); }
    	if (blockSize >=   8) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  4); }
    	if (blockSize >=   4) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  2); }
    	if (blockSize >=   2) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  1); }
    }

	if (tid == 0) {
		xwu[choice * nPred + pred] = SDATA_XWU(pred, ch, 0);
	}
}

template <unsigned int blockSize>
__global__
void kernelReduceXtWU_new(
		REAL* xwu,
		REAL* x,
		REAL* u,
		REAL* w,
		uint totalRows) {

	const uint tid = threadIdx.x;	
	const uint pred = blockIdx.x;	
	uint i = tid;

	REAL mySum = 0;

	while (i < totalRows) {
		mySum += x[pred * totalRows + i] * u[i] * w[i];
		i += blockSize;
	}

	REAL *sdata = SharedMemory<REAL>();
	SDATA_XWU(pred, ch, tid) = mySum;
	__syncthreads();

    if (blockSize >= 512) { if (tid < 256) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 256); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 128); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  64); } __syncthreads(); }

    if (tid < 32) {
    	if (blockSize >=  64) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 32); }
    	if (blockSize >=  32) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid + 16); }
    	if (blockSize >=  16) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  8); }
    	if (blockSize >=   8) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  4); }
    	if (blockSize >=   4) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  2); }
    	if (blockSize >=   2) { SDATA_XWU(pred, ch, tid) = mySum = mySum + SDATA_XWU(pred, ch, tid +  1); }
    }

	if (tid == 0) {
		xwu[pred] = SDATA_XWU(pred, ch, 0);
	}
}

cudaError_t gpuReduceXWU(
		REAL* dXWU,
		REAL* dX,
		REAL* dU,
		REAL* dWeight,
		uint nXWUReducedRows,
		uint nChoices,
		uint nData,
		uint nPredictors) {

	dim3 grid(4, nChoices);
	dim3 block(REDUCE_XWU_THREADS);
	kernelReduceXWU<REDUCE_XWU_THREADS><<<grid, block, sizeof(REAL) * REDUCE_XWU_THREADS>>>(dXWU, dX, dU, dWeight, nChoices * nPredictors, nData);
	
#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif	

	return cudaSuccess;
}

cudaError_t gpuReduceXtWU_new(
		REAL* dXWU,
		REAL* dX,
		REAL* dU,
		REAL* dWeight,		
		uint nChoices,
		uint nData,
		uint nPredictors) {

	dim3 grid(nPredictors);
	dim3 block(REDUCE_XTWU_NEW_THREADS);
	kernelReduceXtWU_new<REDUCE_XTWU_NEW_THREADS><<<grid, block, sizeof(REAL) * REDUCE_XTWU_NEW_THREADS>>>(dXWU, dX, dU, dWeight, nChoices * nData);
	
#ifdef ACCURATE_TIMING
	cudaThreadSynchronize();
#endif	

	return cudaSuccess;
}

