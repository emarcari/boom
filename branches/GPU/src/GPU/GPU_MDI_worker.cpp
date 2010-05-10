/*
 * GPU_MDI_worker.cpp
 *
 *  Created on: Apr 10, 2010
 *      Author: msuchard
 */

#define CPU_PARALLEL
#define UNIFY_RNG
#define LOG_TEST

//#define DO_CPU

#define DO_GPU
#define REDUCE_ON_GPU
#define MT_ON_GPU

#include "GPU_MDI_worker.hpp"
#include "Models/Glm/PosteriorSamplers/MLVS.hpp"

#include <boost/ref.hpp>
#include <boost/cast.hpp>

#include <cpputil/math_utils.hpp>
#include <cpputil/lse.hpp>
#include <stats/logit.hpp>
#include <distributions.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "GPU_MDI_worker_kernel.h"
#include "MersenneTwister.h"
#include "MersenneTwister_kernel.h"

#define SAFE_CUDA(call,ptr)		cudaError_t error = call; \
								if( error != 0 ) { \
									cerr << "CUDA Error " << cudaGetErrorString(error) << endl; \
									exit(-1); \
								}

using namespace std;

namespace BOOM {

namespace mlvs_impute {

typedef GPU_MDI_worker GMDIW;

GPU_MDI_worker::GPU_MDI_worker(Ptr<MLogitBase> mod, Ptr<MlvsCdSuf> s,
		uint Thread_id, uint Nthreads, uint device) :
	MDI_worker(mod, s, Thread_id, Nthreads) {

#ifdef DO_GPU
	initializeGPU(device); // TODO All initializers should throw exception if error
#endif

	initializeData();

	initializeInternalMemory();

	initializeMixturePrior();

	initializeOutProducts();

#ifdef MT_ON_GPU
    loadMTGPU("data/MersenneTwister.dat");
#endif

}

void GMDIW::initializeData() {
	// Initialize data
	nChoices = mlm->Nchoices();
	nNonZeroChoices = nChoices - 1; // For eta, U, etc., only need eta, u, etc., for y > 0
	nPredictors = mlm->beta_size() / nChoices;

	const std::vector<Ptr<ChoiceData> > & dat(mlm->dat());
	uint nTotalData = dat.size();
	uint i = thread_id;
	std::vector<std::vector<Real> > allX(nChoices);
	while (i < nTotalData) {
		Ptr<ChoiceData> dp(dat[i]);
		hY.push_back(dp->value());
		i += nthreads;
	}
	dataChuckSize = hY.size();
	paddedDataChuckSize = dataChuckSize;
	uint remainder = paddedDataChuckSize % 16;
	if (remainder != 0) {
		paddedDataChuckSize += 16 - remainder;
		for (uint i = dataChuckSize; i < paddedDataChuckSize; i++) {
			hY.push_back(0);
		}
	}
	betaSize = nPredictors * nNonZeroChoices;
	paddedBetaSize = betaSize; // TODO make multiple of 16 ?
	priorMixtureSize = post_prob_.size();

	cerr << "dataChuckSize = " << dataChuckSize << endl;
	cerr << "paddedDataChuckSize = " << paddedDataChuckSize << endl;
	cerr << "nChoices = " << nChoices << endl;
	cerr << "nPredictors = " << nPredictors << endl;
	cerr << "betaSize = " << betaSize << endl;
	cerr << "paddedBetaSize = " << paddedBetaSize << endl;
	cerr << "mixture size = " << priorMixtureSize << endl;

	hX = (Real*) calloc(sizeof(Real),
			paddedDataChuckSize	* nPredictors); // temporary contiguous host memory to hold design matrix

//	hXt = (Real*) calloc(sizeof(Real),
//			paddedDataChuckSize * nPredictors);

	for (uint index = 0; index < dataChuckSize; index++) {
		uint datumIndex = thread_id + index * nthreads;
		Ptr<ChoiceData> dp(dat[datumIndex]);
		const Mat &datumX(dp->X());
		const double* oldDatum = datumX.data();
		for (uint k = 0; k < nPredictors; k++) {
			hX[index + paddedDataChuckSize * k] = (Real) oldDatum[nChoices * k];
//			hXt[k + nPredictors * index] = (Real) oldDatum[nChoices * k];
		}
	}
	for (uint index = dataChuckSize; index < paddedBetaSize; index++) {
		for (uint k = 0; k < nPredictors; k++) {
			hX[index + paddedDataChuckSize * k] = 0;
		}
	}

	// Load data onto GPU
#ifdef DO_GPU
	dY = (uint*) allocateGPUMemory(sizeof(uint) * paddedDataChuckSize);
	dX = (Real*) allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nPredictors);

	cudaMemcpy(dY, &hY[0], sizeof(uint) * paddedDataChuckSize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(dX, hX, sizeof(Real) * paddedDataChuckSize * nPredictors,
			cudaMemcpyHostToDevice);
#endif

}

void GMDIW::initializeInternalMemory() {

	nRandomNumbers = paddedDataChuckSize * (2 * nChoices + 1);
#ifdef MT_ON_GPU
	uint remainder = nRandomNumbers % MT_RNG_COUNT;
	if (remainder != 0) {
		nRandomNumbers += MT_RNG_COUNT - remainder;
	}
#endif

	hBeta = (Real*) calloc(sizeof(Real), paddedBetaSize);
	hEta = (Real*) calloc(sizeof(Real), paddedDataChuckSize * nNonZeroChoices);
	hLogZMin = (Real*) malloc(sizeof(Real) * paddedDataChuckSize);
	hRng = (Real*) malloc(sizeof(Real) * paddedDataChuckSize * (2 * nChoices + 1) ); // Two draws per datum
	hU = (Real*) malloc(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
	hK = (uint*) malloc(sizeof(uint) * paddedDataChuckSize * nNonZeroChoices);
	hWeight = (Real*) malloc(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);

#ifdef DO_GPU
	dBeta = (Real*) allocateGPUMemory(sizeof(Real) * paddedBetaSize);
	dEta = (Real*) allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
	dLogZMin = (Real*) allocateGPUMemory(sizeof(Real) * paddedDataChuckSize);
	dRng = (Real*) allocateGPUMemory(sizeof(Real) * nRandomNumbers);
	dU = (Real*) allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
	dWeight = (Real*) allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
#endif
}

void GMDIW::initializeMixturePrior() {

	// Temporary space for prior mixture information
	uint priorMemorySize = sizeof(Real) * priorMixtureSize;
	hMu = (Real*) malloc(priorMemorySize);
	hStd = (Real*) malloc(priorMemorySize);
	hLogPriorWeight = (Real*) malloc(priorMemorySize);
	hPostWeight = (Real*) malloc(priorMemorySize);
	hSigmaSqInv = (Real*) malloc(priorMemorySize);

	for (uint k = 0; k < priorMixtureSize; k++) {
		hMu[k] = (Real) mu_[k];
		hStd[k] = (Real) sd_[k];
		hLogPriorWeight[k] = (Real) logpi_[k];
		hSigmaSqInv[k] = (Real) sigsq_inv_[k];
	}

#ifdef DO_GPU

#ifdef USE_CONSTANT

	gpuLoadConstantMemory(hMu, hSigmaSqInv, hLogPriorWeight, priorMemorySize);
//	cudaMemcpyToSymbol(cMu, hMu, priorMemorySize, 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(cLogPrior, hLogPriorWeight, priorMemorySize, 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(cPrec, hSigmaSqInv, priorMemorySize, 0, cudaMemcpyHostToDevice);

#else
	dMu = (Real*) allocateGPUMemory(priorMemorySize);
	dLogPriorWeight = (Real*) allocateGPUMemory(priorMemorySize);
	dSigmaSqInv = (Real*) allocateGPUMemory(priorMemorySize);

	cudaMemcpy(dMu, hMu, priorMemorySize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(dLogPriorWeight, hLogPriorWeight, priorMemorySize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(dSigmaSqInv, hSigmaSqInv, priorMemorySize,
			cudaMemcpyHostToDevice);
#endif
#endif
}

GMDIW::~GPU_MDI_worker() {
	if (hX) {
		free(hX);
	}
	if (hBeta) {
		free(hBeta);
	}
	if (hEta) {
		free(hEta);
	}
	if (hLogZMin) {
		free(hLogZMin);
	}
	if (hMu) {
		free(hMu);
	}
	if (hStd) {
		free(hStd);
	}
	if (hLogPriorWeight) {
		free(hLogPriorWeight);
	}
	if (hPostWeight) {
		free(hPostWeight);
	}
	if (hSigmaSqInv) {
		free(hSigmaSqInv);
	}
	if (hWeight) {
		free(hWeight);
	}
	if (hXtX) {
		free(hXtX);
	}
	if (hTmp) {
		free(hTmp);
	}
	// TODO Free all CUDA memory, should be released when context goes out of scope
}

void GMDIW::initializeOutProducts() {

#ifdef DO_CPU
	// Precompute all cross terms for speed
	uint nTerms = nPredictors * (nPredictors + 1) / 2;
//	hXtX = (Real*) malloc(sizeof(Real) * paddedDataChuckSize * nTerms);
//	for (uint index = 0; index < paddedDataChuckSize; index++) {
//		uint term = 0;
//		for (uint i = 0; i < nPredictors; i++) {
//			Real xi = hX[index + paddedDataChuckSize * i];
//			for (uint j = i; j < nPredictors; j++) {
//				Real xj = hX[index + paddedDataChuckSize *j];
//				hXtX[index + paddedDataChuckSize * term] = xi * xj;
//				term++;
//			}
//		}
//	}
#endif

#ifdef DO_GPU
#ifdef REDUCE_ON_GPU
	const uint dim = nNonZeroChoices * nPredictors;
	const uint dim2 = dim * (nPredictors + 1) / 2;

	dXtX = (Real*) allocateGPUMemory(sizeof(Real) * (dim2 + dim)); // * nXtXReducedRows);
	dXWU = dXtX + dim2;

	hTmp = (Real*) malloc(sizeof(Real) * (dim + dim * dim));
#endif
#endif
}

void GMDIW::computeWeightedOuterProducts() {

	const uint dim = nPredictors * nNonZeroChoices;
	const uint dim2 = dim * (nPredictors + 1) / 2;

	Spd totalWeightedXXt(dim);
	std::vector<double> totalWeightedUtility(dim);

#ifdef REDUCE_ON_GPU
#ifdef DO_GPU
	gpuReduceXtWX(dXtX,
			dX,dWeight, 0, nXtXReducedRows, nNonZeroChoices,
			paddedDataChuckSize, nPredictors);

	gpuReduceXWU(dXWU,
			dX, dU, dWeight, nXWUReducedRows, nNonZeroChoices,
			paddedDataChuckSize, nPredictors);

	cudaMemcpy(hTmp, dXtX, sizeof(Real) * (dim2 + dim),
			cudaMemcpyDeviceToHost);

	uint index = 0;
	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		uint offset = choice * nPredictors;
		for (uint i = 0; i < nPredictors; i++) {
			for (uint j = i; j < nPredictors; j++) {
				totalWeightedXXt(offset + i, offset + j) = hTmp[index];
				totalWeightedXXt(offset + j, offset + i) = hTmp[index];
				index++;
			}
		}
	}

	const Real* tmp = hTmp  + dim2;
	for (uint i = 0; i < dim; ++i) {
		totalWeightedUtility[i] = tmp[i];
	}
#endif
#else
#ifdef DO_GPU
	cudaMemcpy(hU, dU, sizeof(Real) * paddedDataChuckSize * nNonZeroChoices,
			cudaMemcpyDeviceToHost);
	cudaMemcpy(hWeight, dWeight, sizeof(Real) * paddedDataChuckSize * nNonZeroChoices,
			cudaMemcpyDeviceToHost);
#endif
	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		uint offset = (choice - 0) * nPredictors;
		const Real* thisWeight = hWeight + choice * paddedDataChuckSize;


		// Compute outer products
//		uint termTriangle = 0;
		for (uint i = 0; i < nPredictors; i++) {
			for (uint j = 0; j < nPredictors; j++) {
				Real sum = 0;
				if (j >= i) {
//					const Real* thisXtX = hXtX + paddedDataChuckSize * termTriangle;
					const Real* thisXi = hX + paddedDataChuckSize * i;
					const Real* thisXj = hX + paddedDataChuckSize * j;
					for (uint index = 0; index < dataChuckSize; index++) {
//						sum += thisXtX[index] * thisWeight[index];
						sum += thisXi[index] * thisXj[index] * thisWeight[index];
					}
//					termTriangle++;
				}
				totalWeightedXXt(offset + i, offset + j) = sum;
			}
		}

		// Compute weighted utility
		const Real* thisUtility = hU + choice * paddedDataChuckSize;
		for (uint i = 0; i < nPredictors; i++) {
			const Real* thisX = hX + paddedDataChuckSize * i;
			Real sum = 0;
			for (uint index = 0; index < dataChuckSize; index++) {
				sum += thisX[index] * thisUtility[index] * thisWeight[index];
			}
			totalWeightedUtility[offset + i] = sum;
		}
	}
#endif

	Ptr<MlvsCdSuf_ml> news(suf_.dcast<MlvsCdSuf_ml>());
	news->clear();
	news->update(totalWeightedXXt, totalWeightedUtility);
//	printfVector(&totalWeightedUtility[0], dim);
//	if (stop) {
//		exit(0);
//	}
//	cerr << endl;
//	cerr << "SUF = " << endl << news->xtwx() << endl;
//	cerr << "SUF = " << news->xtwu() << endl;
//	cerr << endl;
//	exit(0);
}

void GMDIW::uploadBeta() {
	MultinomialLogitModel *ptrMLM = dynamic_cast<MultinomialLogitModel*>(mlm.dumb_ptr());
	const double* modelBeta = ptrMLM->beta().data();
	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		for (uint k = 0; k < nPredictors; k++) {
			hBeta[choice * nPredictors + k] = modelBeta[choice * nPredictors + k];
		}
	}

#ifdef DO_GPU
	cudaMemcpy(dBeta, hBeta, sizeof(Real) * nPredictors * nNonZeroChoices,
			cudaMemcpyHostToDevice);
#endif
}

void GMDIW::generateRngNumbers() {

#ifdef MT_ON_GPU
	uint seed = 0;
	while (seed <= 2) {
		double u = runif_mt(rng) * std::numeric_limits<int>::max();
		seed = lround(u);
	}
	seedMTGPU(seed);
	gpuRandomMT(dRng, nRandomNumbers);
#else
	for (uint index = 0; index < dataChuckSize; index++) {
		for (uint k = 0; k < (2 * nChoices + 1); k++) { // TODO Need any fewer!
			hRng[k * paddedDataChuckSize + index] = runif_mt(rng);
		}
	}
	for (uint index = dataChuckSize; index < paddedDataChuckSize; index++) {
		for (uint k = 0; k < (2 * nChoices + 1); k++) {
			hRng[k * paddedDataChuckSize + index] = 0.5; // Just avoid denormalized numbers
		}
	}
#ifdef DO_GPU
	cudaMemcpy(dRng, hRng, sizeof(Real) * nRandomNumbers,
			cudaMemcpyHostToDevice);
#endif // DO_GPU
#endif // MT_ON_GPU
}

void GMDIW::computeEta() {
#ifdef DO_CPU
	for (uint index = 0; index < dataChuckSize; index++) {
		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
			Real sum = 0;
			uint rowOffset = paddedDataChuckSize * choice + index;
			for (uint k = 0; k < nPredictors; k++) {
				sum += hX[index + paddedDataChuckSize * k] *
						hBeta[nPredictors * choice + k];
			}
			// TODO Adjust for down sampling
			hEta[rowOffset] = sum;
		}
	}
#endif

#ifdef DO_GPU
	assert(nNonZeroChoices * nPredictors <= COMPUTE_ETA_DATA_BLOCK_SIZE);
	gpuComputeEta(dEta, dX, dBeta, paddedDataChuckSize, nNonZeroChoices, nPredictors);
#endif
}

void GMDIW::reduceEta() {
#ifdef DO_CPU
	for (uint index = 0; index < dataChuckSize; index++) {
		Real sum = 1;
		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
			sum += exp(hEta[paddedDataChuckSize * choice + index]);
		}
#ifdef UNIFY_RNG
		hLogZMin[index] = sum;
#else
		hLogZMin[index] = log(-log(hRng[index])) - log(sum); // TODO Minimize logs
#endif
	}
#endif

#ifdef DO_GPU
	gpuReduceEta(dLogZMin, dEta, dRng, paddedDataChuckSize, nNonZeroChoices);
#endif
}

void GMDIW::sampleAllU() {
#ifdef DO_CPU
	for (uint index = 0; index < dataChuckSize; index++) {
		uint y = hY[index];
#ifdef UNIFY_RNG
#ifdef LOG_TEST
		Real zmin = -log(hRng[index]) / hLogZMin[index];
#else
		Real logzmin = log(-log(hRng[index])) - log(hLogZMin[index]);
#endif
#else
	 	Real logzmin = hLogZMin[index];
#endif
		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
			uint thread = index + paddedDataChuckSize * choice;
#ifdef LOG_TEST
			Real z = zmin;
#else
			Real logz;
#endif
			if ((choice + 1) != y) {
#ifdef LOG_TEST
				z += - log(hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 1)])
				              / exp(hEta[thread]);
			}
			Real minusLogZ = -log(z);
#else
				Real tmp = log(-log(hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 1)])) // Convoluted indices to match with serial version
						- hEta[thread];
				logz = -lse2(logzmin, tmp);
			} else {
				logz = -logzmin;
			}
			hU[thread] = logz; // No need to store here; store once later
#endif

			// sample single U
			Real unif = hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 2)];
#ifdef LOG_TEST
			uint K = sampleOneU(minusLogZ - hEta[thread], unif); // No need to store
			hU[thread] = minusLogZ - hMu[K];
#else
			uint K = sampleOneU(hU[thread] - hEta[thread], unif); // No need to store
			hU[thread] -= hMu[K]; // Just store here once
#endif
			hWeight[thread] = hSigmaSqInv[K];
		}
	}
#endif

#ifdef DO_GPU
	assert(priorMixtureSize < 16);
	gpuSampleAllU(dU, dWeight, dY, dEta, dLogZMin, dRng,
			dMu, dSigmaSqInv, dLogPriorWeight,
			paddedDataChuckSize, nNonZeroChoices, priorMixtureSize);
//	printfCudaVector(dU, 100);
//	exit(0);
#endif
}

uint GMDIW::sampleOneU(Real x, Real unif) {
	Real sum = 0;
	for (uint k = 0; k < priorMixtureSize; k++) {
		Real inc = exp(hLogPriorWeight[k] + dnorm(x, hMu[k], hStd[k], true));
		hPostWeight[k] = inc;
		sum += inc;
	}
	sum *= unif;
	uint K = 0;
	while (sum > hPostWeight[K]) {
		sum -= hPostWeight[K];
		K++;
	}
	return K;
}

void GMDIW::impute_u(Ptr<ChoiceData> dp, uint index) {

	mlm->fill_eta(dp, eta); // eta+= downsampling_logprob
	if (downsampling_)
		eta += log_sampling_probs_; //
	uint M = mlm->Nchoices();
	uint y = dp->value();
	assert(y<M);

	double loglam = lse(eta);
	double logzmin = rlexp_mt(rng, loglam);

	u[y] = -logzmin;
	for (uint m = 0; m < M; ++m) {
		if (m != y) {
			double tmp = rlexp_mt(rng, eta[m]);
			double logz = lse2(logzmin, tmp);
			u[m] = -logz;
		} else {
			double tmp = rlexp_mt(rng, 0.0); // Make access to random numbers regular
		}
		uint k = unmix(u[m] - eta[m]);
		u[m] -= mu_[k];
		wgts[m] = sigsq_inv_[k];
	}
}

//----------------------------------------------------------------------

uint GMDIW::unmix(double u) {
	uint K = post_prob_.size();
	for (uint k = 0; k < K; ++k)
		post_prob_[k] = logpi_[k] + dnorm(u, mu_[k], sd_[k], true);
	post_prob_.normalize_logprob();
	return rmulti_mt(rng, post_prob_);
}

//----------------------------------------------------------------------

void GMDIW::operator()() {

#ifdef CPU_PARALLEL
	uploadBeta();
	generateRngNumbers();
	computeEta();
	reduceEta();
	sampleAllU();
	computeWeightedOuterProducts();
#else
	const std::vector<Ptr<ChoiceData> > & dat(mlm->dat());
	suf_->clear();
	uint n = dat.size();
	uint i = thread_id;
	uint index = 0;
	while (i < n) {
		Ptr<ChoiceData> dp(dat[i]);
		dp->set_wsp(thisX);
		impute_u(dp, index);
		suf_->update(dp, wgts, u);
		i += nthreads;
		index++;
	}
	Ptr<MlvsCdSuf_ml> news(suf_.dcast<MlvsCdSuf_ml>());
//	cerr << "oldSUF: " << endl << news->xtwx() << endl;
//	cerr << "oldSUF: " << news->xtwu() << endl << endl;
//	exit(0);
#endif
}



int GMDIW::getGPUDeviceCount() {
	int cDevices;
	CUresult status;
	status = cuInit(0);
	if (CUDA_SUCCESS != status)
		return 0;
	status = cuDeviceGetCount(&cDevices);
	if (CUDA_SUCCESS != status)
		return 0;
	return cDevices;
}

void GMDIW::printGPUInfo(int iDevice) {

	fprintf(stderr,"GPU Device Information:");

		char name[256];
		int totalGlobalMemory = 0;
		int clockSpeed = 0;

		// New CUDA functions in cutil.h do not work in JNI files
		getGPUInfo(iDevice, name, &totalGlobalMemory, &clockSpeed);
		fprintf(stderr,"\nDevice #%d: %s\n",(iDevice+1),name);
		double mem = totalGlobalMemory / 1024.0 / 1024.0;
		double clo = clockSpeed / 1000000.0;
		fprintf(stderr,"\tGlobal Memory (MB) : %3.0f\n",mem);
		fprintf(stderr,"\tClock Speed (Ghz)  : %1.2f\n",clo);
}

void GMDIW::getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed) {
	cudaDeviceProp deviceProp;
	memset(&deviceProp, 0, sizeof(deviceProp));
	cudaGetDeviceProperties(&deviceProp, iDevice);
	*oMemory = deviceProp.totalGlobalMem;
	*oSpeed = deviceProp.clockRate;
	strcpy(oName, deviceProp.name);
}

void GMDIW::initializeGPU(int device) {

	cout << "Attempting to initialize GPU device(s)..." << endl;
	int totalNumDevices = getGPUDeviceCount();
	if (totalNumDevices == 0) {
		cerr << "No GPU devices found!" << endl;
		exit(-1); // TODO Throw exception
	}

	if (totalNumDevices <= device) {
		cerr << "Fewer than " << (device + 1) << " devices found!" << endl;
		exit(-1); // TODO Throw exception
	}
	printGPUInfo(device);
	cudaSetDevice(device);
	cout << "Device enabled!" << endl;
}

void* GMDIW::allocateGPUMemory(size_t size) {
	void* ptr;
	SAFE_CUDA(cudaMalloc((void**) &ptr, size), ptr);
	if (ptr == NULL) {
		cerr << "Failed to allocate " << size << " bytes of memory on device!" << endl;
		exit(-1); // TODO Throw exception
	}
	return ptr;
}

template <class RealType>
void GMDIW::printfVector(RealType *hPtr, uint length) {
	if (length > 0) {
		cout << hPtr[0];
	}
	for (uint i = 1; i < length; i++) {
		cout << " " << hPtr[i];
	}
	cout << endl;
}

template <class RealType>
void GMDIW::printfCudaVector(RealType* dPtr, uint length) {

	RealType* hPtr = (RealType *) malloc(sizeof(RealType) * length);
	SAFE_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(RealType)*length, cudaMemcpyDeviceToHost),dPtr);
	printfVector(hPtr, length);
	free(hPtr);
}

//----------------------------------------------------------------------

} // mlvs_impute

} // BOOM
