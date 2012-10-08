/*
 * GPU_MDI_worker.cpp
 *
 *  Created on: Apr 10, 2010
 *      Author: msuchard
 */

//#define DEBUG_PRINT

//#define TIME_CUBLAS

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

namespace ComputeMode {
	int parseComputeModel(std::string option) {
		boost::to_upper(option);
		int mode;
		if (option == "GPU") {
			mode = GPU;
		} else
		if (option == "ORIGINAL") {
			mode = CPU_ORIGINAL;
		}	else
		if (option == "FLOW") {
			mode = CPU_NEW_FLOW;
		} else
		if (option == "PARALLEL") {
			mode = CPU_PARALLEL;
		} else
		if (option == "NEW") {
			mode = CPU_NEW_PARALLEL;
		} else
		if (option == "NEWGPU") {
			mode = GPU_NEW;
		} else
		if (option == "GPUHOSTMT") {
			mode = GPU_HOST_MT;
		} else
		if (option == "NEWGPUHOSTMT") {
			mode = GPU_NEW_HOST_MT;
		} else {
			mode = CPU;
		}
		cout << "ComputeMode = " << option << "(" << mode << ")" << endl;
		return mode;
	}
}

namespace mlvs_impute {

typedef GPU_MDI_worker_new_parallel GMDIWNP;
typedef CPU_MDI_worker_new_parallel CMDIWNP;

GMDIWNP::GPU_MDI_worker_new_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s, bool mtOnGpu_, uint Thread_id,
		uint Nthreads, uint device_) :
		CPU_MDI_worker_new_parallel(mod, s, Thread_id, Nthreads, device), mtOnGpu(mtOnGpu_), device(device_),
		gpuType(GPUType::BIG) {
//	std::cerr << "In GPU_MDI_worker_new_parallel ctor" << std::endl;

	dY = NULL;
	dX = NULL; dBeta = NULL;
  dRng = NULL; dEta = NULL; dLogZMin = NULL; dU = NULL;
  dWeight = NULL; dXtX = NULL; dXWU = NULL;

  dMu = NULL; dLogPriorWeight = NULL; dSigmaSqInv = NULL;
}

GMDIWNP::~GPU_MDI_worker_new_parallel() {
	// TODO Free all CUDA memory, should be released when context goes out of scope
}


int GMDIWNP::initializeData() {
	int error = CMDIWNP::initializeData(); if (error) return error;

	// Load data onto GPU
	dY = (uint*) util::allocateGPUMemory(sizeof(uint) * paddedDataChuckSize);
	if (!dY) return DeviceError::OUT_OF_MEMORY;

//	dX = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nChoices * paddedBetaSize);
//	if (!dX) return DeviceError::OUT_OF_MEMORY;

	dX = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nChoices * paddedBetaSize);
	if (!dX) return DeviceError::OUT_OF_MEMORY;

	cudaMemcpy(dY, &hY[0], sizeof(uint) * paddedDataChuckSize,
			cudaMemcpyHostToDevice);

	cudaMemcpy(dX, &hX[0], sizeof(Real) * paddedDataChuckSize * nChoices * paddedBetaSize,
			cudaMemcpyHostToDevice);

	std::cout << "Loaded data onto device!" << endl;

	// TODO Free hY and hX/hXt

	return DeviceError::NO_ERROR;
}

int GMDIWNP::initializeDevice() {
	cout << "Attempting to initialize GPU device(s)..." << endl;
	int totalNumDevices = util::getGPUDeviceCount();
	if (totalNumDevices == 0) {
		cerr << "No GPU devices found!" << endl;
		return DeviceError::NO_DEVICE;
	}

	if (totalNumDevices <= device) {
		cerr << "Fewer than " << (device + 1) << " devices found!" << endl;
		return DeviceError::NO_DEVICE;
	}
	util::printGPUInfo(device);
	cudaSetDevice(device);

	cublasStatus_t stat = cublasCreate(&handle);
	if	( stat !=	CUBLAS_STATUS_SUCCESS )	{
		cerr << "CUBLAS initialization failed" << endl;
		return	DeviceError::NO_DEVICE;
	}
	cout << "Device enabled!" << endl;
	return DeviceError::NO_ERROR;
}

int GMDIWNP::initializeInternalMemory(bool, bool) {
	int error = CMDIWNP::initializeInternalMemory(!mtOnGpu, false); if (error) return error;

	nRandomNumbers = paddedDataChuckSize * (2 * nChoices + 1);
	if (mtOnGpu) {
		uint remainder = nRandomNumbers % MT_RNG_COUNT;
		if (remainder != 0) {
			nRandomNumbers += MT_RNG_COUNT - remainder;
		}
	}

	dBeta = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedBetaSize);
	dEta = (Real*) util::allocateGPUMemory(sizeof(Real) * getEtaSize());
	dLogZMin = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize);
	dRng = (Real*) util::allocateGPUMemory(sizeof(Real) * nRandomNumbers);
	dU = (Real*) util::allocateGPUMemory(sizeof(Real) * getEtaSize());
	dWeight = (Real*) util::allocateGPUMemory(sizeof(Real) * getEtaSize());
	if (!dBeta || !dEta || !dLogZMin || !dRng || !dU || !dWeight) return DeviceError::OUT_OF_MEMORY;

	if (mtOnGpu) {
    loadMTGPU("MersenneTwister.dat"); // TODO Convert to mtrg in CUDA library
	}

  cout << "Completed internal allocation!" << endl;
	return DeviceError::NO_ERROR;
}

int GMDIWNP::initializeMixturePrior() {
	int error = CMDIWNP::initializeMixturePrior(); if (error) return error;
	gpuLoadConstantMemory(hMu, hSigmaSqInv, hLogPriorWeight, sizeof(Real) * priorMixtureSize);
	cout << "Completed constant memory load!" << endl;
	return DeviceError::NO_ERROR;
}


int GMDIWNP::initializeOutProducts() {
	const uint dim = paddedBetaSize;
//	const uint dim2 = dim * (betaSize + 1) / 2;
	const uint dim2 = dim * dim;

	dXtX = (Real*) util::allocateGPUMemory(sizeof(Real) * (dim2 + dim));
	dXWU = dXtX + dim2;

	hTmp = (Real*) calloc(sizeof(Real), (dim + dim * dim));
	cout << "Finished all initialization on GPU!" << endl;
	return DeviceError::NO_ERROR;
}

void GMDIWNP::operator()() {
	uploadBeta();
	generateRngNumbers();
	computeEta();
	reduceEta();
	sampleAllU();
	computeWeightedOuterProducts();
}

void GMDIWNP::uploadBeta() {
	CMDIWNP::uploadBeta();
	cudaMemcpy(dBeta, hBeta, sizeof(Real) * betaSize, cudaMemcpyHostToDevice);
}

void GMDIWNP::reduceEta() {
	gpuReduceEta_new(dLogZMin, dEta, dRng, paddedDataChuckSize, nChoices);
#ifdef DEBUG_PRINT
	cerr << endl;
	util::printfCudaVector(dLogZMin, dataChuckSize);
//	exit(-1);
#endif
}

void GMDIWNP::generateRngNumbers() {
	if (mtOnGpu) {
		uint seed = 0;
		while (seed <= 2) {
			double u = runif_mt(rng) * std::numeric_limits<int>::max();
			seed = lround(u);
		}
		seedMTGPU(seed); // cudaMemcpyHostToDevice inside
		gpuRandomMT(dRng, nRandomNumbers, gpuType = GPUType::BIG);
	} else {
		CMDIWNP::generateRngNumbers();
		cudaMemcpy(dRng, hRng, sizeof(Real) * nRandomNumbers, cudaMemcpyHostToDevice);
	}
}

void GMDIWNP::computeEta() {
//	gpuComputeEta_new(handle, dEta, dX, dBeta, paddedDataChuckSize, betaSize);
	Real alpha = 1;
	Real beta = 0;

	cublasSgemv(handle, CUBLAS_OP_N, paddedDataChuckSize * nChoices, paddedBetaSize,  &alpha, dX, paddedDataChuckSize * nChoices,
				dBeta, 1, &beta, dEta, 1);

#ifdef DEBUG_PRINT
	cerr << endl << "beta: " << betaSize << ": ";
	util::printfCudaVector(dBeta, betaSize);
	cerr << "eta: ";
	util::printfCudaVector(dEta, paddedDataChuckSize * nChoices);
//	exit(-1);
#endif
}

void GMDIWNP::sampleAllU() {
	assert(priorMixtureSize < 16);
	gpuSampleAllU_new(dU, dWeight, dY, dEta, dLogZMin, dRng,
			dMu, dSigmaSqInv, dLogPriorWeight,
			paddedDataChuckSize, dataChuckSize, nChoices, priorMixtureSize);
#ifdef DEBUG_PRINT
	cerr << endl << endl << "START";
	cerr << endl << "lzm:  ";
	util::printfCudaVector(dLogZMin, paddedDataChuckSize);
	cerr << endl << "eta:  ";
	util::printfCudaVector(dEta, paddedDataChuckSize * nChoices);
	cerr << endl << "u:    ";
	util::printfCudaVector(dU, paddedDataChuckSize * nChoices);
  cerr << endl << "wgts: ";
  util::printfCudaVector(dWeight, paddedDataChuckSize * nChoices);

//	exit(-1);
#endif
}

void GMDIWNP::computeWeightedOuterProducts() {

	Spd totalWeightedXXt(betaSize);
	std::vector<double> totalWeightedUtility(betaSize);

	bool smallBeta = false;

	if (smallBeta) {
		gpuReduceXtWX_new(dXtX,
				dX, dWeight, nChoices,
				paddedDataChuckSize, betaSize, paddedBetaSize);
	} else {

		int k = paddedDataChuckSize * nChoices;
		int n = paddedBetaSize;


#ifdef TIME_CUBLAS
		// Compare timing to CUBLAS SSYRK
		std::vector<Real> result(paddedBetaSize * paddedBetaSize);
		Real* dResult = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedBetaSize * paddedBetaSize);

		Real alpha = 1;
		Real beta = 0;

		cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k,
					&alpha, dX, paddedDataChuckSize * nChoices,
					&beta, dResult,  paddedBetaSize);
		cudaThreadSynchronize();

		cudaFree(dResult);
#endif

		ripSsyrk (n, k, dX, paddedDataChuckSize * nChoices, dWeight, dXtX, paddedBetaSize);
	}

	gpuReduceXtWU_new(dXWU,
			dX, dU, dWeight, nChoices,
			paddedDataChuckSize, betaSize);

	cudaMemcpy(hTmp, dXtX, sizeof(Real) * (paddedBetaSize + 1) * paddedBetaSize,
				cudaMemcpyDeviceToHost);

#if 1
	for (uint i = 0; i < betaSize; ++i) {
		for (uint j = 0; j < betaSize; ++j) {
			totalWeightedXXt(i,j) = hTmp[i * paddedBetaSize + j];
		}
	}
#else
	uint index = 0;
	for (uint i = 0; i < betaSize; ++i) {
		for (uint j = i; j < betaSize; ++j) {
			totalWeightedXXt(i,j) = hTmp[index];
			index++;
		}
	}
#endif



	const Real* tmp = hTmp  + paddedBetaSize * paddedBetaSize;
	for (uint i = 0; i < betaSize; ++i) {
		totalWeightedUtility[i] = tmp[i];
	}

	Ptr<MlvsCdSuf_ml> try2 = new MlvsCdSuf_ml(totalWeightedXXt, totalWeightedUtility);
	suf_->clear();
	suf_->add(try2);

#ifdef DEBUG_PRINT
	  cerr << endl << "dxtx: ";
	  util::printfCudaVector(dXtX, paddedBetaSize * (paddedBetaSize + 1));
	  cerr << "XtWU: ";
//	  util::printfCudaVector(dXWU, betaSize);
//	  cerr << endl;
		util::printfVector(&totalWeightedUtility[0], betaSize);
		cerr << endl;
//		util::printfCudaVector(dXtX, dim2);
		cerr << totalWeightedXXt;
//		cerr << "SUF = " << endl << suf_->xtwx() << endl;
//		cerr << "SUF = " << suf_->xtwu() << endl;
//		cerr << "New GPU" << endl;
		cerr << endl;
//		util::printfCudaVector(dWeight, paddedDataChuckSize * nChoices + 2);
//		exit(0);
#endif
}

/*****************/

typedef GPU_MDI_worker GMDIW;

GMDIW::GPU_MDI_worker(MLogitBase *mod, Ptr<MlvsCdSuf> s,
		bool mtOnGpu_,
		uint Thread_id, uint Nthreads, uint device) :
	MDI_worker(mod, s, Thread_id, Nthreads), mtOnGpu(mtOnGpu_),
	gpuType(GPUType::BIG) {

	std::cerr << "In GPU constructor" << std::endl;

	initializeGPU(device);

	initializeData();

	initializeInternalMemory();

	initializeMixturePrior();

	initializeOutProducts();

	if (mtOnGpu) {
    loadMTGPU("MersenneTwister.dat"); // TODO Convert to mtrg in CUDA library
	}
}

void GMDIW::initializeData() {

	// Initialize data
	nSubjectVars = mlm->subject_nvars();
	nChoiceVars = mlm->choice_nvars();
	nChoices = mlm->Nchoices();

	nNonZeroChoices = nChoices - 1; // For eta, U, etc., only need eta, u, etc., for y > 0

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
	betaSize = nSubjectVars * nNonZeroChoices + nChoiceVars;
	paddedBetaSize = betaSize;  // No need to pad in this dimension of X
	priorMixtureSize = post_prob_.size();

	hX = (Real*) calloc(sizeof(Real),
			paddedDataChuckSize	* nSubjectVars); // temporary contiguous host memory to hold design matrix

	for (uint index = 0; index < dataChuckSize; index++) {
		uint datumIndex = thread_id + index * nthreads;
		Ptr<ChoiceData> dp(dat[datumIndex]);
		const Mat &datumX(dp->X());

		const double* oldDatum = datumX.data();
		for (uint k = 0; k < nSubjectVars; k++) {
			hX[index + paddedDataChuckSize * k] = (Real) oldDatum[nChoices * k];
		}
	}
	for (uint index = dataChuckSize; index < paddedBetaSize; index++) {
		for (uint k = 0; k < nSubjectVars; k++) {
			hX[index + paddedDataChuckSize * k] = 0;
		}
	}

	// Load data onto GPU
	dY = (uint*) util::allocateGPUMemory(sizeof(uint) * paddedDataChuckSize);
	dX = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nSubjectVars);

	cudaMemcpy(dY, &hY[0], sizeof(uint) * paddedDataChuckSize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(dX, hX, sizeof(Real) * paddedDataChuckSize * nSubjectVars,
			cudaMemcpyHostToDevice);
}

void GMDIW::initializeInternalMemory() {

	nRandomNumbers = paddedDataChuckSize * (2 * nChoices + 1);
	if (mtOnGpu) {
		uint remainder = nRandomNumbers % MT_RNG_COUNT;
		if (remainder != 0) {
			nRandomNumbers += MT_RNG_COUNT - remainder;
		}
	}

	hBeta = (Real*) calloc(sizeof(Real), paddedBetaSize);
	hEta = (Real*) calloc(sizeof(Real), paddedDataChuckSize * nNonZeroChoices);
	hLogZMin = (Real*) malloc(sizeof(Real) * paddedDataChuckSize);
	hRng = (Real*) malloc(sizeof(Real) * nRandomNumbers);
//			paddedDataChuckSize * (2 * nChoices + 1) ); // Two draws per datum
	hU = (Real*) malloc(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
	hK = (uint*) malloc(sizeof(uint) * paddedDataChuckSize * nNonZeroChoices);
	hWeight = (Real*) malloc(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);


	dBeta = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedBetaSize);
	dEta = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
	dLogZMin = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize);
	dRng = (Real*) util::allocateGPUMemory(sizeof(Real) * nRandomNumbers);
	dU = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
	dWeight = (Real*) util::allocateGPUMemory(sizeof(Real) * paddedDataChuckSize * nNonZeroChoices);
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

#ifdef USE_CONSTANT

	gpuLoadConstantMemory(hMu, hSigmaSqInv, hLogPriorWeight, priorMemorySize);
//	cudaMemcpyToSymbol(cMu, hMu, priorMemorySize, 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(cLogPrior, hLogPriorWeight, priorMemorySize, 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(cPrec, hSigmaSqInv, priorMemorySize, 0, cudaMemcpyHostToDevice);

#else
	dMu = (Real*) util::allocateGPUMemory(priorMemorySize);
	dLogPriorWeight = (Real*) util::allocateGPUMemory(priorMemorySize);
	dSigmaSqInv = (Real*) util::allocateGPUMemory(priorMemorySize);

	cudaMemcpy(dMu, hMu, priorMemorySize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(dLogPriorWeight, hLogPriorWeight, priorMemorySize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(dSigmaSqInv, hSigmaSqInv, priorMemorySize,
			cudaMemcpyHostToDevice);
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
	const uint dim = nNonZeroChoices * nSubjectVars;
	const uint dim2 = dim * (nSubjectVars + 1) / 2;

	dXtX = (Real*) util::allocateGPUMemory(sizeof(Real) * (dim2 + dim)); // * nXtXReducedRows);
	dXWU = dXtX + dim2;

	hTmp = (Real*) malloc(sizeof(Real) * (dim + dim * dim));
}

void GMDIW::computeWeightedOuterProducts() {

	const uint dim = nSubjectVars * nNonZeroChoices;
	const uint dim2 = dim * (nSubjectVars + 1) / 2;

	Spd totalWeightedXXt(dim);
	std::vector<double> totalWeightedUtility(dim);

	gpuReduceXtWX(dXtX,
			dX,dWeight, 0, nXtXReducedRows, nNonZeroChoices,
			paddedDataChuckSize, nSubjectVars);

	gpuReduceXWU(dXWU,
			dX, dU, dWeight, nXWUReducedRows, nNonZeroChoices,
			paddedDataChuckSize, nSubjectVars);

	cudaMemcpy(hTmp, dXtX, sizeof(Real) * (dim2 + dim),
			cudaMemcpyDeviceToHost);

	uint index = 0;
	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		uint offset = choice * nSubjectVars;
		for (uint i = 0; i < nSubjectVars; i++) {
			for (uint j = i; j < nSubjectVars; j++) {
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

	Ptr<MlvsCdSuf_ml> try2 = new MlvsCdSuf_ml(totalWeightedXXt, totalWeightedUtility);
	suf_->clear();
	suf_->add(try2);
}

void GMDIW::uploadBeta() {
	MultinomialLogitModel *ptrMLM = dynamic_cast<MultinomialLogitModel*>(mlm);
	const double* modelBeta = ptrMLM->beta().data();
	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		for (uint k = 0; k < nSubjectVars; k++) {
			hBeta[choice * nSubjectVars + k] = modelBeta[choice * nSubjectVars + k];
		}
	}

	cudaMemcpy(dBeta, hBeta, sizeof(Real) * nSubjectVars * nNonZeroChoices,
			cudaMemcpyHostToDevice);
}

void GMDIW::generateRngNumbers() {

	if (mtOnGpu) {
		uint seed = 0;
		while (seed <= 2) {
			double u = runif_mt(rng) * std::numeric_limits<int>::max();
			seed = lround(u);
		}
		seedMTGPU(seed);
		gpuRandomMT(dRng, nRandomNumbers, gpuType == GPUType::BIG);
	} else {

		for (uint index = 0; index < dataChuckSize; index++) {
			for (uint k = 0; k < (2 * nChoices + 1); k++) {
				hRng[k * paddedDataChuckSize + index] = runif_mt(rng);
			}
		}
		for (uint index = dataChuckSize; index < paddedDataChuckSize; index++) {
			for (uint k = 0; k < (2 * nChoices + 1); k++) {
				hRng[k * paddedDataChuckSize + index] = 0.5; // Just avoid denormalized numbers
			}
		}
		cudaMemcpy(dRng, hRng, sizeof(Real) * nRandomNumbers,
				cudaMemcpyHostToDevice);
	}
}

void GMDIW::computeEta() {
	assert(nNonZeroChoices * nSubjectVars <= COMPUTE_ETA_DATA_BLOCK_SIZE);
	gpuComputeEta(dEta, dX, dBeta, paddedDataChuckSize, nNonZeroChoices, nSubjectVars);
}

void GMDIW::reduceEta() {
	gpuReduceEta(dLogZMin, dEta, dRng, paddedDataChuckSize, nNonZeroChoices);
}

void GMDIW::sampleAllU() {
	assert(priorMixtureSize < 16);
	gpuSampleAllU(dU, dWeight, dY, dEta, dLogZMin, dRng,
			dMu, dSigmaSqInv, dLogPriorWeight,
			paddedDataChuckSize, nNonZeroChoices, priorMixtureSize);

#ifdef DEBUG_PRINT
	util::printfCudaVector(dWeight, paddedDataChuckSize * nNonZeroChoices);
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

void GMDIW::operator()() {
	uploadBeta();
	generateRngNumbers();
	computeEta();
	reduceEta();
	sampleAllU();
	computeWeightedOuterProducts();
}

int util::getGPUDeviceCount() {
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

void util::printGPUInfo(int iDevice) {

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

void util::getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed) {
	cudaDeviceProp deviceProp;
	memset(&deviceProp, 0, sizeof(deviceProp));
	cudaGetDeviceProperties(&deviceProp, iDevice);
	*oMemory = deviceProp.totalGlobalMem;
	*oSpeed = deviceProp.clockRate;
	strcpy(oName, deviceProp.name);
}

void GMDIW::initializeGPU(int device) {
	cout << "Attempting to initialize GPU device(s)..." << endl;
	int totalNumDevices = util::getGPUDeviceCount();
	if (totalNumDevices == 0) {
		cerr << "No GPU devices found!" << endl;
		exit(-1);
	}

	if (totalNumDevices <= device) {
		cerr << "Fewer than " << (device + 1) << " devices found!" << endl;
		exit(-1);
	}
	util::printGPUInfo(device);
	cudaSetDevice(device);
	cout << "Device enabled!" << endl;
}

void* util::allocateGPUMemory(size_t size) {
	void* ptr;
	SAFE_CUDA(cudaMalloc((void**) &ptr, size), ptr);
	if (ptr == NULL) {
		cerr << "Failed to allocate " << size << " bytes of memory on device!" << endl;
	} else {
		SAFE_CUDA(cudaMemset(ptr, 0, size), ptr);
	}
	return ptr;
}

template <class RealType>
void printOne(RealType x) {
	printf("% 3.2e ", x);
}

template <class RealType>
void util::printfVector(RealType *hPtr, uint length) {
//#define ALIGNED
#ifndef ALIGNED
	if (length > 0) {
		cout << hPtr[0];
	}
	for (uint i = 1; i < length; i++) {
		cout << " " << hPtr[i];
		if (hPtr[i] != hPtr[i]) {
			cerr << endl << "Nan!" << endl;
			exit(-1);
		}
	}
	cout << endl;
#else
	for (uint i = 0; i < length; ++i) {
		printOne(hPtr[i]);
		if (hPtr[i] != hPtr[i]) {
			cerr << endl << "Nan!" << endl;
			exit(-1);
		}
	}
	printf("\n");
#endif
}

template <class RealType>
void util::printfCudaVector(RealType* dPtr, uint length) {

	RealType* hPtr = (RealType *) malloc(sizeof(RealType) * length);
	SAFE_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(RealType)*length, cudaMemcpyDeviceToHost),dPtr);
	util::printfVector(hPtr, length);
	free(hPtr);
}

//----------------------------------------------------------------------

} // mlvs_impute

} // BOOM
