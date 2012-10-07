/*
 * GPU_MDI_worker.h
 *
 *  Created on: Apr 10, 2010
 *      Author: Marc A. Suchard
 */

#ifndef GPU_MDI_WORKER_H_
#define GPU_MDI_WORKER_H_

#include <boost/algorithm/string.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <cublas_v2.h>
#include "CPU_MDI_worker_MS.hpp"

typedef float Real;

void ripSsyrk (int n, int k,
                                       const float *A, int lda, const float *W,
                                       float *C, int ldc);

namespace BOOM {

/**
 * CPU/GPU_NEW are the main compute modes.
 *
 * Several of these compute modes exist to debug code.
 * The following pairs should return approximately the same bit stream
 * (only differences due to precision and non-transitivity of floating point):
 * 		CPU and CPU_ORIGINAL
 *		CPU_NEW_FLOW and CPU_NEW_PARALLEL
 * 		CPU_NEW_PARALLEL and GPU_NEW_HOST_MT
 *
 * The following pairs should return different bit streams (assumed not to result from bugs):
 *		CPU_ORIGINAL and CPU_NEW_FLOW -- differ in their RNG access only
 *		GPU_NEW_HOST_MT and GPU_NEW -- RNG moved to GPU
 *
 * Specialized modes for data sets with only subject-specific predictors:
 * 		CPU_PARALLEL
 * 		GPU_HOST_MT
 * 		GPU
 */

namespace ComputeMode {
	enum {
		CPU=0, CPU_ORIGINAL, CPU_NEW_FLOW, CPU_PARALLEL, CPU_NEW_PARALLEL,
		GPU, GPU_NEW, GPU_HOST_MT, GPU_NEW_HOST_MT
	};

	int parseComputeModel(std::string option);
}

namespace DeviceError {
	enum { NO_ERROR=0, NO_DEVICE, OUT_OF_MEMORY };
}

namespace GPUType {
	enum { SMALL, BIG };
}

namespace mlvs_impute {

class GPU_MDI_worker_new_parallel : public CPU_MDI_worker_new_parallel {
public:
	GPU_MDI_worker_new_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s, bool mtOnGPU = true, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);

	virtual ~GPU_MDI_worker_new_parallel();

	virtual void operator()();

protected:
	virtual int initializeData(void);
  virtual int initializeDevice(void);
  virtual int initializeInternalMemory(bool, bool);
  virtual int initializeMixturePrior(void);
  virtual int initializeOutProducts(void);

private:
  void uploadBeta(void);
  void generateRngNumbers(void);
  void computeEta(void);
  void reduceEta(void);
  void sampleAllU(void);
  void computeWeightedOuterProducts(void); // TODO Change to *Impl
//  virtual void reduceEta(void);
//  virtual void sampleAllU(void);
//  virtual void computeWeightedOuterProducts(void);

private:
  uint device;
  cublasHandle_t handle;
  bool mtOnGpu;
  uint gpuType;

  // permanent device memory
  uint *dY;
  Real *dX;
  Real *dBeta;
  Real *dRng;
  Real *dEta;
  Real *dLogZMin; // TODO Rename
  Real *dU;
  Real *dWeight;
  Real *dXtX;
  Real *dXWU;

  Real *dMu;
  Real *dLogPriorWeight;
  Real *dSigmaSqInv;
};



class GPU_MDI_worker: public BOOM::mlvs_impute::MDI_worker {
public:
	GPU_MDI_worker(MLogitBase *mod, Ptr<MlvsCdSuf> s, bool mtOnGpu = true, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);
	virtual ~GPU_MDI_worker();

  void impute_u(Ptr<ChoiceData> dp, uint index);
  virtual void operator()();

private:
    bool mtOnGpu;
    uint gpuType;

    // GPU functions
    void uploadPrior(void);
    void uploadBeta(void);
    void computeEta(void);
    void reduceEta(void);
    void generateRngNumbers(void);
    void sampleAllU(void);
    uint sampleOneU(Real x, Real unif); // TODO Move to just GPU
    void reduceU(void);
    void computeCdSf(void);
    void initializeOutProducts(void);
    void computeWeightedOuterProducts(void);
    void initializeMixturePrior(void);
    void initializeData(void);
    void initializeInternalMemory(void);

    // temporary host memory
    std::vector<uint> hY;

    Real* hX;
    Real* hXt;
    Real* hBeta;
    Real* hRng;
    Real* hEta; // TODO To be removed
    Real* hLogZMin; // TODO To be removed
    Real* hU; // TODO To be removed
    uint* hK; // TODO To be removed
    Real* hWeight; // TODO To be removed
    Real* hXtX; // TODO To be removed
    Real* hTmp;

    Real* hMu; // TODO To be removed
    Real* hStd; // TODO To be removed
    Real* hLogPriorWeight; // TODO To be removed
    Real* hPostWeight; // TODO To be removed
    Real* hSigmaSqInv; // TODO To be removed

    // permanent device memory
    uint *dY;
    Real *dX;
    Real *dXt;
    Real *dBeta;
    Real *dRng;
    Real *dEta;
    Real *dLogZMin; // TODO Rename
    Real *dU;
    Real *dWeight;
    Real *dXtX;
    Real *dXWU;

    Real *dMu;
    Real *dLogPriorWeight;
    Real *dSigmaSqInv;

    uint dataChuckSize;
    uint paddedDataChuckSize;
    uint betaSize;
    uint paddedBetaSize;
    uint nChoices;
    uint nChoiceVars;
//    uint nSubjectVars;
    uint nNonZeroChoices;
    uint nSubjectVars;
    uint strideX;
    uint priorMixtureSize;

    uint nRandomNumbers;

    uint nXtXReducedRows;
    uint nXWUReducedRows;

    // Basic GPU functions
    void initializeGPU(int device);
//    int getGPUDeviceCount();
//    void printGPUInfo(int iDevice);
//    void getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed);
//    void* allocateGPUMemory(size_t size);



};

namespace util {
	int getGPUDeviceCount();
	void printGPUInfo(int iDevice);
	void getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed);
	void* allocateGPUMemory(size_t size);

  template <class RealType>
  void printfCudaVector(RealType* dPtr, uint length);

  template <class RealType>
  void printfVector(RealType* hPtr, uint length);
}

}

}

#endif /* GPU_MDI_WORKER_H_ */
