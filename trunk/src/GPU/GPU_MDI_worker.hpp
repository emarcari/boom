/*
 * GPU_MDI_worker.h
 *
 *  Created on: Apr 10, 2010
 *      Author: Marc A. Suchard
 */

#ifndef GPU_MDI_WORKER_H_
#define GPU_MDI_WORKER_H_

#include <Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>

typedef float Real;

namespace BOOM {

namespace ComputeMode {
	enum { CPU=0, GPU, CPU_ORIGINAL, CPU_NEW_FLOW, CPU_PARALLEL, CPU_NEW_PARALLEL };
}

namespace mlvs_impute {

class CPU_MDI_worker : public BOOM::mlvs_impute::MDI_worker {
public:
	CPU_MDI_worker(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);
	virtual ~CPU_MDI_worker();
  void impute_u(Ptr<ChoiceData> dp, uint index);
  virtual void operator()();
};

class GPU_MDI_worker: public BOOM::mlvs_impute::MDI_worker {
public:
	GPU_MDI_worker(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);
	virtual ~GPU_MDI_worker();

  void impute_u(Ptr<ChoiceData> dp, uint index);
  virtual void operator()();

private:

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
    int getGPUDeviceCount();
    void printGPUInfo(int iDevice);
    void getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed);
    void* allocateGPUMemory(size_t size);

    template <class RealType>
    void printfCudaVector(RealType* dPtr, uint length);

    template <class RealType>
    void printfVector(RealType* hPtr, uint length);


};

}

}

#endif /* GPU_MDI_WORKER_H_ */
