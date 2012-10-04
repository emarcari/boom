/*
 * GPU_MDI_worker.h
 *
 *  Created on: Apr 10, 2010
 *      Author: Marc A. Suchard
 */

#ifndef CPU_MDI_WORKER_MS_H_
#define CPU_MDI_WORKER_MS_H_

#include <Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>

typedef float Real;

namespace BOOM {

namespace mlvs_impute {

class CPU_MDI_worker_original: public BOOM::mlvs_impute::MDI_worker {
public:
	CPU_MDI_worker_original(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);
	virtual ~CPU_MDI_worker_original();

  virtual void impute_u(Ptr<ChoiceData> dp, uint index);
  virtual void operator()();
};

class CPU_MDI_worker_new_flow : public CPU_MDI_worker_original {
public:
	CPU_MDI_worker_new_flow(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0); // : CPU_MDI_worker_original(mod, s, Thread_id, Nthreads, device) {}
	virtual ~CPU_MDI_worker_new_flow();
	virtual void impute_u(Ptr<ChoiceData> dp, uint index);
};

class CPU_MDI_worker_parallel : public BOOM::mlvs_impute::MDI_worker {
public:
	CPU_MDI_worker_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);

	virtual ~CPU_MDI_worker_parallel();

	void initialize(void);

  virtual void operator()();

protected: // TODO Encapsulate

    // Parallel structure functions
    void uploadPrior(void);

    void generateRngNumbers(void);
    uint sampleOneU(Real x, Real unif); // TODO Move to just GPU
    void reduceU(void);
    void computeCdSf(void);
    void initializeOutProducts(void);
    void initializeMixturePrior(void);

    void initializeInternalMemory(void);

    // Virtualized for testing
    virtual uint getEtaSize(void);
    virtual void initializeData(void);
    virtual void computeEta(void);
    virtual void uploadBeta(void);
    virtual void reduceEta(void);
    virtual void sampleAllU(void);
    virtual void computeWeightedOuterProducts(void);

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

//     // permanent device memory
//     uint *dY;
//     Real *dX;
//     Real *dXt;
//     Real *dBeta;
//     Real *dRng;
//     Real *dEta;
//     Real *dLogZMin; // TODO Rename
//     Real *dU;
//     Real *dWeight;
//     Real *dXtX;
//     Real *dXWU;
// 
//     Real *dMu;
//     Real *dLogPriorWeight;
//     Real *dSigmaSqInv;

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

//     // Basic GPU functions
//     void initializeGPU(int device);
//     int getGPUDeviceCount();
//     void printGPUInfo(int iDevice);
//     void getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed);
//     void* allocateGPUMemory(size_t size);
// 
//     template <class RealType>
//     void printfCudaVector(RealType* dPtr, uint length);

    template <class RealType>
    void printfVector(RealType *hPtr, uint length) {
    	if (length > 0) {
    		cout << hPtr[0];
    	}
    	for (uint i = 1; i < length; i++) {
    		cout << " " << hPtr[i];
    	}
    	cout << endl;
    }

};

class CPU_MDI_worker_new_parallel : public CPU_MDI_worker_parallel {
public:
	CPU_MDI_worker_new_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);

	virtual ~CPU_MDI_worker_new_parallel();

protected:
  inline uint getIndex(uint row, uint k);
  inline uint getRow(uint i, uint j);

  virtual uint getEtaSize(void);
	virtual void initializeData(void);
	virtual void computeEta(void);
	virtual void uploadBeta(void);
	virtual void reduceEta(void);
	virtual void sampleAllU(void);
	virtual void computeWeightedOuterProducts(void);
};

}

}

#endif /* CPU_MDI_WORKER_MS_H_ */
