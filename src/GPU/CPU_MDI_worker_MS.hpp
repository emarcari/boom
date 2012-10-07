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

	int initialize(void);

  virtual void operator()();

protected:

    // Parallel structure functions
    void uploadPrior(void);

    void generateRngNumbers(void);
    uint sampleOneU(Real x, Real unif);

//    void reduceU(void);
//    void computeCdSf(void);

    // Virtual for testing
//    virtual uint getEtaSize(void);
//    virtual void computeEta(void);
//    virtual void uploadBeta(void);
//    virtual void reduceEta(void);
//    virtual void sampleAllU(void);
//    virtual void computeWeightedOuterProducts(void);

    // Initialization is mostly likely always virtual
    virtual int initializeDevice(void);
    virtual int initializeData(void);
    virtual int initializeOutProducts(void);
    virtual int initializeMixturePrior(void);
    virtual int initializeInternalMemory(bool rng, bool intermediates);

    std::vector<uint> hY;
    std::vector<Real> hX;

    // TODO Convert all to std::vector<Real>
    Real* hBeta;
    Real* hRng;
    Real* hEta; // TODO To be removed
    Real* hLogZMin; // TODO To be removed
    Real* hU; // TODO To be removed

    Real* hWeight; // TODO To be removed
    Real* hXtX; // TODO To be removed
    Real* hTmp;

    Real* hMu; // TODO To be removed
    Real* hStd; // TODO To be removed
    Real* hLogPriorWeight; // TODO To be removed
    Real* hPostWeight; // TODO To be removed
    Real* hSigmaSqInv; // TODO To be removed

    uint dataChuckSize;
    uint paddedDataChuckSize;
    uint betaSize;
    uint paddedBetaSize;
    uint nChoices;
    uint nChoiceVars;
    uint nNonZeroChoices;
    uint nSubjectVars;
    uint strideX;
    uint priorMixtureSize;

    uint nRandomNumbers;
    uint nXtXReducedRows;
    uint nXWUReducedRows;

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

private:
    virtual uint getEtaSize(void);
    void computeEta(void);
    void uploadBeta(void);
    void reduceEta(void);
    void sampleAllU(void);
    void computeWeightedOuterProducts(void);

};

class CPU_MDI_worker_new_parallel : public CPU_MDI_worker_parallel {
public:
	CPU_MDI_worker_new_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s, uint Thread_id = 0,
			uint Nthreads = 1, uint device = 0);

	virtual ~CPU_MDI_worker_new_parallel();

  virtual void operator()();

protected:
  // Accessor functions
  inline uint getRow(uint i, uint j);

	virtual int initializeData(void);
	virtual uint getEtaSize(void);

	void computeEta(void);
	void uploadBeta(void);
  void reduceEta(void);
	void sampleAllU(void);
	void computeWeightedOuterProducts(void);
};

}

}

#endif /* CPU_MDI_WORKER_MS_H_ */
