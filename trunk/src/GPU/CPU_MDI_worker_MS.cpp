/*
 * GPU_MDI_worker.cpp
 *
 *  Created on: Apr 10, 2010
 *      Author: msuchard
 */

#define ROW_MAJOR

//#define FIX_BETA // TODO Remove

//#define DEBUG_PRINT

#include "CPU_MDI_worker_MS.hpp"
#include "Models/Glm/PosteriorSamplers/MLVS.hpp"

#include <boost/ref.hpp>
#include <boost/cast.hpp>

#include <cpputil/math_utils.hpp>
#include <cpputil/lse.hpp>
#include <stats/logit.hpp>
#include <distributions.hpp>

#include "GPU_MDI_worker.hpp"

using namespace std;

namespace BOOM {

namespace mlvs_impute {

typedef CPU_MDI_worker_original CMDIWO;

CMDIWO::CPU_MDI_worker_original(MLogitBase *mod, Ptr<MlvsCdSuf> s,
		uint Thread_id, uint Nthreads, uint device) :
	MDI_worker(mod, s, Thread_id, Nthreads) {
	std::cerr << "In CPU_MDI_worker_original cstor" << std::endl;
}

CMDIWO::~CPU_MDI_worker_original() {
}

void CMDIWO::impute_u(Ptr<ChoiceData> dp, uint index) {
	mlm->fill_eta(*dp, eta); // eta+= downsampling_logprob
#ifdef DEBUG_PRINT
  const Mat &X(dp->X(false));
  std::cerr << X << std::endl;

	MultinomialLogitModel *ptrMLM = dynamic_cast<MultinomialLogitModel*> (mlm);
	std::cerr << ptrMLM->beta_with_zeros() << std::endl;
//	exit(-1);
#endif
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
		}
		uint k = unmix(u[m] - eta[m]);
		u[m] -= mu_[k];
		wgts[m] = sigsq_inv_[k];
	}
}

void CMDIWO::operator()() {
	const std::vector<Ptr<ChoiceData> > & dat(mlm->dat());
	suf_->clear();
	uint n = dat.size();
	uint i = thread_id;
	uint index = 0;
	while (i < n) {
		Ptr<ChoiceData> dp(dat[i]);
		impute_u(dp, index);
		suf_->update(dp, wgts, u);
		i += nthreads;
		index++;
#ifdef DEBUG_PRINT
		cerr << "wgts: " << wgts << endl;
//		exit(0);
#endif
	}
}

typedef CPU_MDI_worker_new_flow CMDIWNF;

CMDIWNF::CPU_MDI_worker_new_flow(MLogitBase *mod, Ptr<MlvsCdSuf> s,
		uint Thread_id, uint Nthreads, uint device) :
	CPU_MDI_worker_original(mod, s, Thread_id, Nthreads, device) {
	std::cerr << "In CPU_MDI_worker_new_flow cstor" << std::endl;
}

CMDIWNF::~CPU_MDI_worker_new_flow() {
}

void CMDIWNF::impute_u(Ptr<ChoiceData> dp, uint index) {
	mlm->fill_eta(*dp, eta); // eta+= downsampling_logprob
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

typedef CPU_MDI_worker_new_parallel CMDIWNP;

CMDIWNP::CPU_MDI_worker_new_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s,
		uint Thread_id, uint Nthreads, uint device)
		: CPU_MDI_worker_parallel(mod, s, Thread_id, Nthreads, device)
		  {
//	std::cerr << "In CPU_MDI_worker_new_parallel ctor" << std::endl;
}

CMDIWNP::~CPU_MDI_worker_new_parallel() { }

inline uint CMDIWNP::getRow(uint i, uint j) {
	return i + j * paddedDataChuckSize;
}

//inline uint CMDIWNP::getIndex(uint row, uint k) {
//
//	// Let x_{ij} be the row vector of attributes for subject i and choice j, then
//	// X = ( x_{11}, \ldots, x_{N1}, x_{12}, \ldots, x_{N2}, \ldots, x_{NC} )^t
//#ifdef ROW_MAJOR
//	// Use: row-major storage (X\beta), stride = paddedBetaSize
//	return row * paddedBetaSize + k;
//#else
//	// Use: column-major storage, stride = paddedDataChuckSize * nChoices
//	uint stride = paddedDataChuckSize * nChoices;
//	return row + stride * k;
//#endif
//}

void CMDIWNP::computeEta() {
//	for (uint index = 0; index < dataChuckSize; index++) {
//		for (uint choice = 0; choice < nChoices; choice++) {
//			Real sum = 0;
//			uint rowOffset = getRow(index, choice);
//
//#ifdef ROW_MAJOR
//			Real* x = &hX[0] + getIndex(rowOffset, 0);
//			for (uint k = 0; k < betaSize; k++) {
//				sum += x[k] * hBeta[k]; // Better caching
//			}
//#else
//			for (uint k = 0; k < betaSize; k++) {
//				sum += hX[getIndex(rowOffset, k)] * hBeta[k];
//			}
//#endif
//
//			// TODO Adjust for down sampling
//			hEta[rowOffset] = sum;
//		}
//	}
	uint length = paddedDataChuckSize * nChoices;
	for (uint i = 0; i < length; ++i) {
		Real sum = 0;
		for (uint j = 0; j < betaSize; ++j) {
			sum += hX[i + length * j] * hBeta[j];
		}
		hEta[i] = sum;
	}

#ifdef DEBUG_PRINT
	std::cerr << "beta: " << betaSize << ": ";
	for (uint i = 0; i < betaSize; ++i) {
		std::cerr << hBeta[i] << " ";
	}
	std::cerr << std::endl;
	std::cerr << "eta: ";
	for (uint i = 0; i < getEtaSize(); ++i) {
		std::cerr << hEta[i] << " ";
	}
	std::cerr << std::endl << std::endl;
//	exit(0);
#endif
}

int CMDIWNP::initializeData() {

	// Initialize data
	nSubjectVars = mlm->subject_nvars();
	nChoiceVars = mlm->choice_nvars();
	nChoices = mlm->Nchoices();
	nNonZeroChoices = nChoices - 1; // For eta, U, etc., only need eta, u, etc., for y > 0

	int padDim = 32; // Should be multiple of 16 and equal to TILE_DIM for weighted syrk

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
	uint remainder = paddedDataChuckSize % padDim;
	if (remainder != 0) {
		paddedDataChuckSize += padDim - remainder;
		for (uint i = dataChuckSize; i < paddedDataChuckSize; i++) {
			hY.push_back(0);
		}
	}
	betaSize = nSubjectVars * nNonZeroChoices + nChoiceVars;
	paddedBetaSize = betaSize;

	bool padBeta = true; // Should depend on betaSize;

	if (padBeta) {
		remainder = paddedBetaSize % padDim;
		if (remainder != 0) {
			paddedBetaSize += padDim - remainder;
		}
	}
	priorMixtureSize = post_prob_.size();

	hX.resize(paddedDataChuckSize * nChoices * paddedBetaSize); // lda: paddedDataChuckSize * nChoices

	for (uint index = 0; index < dataChuckSize; index++) {
		uint datumIndex = thread_id + index * nthreads;
		Ptr<ChoiceData> dp(dat[datumIndex]);
		const Mat& datumNoIntercept(dp->X(false));
		const double* oldDatum = datumNoIntercept.data(); // Stored in column-major, stride = nChoices

		for (uint j = 0; j < nChoices; ++j) {
			uint rowOffset = getRow(index, j);
			for (uint k = 0; k < betaSize; k++) {
				hX[rowOffset + paddedDataChuckSize * nChoices * k] = (Real) oldDatum[j + nChoices * k];
			}
		}
	}
	return DeviceError::NO_ERROR;
}


typedef CPU_MDI_worker_parallel CMDIWP;

CMDIWP::CPU_MDI_worker_parallel(MLogitBase *mod, Ptr<MlvsCdSuf> s,
		uint Thread_id, uint Nthreads, uint device) :
	MDI_worker(mod, s, Thread_id, Nthreads)
,
  hBeta(NULL),
  hRng(NULL),
  hEta(NULL),
  hLogZMin(NULL),
  hU(NULL),
  hWeight(NULL),
  hXtX(NULL),
  hTmp(NULL)
{
//	std::cerr << "In CPU_MDI_worker_parallel cstor" << std::endl;
}

int CMDIWP::initialize() {
	int error = initializeDevice();
//	cerr << "device: " << error << endl;
	if (error) return error;
	error = initializeData();
//	cerr << "data: " << error << endl;
	if (error) return error;
	error = initializeInternalMemory(true, true);
//	cerr << "mem: " << error << endl;
	if (error) return error;
	error = initializeMixturePrior();
//	cerr << "prior: " << error << endl;
	if (error) return error;
	error = initializeOutProducts();
//	cerr << "products: " << error << endl;
	if (error) return error;
	return error;
}

int CMDIWP::initializeDevice() {
	return DeviceError::NO_ERROR;
}

int CMDIWP::initializeData() {

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
	paddedBetaSize = betaSize; // No padding in this dimension for X
	priorMixtureSize = post_prob_.size();
//
//	cerr << "dataChuckSize = " << dataChuckSize << endl;
//	cerr << "paddedDataChuckSize = " << paddedDataChuckSize << endl;
//	cerr << "nChoices = " << nChoices << endl;
//	cerr << "nSubjectVars = " << nSubjectVars << endl;
//	cerr << "nChoiceVars = " << nChoiceVars << endl;
//	cerr << "betaSize = " << betaSize << endl;
//	cerr << "paddedBetaSize = " << paddedBetaSize << endl;
//	cerr << "mixture size = " << priorMixtureSize << endl;

		hX.resize(paddedDataChuckSize * nSubjectVars); // temporary contiguous host memory to hold design matrix

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
	return DeviceError::NO_ERROR;
}

uint CMDIWNP::getEtaSize() {
	return paddedDataChuckSize * nChoices;
}

uint CMDIWP::getEtaSize() {
	return paddedDataChuckSize * nNonZeroChoices;
}

int CMDIWP::initializeInternalMemory(bool rng, bool intermediates) {

	nRandomNumbers = paddedDataChuckSize * (2 * nChoices + 1);
	hBeta = (Real*) calloc(sizeof(Real), paddedBetaSize);

	if (rng) {
		hRng = (Real*) calloc(sizeof(Real), paddedDataChuckSize * (2 * nChoices + 1)); // Two draws per datum
	}

	if (intermediates) {
		hEta = (Real*) calloc(sizeof(Real), getEtaSize());
		hLogZMin = (Real*) calloc(sizeof(Real), paddedDataChuckSize);
		hU = (Real*) calloc(sizeof(Real), getEtaSize());
		hWeight	= (Real*) calloc(sizeof(Real), getEtaSize());
	}

	return DeviceError::NO_ERROR;
}

int CMDIWP::initializeMixturePrior() {

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
	return DeviceError::NO_ERROR;
}

CMDIWP::~CPU_MDI_worker_parallel() {
//	if (hX) {
//		free(hX);
//	}
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
	// TODO No need for these if using std::vector
}

int CMDIWP::initializeOutProducts() {
	return DeviceError::NO_ERROR;
}

void CMDIWNP::computeWeightedOuterProducts() {

	const uint dim = betaSize;

	Spd totalWeightedXXt(dim);
	std::vector<double> totalWeightedUtility(dim);

	// Form XtWX
	uint length = paddedDataChuckSize * nChoices;
	for (uint k1 = 0; k1 < betaSize; ++k1) {
		for (uint k2 = k1; k2 < betaSize; ++k2) {
			Real* x1 = &hX[0] + length * k1;
			Real* x2 = &hX[0] + length * k2;
			Real* w = &hWeight[0];
			Real sum = 0;
			// Manually unroll, length is always a multiple of 4
 			for (uint i = 0; i < length; i += 4) {
 				sum += *x1++ * *w++ * *x2++;
 				sum += *x1++ * *w++ * *x2++;
 				sum += *x1++ * *w++ * *x2++;
 				sum += *x1++ * *w++ * *x2++;
			}
			totalWeightedXXt(k1, k2) = sum;
		}
	}

	// Form XtWU
	for (uint k = 0; k < betaSize; ++k) {
		Real* x = &hX[0] + length * k;
		Real* w = &hWeight[0];
		Real* u = &hU[0];
		Real sum = 0;
		for (uint i = 0; i < length; ++i) {
			sum += *x++ * *w++ * *u++;
		}
		totalWeightedUtility[k] = sum;
	}

	Ptr<MlvsCdSuf_ml> try2 = new MlvsCdSuf_ml(totalWeightedXXt, totalWeightedUtility);
	suf_->clear();
	suf_->add(try2);

#ifdef DEBUG_PRINT
	  cerr << endl << "XtWU: ";
		printfVector(&totalWeightedUtility[0], dim);
		cerr << endl;
		cerr << totalWeightedXXt;
//		cerr << "SUF = " << endl << suf_->xtwx() << endl;
//		cerr << "SUF = " << suf_->xtwu() << endl;
		cerr << "New parallel" << endl;
//		exit(0);
#endif
}


void CMDIWP::computeWeightedOuterProducts() {

	const uint dim = nSubjectVars * nNonZeroChoices;
	const uint dim2 = dim * (nSubjectVars + 1) / 2;

	Spd totalWeightedXXt(dim);
	std::vector<double> totalWeightedUtility(dim);

	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		uint offset = (choice - 0) * nSubjectVars;
		const Real* thisWeight = hWeight + choice * paddedDataChuckSize;

		// Compute outer products
		//		uint termTriangle = 0;
		for (uint i = 0; i < nSubjectVars; i++) {
			for (uint j = 0; j < nSubjectVars; j++) {
				Real sum = 0;
				if (j >= i) {
					//					const Real* thisXtX = hXtX + paddedDataChuckSize * termTriangle;
					const Real* thisXi = &hX[0] + paddedDataChuckSize * i;
					const Real* thisXj = &hX[0] + paddedDataChuckSize * j;
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
		for (uint i = 0; i < nSubjectVars; i++) {
			const Real* thisX = &hX[0] + paddedDataChuckSize * i;
			Real sum = 0;
			for (uint index = 0; index < dataChuckSize; index++) {
				sum += thisX[index] * thisUtility[index] * thisWeight[index];
			}
			totalWeightedUtility[offset + i] = sum;
		}
	}

	Ptr<MlvsCdSuf_ml> try2 = new MlvsCdSuf_ml(totalWeightedXXt,
			totalWeightedUtility);
	suf_->clear();
	suf_->add(try2);

#ifdef DEBUG_PRINT
	cerr << "XtWU: ";
		printfVector(&totalWeightedUtility[0], dim);
		cerr << endl;
		cerr << totalWeightedXXt;
//		cerr << "SUF = " << endl << suf_->xtwx() << endl;
//		cerr << "SUF = " << suf_->xtwu() << endl;
		cerr << endl;
//		exit(0);
#endif
}

void CMDIWNP::uploadBeta() {
	MultinomialLogitModel *ptrMLM = dynamic_cast<MultinomialLogitModel*> (mlm);
	const double* modelBeta = ptrMLM->beta().data();
	for (uint k = 0; k < betaSize; ++k) {
		hBeta[k] = (Real) modelBeta[k];
	}
#ifdef FIX_BETA

	//Real t = {1.05723, -0.0483772, 0.0957759, 0.637142, -0.243552, -0.0128604, 0.522964, 0.347092};
	for (uint i = 0; i < betaSize; ++i) {
		hBeta[i] = 0.1 * (i + 1);
	}
#endif
}

void CMDIWP::uploadBeta() {
	MultinomialLogitModel *ptrMLM = dynamic_cast<MultinomialLogitModel*> (mlm);
	const double* modelBeta = ptrMLM->beta().data();
	for (uint choice = 0; choice < nNonZeroChoices; choice++) {
		for (uint k = 0; k < nSubjectVars; k++) {
			hBeta[choice * nSubjectVars + k] = modelBeta[choice * nSubjectVars + k];
		}
	}
#ifdef FIX_BETA
	for (uint i = 0; i < betaSize; ++i) {
		hBeta[i] = 0.1 * (i + 1);
	}
#endif
}

void CMDIWP::generateRngNumbers() {

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
}

void CMDIWP::computeEta() {
	for (uint index = 0; index < dataChuckSize; index++) {
		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
			Real sum = 0;
			uint rowOffset = paddedDataChuckSize * choice + index;
			for (uint k = 0; k < nSubjectVars; k++) {
				sum += hX[index + paddedDataChuckSize * k] * hBeta[nSubjectVars	* choice + k];
			}
			// TODO Adjust for down sampling
			hEta[rowOffset] = sum;
		}
	}
#ifdef DEBUG_PRINT
	std::cerr << "beta: ";
	for (uint i = 0; i < betaSize; ++i) {
		std::cerr << hBeta[i] << " ";
	}
	std::cerr << std::endl;
	std::cerr << "eta: ";
	for (uint i = 0; i < getEtaSize(); ++i) {
		std::cerr << hEta[i] << " ";
	}
	std::cerr << std::endl << std::endl;
//	exit(0);
#endif
}

void CMDIWNP::reduceEta() {
	for (uint index = 0; index < dataChuckSize; index++) {
		Real sum = 0;
		for (uint choice = 0; choice < nChoices; choice++) {
			sum += exp(hEta[getRow(index, choice)]);
		}
		hLogZMin[index] = sum;
	}
#ifdef DEBUG_PRINT
	for (uint i = 0; i < dataChuckSize; ++i) {
		std::cerr << hLogZMin[i] << " ";
	}
	std::cerr << std::endl;
//	exit(-1);
#endif
}

void CMDIWP::reduceEta() {
	for (uint index = 0; index < dataChuckSize; index++) {
		Real sum = 1;
		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
			sum += exp(hEta[paddedDataChuckSize * choice + index]);
		}
		hLogZMin[index] = sum;
	}
#ifdef DEBUG_PRINT
	for (uint i = 0; i < dataChuckSize; ++i) {
		std::cerr << hLogZMin[i] << " ";
	}
	std::cerr << std::endl;
//	exit(-1);
#endif
}

void CMDIWNP::sampleAllU() {
	for (uint index = 0; index < dataChuckSize; index++) {

		uint y = hY[index];
		Real zmin = -log(hRng[index]) / hLogZMin[index];

		for (uint choice = 0; choice < nChoices; choice++) {
			uint thread = index + paddedDataChuckSize * choice;

			Real z = zmin;

			if (choice != y) {
				z += -log(hRng[index + paddedDataChuckSize * (2 * choice + 1)])
						/ exp(hEta[thread]);
			}
			Real minusLogZ = -log(z);

			// sample single U
			Real unif = hRng[index + paddedDataChuckSize * (2 * choice + 2)];
			uint K = sampleOneU(minusLogZ - hEta[thread], unif); // No need to store
			hU[thread] = minusLogZ - hMu[K];
			hWeight[thread] = hSigmaSqInv[K];
		}
	}
#ifdef DEBUG_PRINT
	std::cerr << endl << "u: ";
	for (uint i = 0; i < getEtaSize(); ++i) {
		std::cerr << hU[i] << " ";
	}
	std::cerr << std::endl;
	std::cerr << endl << "wgts: ";
	util::printfVector(hWeight, paddedDataChuckSize * nChoices);
//	exit(0);
#endif
}

void CMDIWP::sampleAllU() {
	for (uint index = 0; index < dataChuckSize; index++) {

		uint y = hY[index];
		Real zmin = -log(hRng[index]) / hLogZMin[index];

		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
			uint thread = index + paddedDataChuckSize * choice;

			Real z = zmin;

			if ((choice + 1) != y) {
				z += -log(hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 1)])
						/ exp(hEta[thread]);
			}
			Real minusLogZ = -log(z);

			// sample single U
			Real unif = hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 2)];
			uint K = sampleOneU(minusLogZ - hEta[thread], unif); // No need to store
			hU[thread] = minusLogZ - hMu[K];
			hWeight[thread] = hSigmaSqInv[K];
		}
	}
#ifdef DEBUG_PRINT
	std::cerr << "u: ";
	for (uint i = 0; i < getEtaSize(); ++i) {
		std::cerr << hU[i] << " ";
	}
	std::cerr << std::endl;
//	exit(0);
#endif
}

//void CMDIWP::sampleAllU() {
//	for (uint index = 0; index < dataChuckSize; index++) {
//		uint y = hY[index];
//#ifdef UNIFY_RNG
//#ifdef LOG_TEST
//		Real zmin = -log(hRng[index]) / hLogZMin[index];
//#else
//		Real logzmin = log(-log(hRng[index])) - log(hLogZMin[index]);
//#endif
//#else
//		Real logzmin = hLogZMin[index];
//#endif
//		for (uint choice = 0; choice < nNonZeroChoices; choice++) {
//			uint thread = index + paddedDataChuckSize * choice;
//#ifdef LOG_TEST
//			Real z = zmin;
//#else
//			Real logz;
//#endif
//			if ((choice + 1) != y) {
//#ifdef LOG_TEST
//				z += -log(hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 1)])
//						/ exp(hEta[thread]);
//			}
//			Real minusLogZ = -log(z);
//#else
//			Real tmp = log(-log(hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 1)])) // Convoluted indices to match with serial version
//			- hEta[thread];
//			logz = -lse2(logzmin, tmp);
//		} else {
//			logz = -logzmin;
//		}
//		hU[thread] = logz; // No need to store here; store once later
//#endif
//
//			// sample single U
//			Real unif = hRng[index + paddedDataChuckSize * (2 * (choice + 1) + 2)];
//#ifdef LOG_TEST
//			uint K = sampleOneU(minusLogZ - hEta[thread], unif); // No need to store
//			hU[thread] = minusLogZ - hMu[K];
//#else
//			uint K = sampleOneU(hU[thread] - hEta[thread], unif); // No need to store
//			hU[thread] -= hMu[K]; // Just store here once
//#endif
//			hWeight[thread] = hSigmaSqInv[K];
//		}
//	}
//}

uint CMDIWP::sampleOneU(Real x, Real unif) {
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

void CMDIWNP::operator()() {
	uploadBeta();
	generateRngNumbers();
	computeEta();
	reduceEta();
	sampleAllU();
	computeWeightedOuterProducts();
}

void CMDIWP::operator()() {
	uploadBeta();
	generateRngNumbers();
	computeEta();
	reduceEta();
	sampleAllU();
	computeWeightedOuterProducts();
}

} // mlvs_impute

} // BOOM
