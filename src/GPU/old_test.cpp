#include <cpputil/DataTable.hpp>
#include <cpputil/DesignMatrix.hpp>


#include <Models/Glm/PosteriorSamplers/MLAuxMixSampler.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS.hpp>
#include <Models/Glm/PosteriorSamplers/MLVS_split.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Glm/MLogitSplit.hpp>

#include <Models/MvnModel.hpp>
#include <stats/FreqDist.hpp>
#include <iomanip>
#include <distributions.hpp>
#include <vector>
#include <cpputil/file_utils.hpp>

#include <boost/tuple/tuple.hpp>

#include <time.h>
#include <sys/time.h>

#include "GPU_MDI_worker.hpp"

using namespace std;
using namespace BOOM;
using BOOM::uint;

namespace BOOM{
  typedef MultinomialLogitModel MLM;
  typedef MLogitSplit MLS;
  typedef MLogitBase MLB;
  typedef std::vector<Ptr<CategoricalData> > Responses;

  inline double p_value(double t){ return 2*pnorm(-fabs(t));}
  void test_mle(Ptr<MLB>);
  Ptr<MLM> make_mod(Responses y, const DesignMatrix & X, double inc_prob,
                    const string & hist, uint nthreads, int computeMode);
//  Ptr<MLogitSplit> make_mod_fast(Responses y, const DesignMatrix & X,
//                                 double inc_prob, const string &hist,
//                                 const vector<string> &labs, uint nthreads);
  void mcmc(Ptr<MLB>, uint niter, uint ping, uint writeInterval);
  void set_fname(Ptr<MLM> mod, string hist);
  void set_fnames(Ptr<MLS> mod, string hist, const std::vector<string> & labs);
}

int main(int argc, char **argv){

  double inc_prob = .50;

  uint  nthreads=1;
//   bool useGPU = false;
  int mode = ComputeMode::CPU;
  std::string fileName = "AutoPref.txt";
//  if(argc>1) sscanf(argv[1], "%u", &nthreads);
  if (argc > 2) {
    mode = ComputeMode::parseComputeModel(argv[2]);
  }

  if (argc > 1) {
	  fileName = argv[1];
  }

  // get matrix of subject data.
  DataTable dat(fileName, true, "\t");

  std::vector<std::string> response_levels(3);
  response_levels[0] = "Family";
  response_levels[1] = "Sporty";
  response_levels[2] = "Work";

  Responses y = dat.get_nominal(6,1);
  set_order(y,response_levels);

  uint inc0[3] = {2,3,4};
  std::vector<uint> inc(inc0, inc0+3);
  DesignMatrix X = dat.design(inc, true,1);

  std::string hist;
  Ptr<MLM> mod;
  Ptr<MLS> splitmod;
  std::vector<std::string> labs(response_levels.begin()+1, response_levels.end());

  uint niter_lg = 500;

  uint n = X.nrow();
//  Mat trash(n,50);
//  trash.randomize();
//  X = cbind(X,trash);

  cout << "MLAuxMix with " << ncol(X) << " X's and " << n << " Y's." << endl;
  mod = make_mod(y,X, inc_prob, "hist.mlaux", nthreads, mode);

  struct timeval time1, time2;
  gettimeofday(&time1, NULL);

  mcmc(mod, niter_lg, 100, 10);

  gettimeofday(&time2, NULL);
  double timeInSec = time2.tv_sec - time1.tv_sec +
		  (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
  cout << endl << "MCMC update duration: " << timeInSec << endl;

}
namespace BOOM{
  //----------------------------------------------------------------------
  void set_fname(Ptr<MLM> mod, string hist){
    string base = add_to_path(pwd(), hist);
    mod->set_param_filename(base+"/beta.hist");
  }
  //----------------------------------------------------------------------
  void set_fnames(Ptr<MLS> mod, string hist, const std::vector<string> & labs){
    std::vector<string> fnames;
    string base=add_to_path(pwd(), hist);
    uint n = labs.size();
    for(uint i=0; i<n; ++i){
      string tail = "/beta." + labs[i] + ".hist";
      fnames.push_back(base+tail);
    }
    mod->set_param_filenames(fnames);
  }
  //----------------------------------------------------------------------
  void mcmc(Ptr<MLB> mod, uint niter, uint ping, uint writeInterval){
    mod->io_params(CLEAR);
    mod->track_progress(ping);
    mod->set_bufsize(ping);
    for(uint i=0; i<niter; ++i){
      mod->sample_posterior();
      if (i % writeInterval == 0) {
    	  mod->io_params(WRITE);
      }
    }
    mod->io_params(FLUSH);
  }

  //----------------------------------------------------------------------
  Ptr<MLM> make_mod(Responses y, const DesignMatrix & X, double inc_prob,
                    const string &hist, uint nthreads, int computeMode){
    NEW(MLM, mod)(y,X);
    uint p = mod->beta_size(false);
   // mod->drop_all_slopes();
    uint nvars = X.ncol();
    Vec b(p,0);
    Spd Sigma(p);
    Sigma.set_diag(100.0);
    NEW(MvnModel, beta_prior)(b,Sigma);
	NEW(MLAuxMixSampler, sam)(mod.get(), beta_prior, nthreads, computeMode);
    mod->set_method(sam);
    set_fname(mod,hist);
    return mod;
  }
//  //----------------------------------------------------------------------
//  void test_mle(Ptr<MLB> mod){
//    Mat Hess;
//    Vec beta;
//    double max_loglike;
//    boost::tie(max_loglike, beta, Hess) = mod->mle_result();
//    cout << " MLE(beta) = " << beta << endl;
//    cout << " Maximum loglike = " << max_loglike << endl;
//    cout << " Hessian is " << endl << Hess << endl;
//    Hess*= -1;
//    cout << "Variance Matrix = " << endl << Hess.inv() << endl ;
//
//    Vec se = sqrt(Hess.inv().diag());
//    cout << setw(12) << left << "Variable"
//         << setw(12) << left << "Std Error"
//         << setw(12) << left << "t-ratio"
//         << setw(12) << left << "p-value" << endl
//         << setw(12) << left << "----------"
//         << setw(12) << left << "----------"
//         << setw(12) << left << "----------"
//         << setw(12) << left << "----------" << endl;
//
//    for(uint j=0; j<se.size(); ++j){
//      cout << setw(12) << left <<  beta[j]
//           << setw(12) << left << se[j]
//           << setw(12) << left << beta[j]/se[j]
//           << setw(12) << left << p_value(beta[j]/se[j]) << endl;
//    }
//  }
}
