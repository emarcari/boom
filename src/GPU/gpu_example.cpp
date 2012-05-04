#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Glm/MvnGivenX.hpp>
#include <Models/Glm/PosteriorSamplers/MLAuxMixSampler.hpp>

#include <cpputil/string_utils.hpp>
#include <cpputil/Split.hpp>
#include <cpputil/ProgramOptions.hpp>

#include <vector>
#include <fstream>
#include <string>

using namespace BOOM;
using std::ifstream;

void ReadChoiceData(const std::string &datafile,
                    std::vector<uint> *choice,
                    std::vector<Matrix> *choice_characteristics,
                    Matrix *subject_characteristics) {
  ifstream in(datafile.c_str());
  std::string line;

  std::vector<std::string> lines;
  int count = 0;
  while (getline(in, line)) {
    if (++count % 1000 == 0) {
      cout << "read line " << count << endl;
    }
    if(is_all_white(line)) continue;
    lines.push_back(line);
  }

  int nlines = lines.size();

  int number_of_potential_choices = nlines - 1;
  int number_of_choice_sets = number_of_potential_choices / 4;
  int xdim = 79;
  int choice_set_size = 4;

  std::vector<Matrix> choice_x;
  Mat observation_level_choice_matrix(choice_set_size, xdim);

  subject_characteristics->resize(number_of_choice_sets, 1);
  choice->resize(number_of_choice_sets);

  StringSplitter split;
  std::vector<std::string> vnames = split(lines[0]);
  if(vnames[84] != "y1"){
    cout << "vnames[84] should be 'y1', but it is " << vnames[84] << endl;
  }
  if(strip_white_space(vnames[85]) != "y1"){
    cout << "vnames[85] should be 'y2', but it is [" << vnames[85] << "]" << endl;
  }
  assert(strip_white_space(vnames[84]) == "y1");
  assert(strip_white_space(vnames[85]) == "y2");

  int observation_number = -1;
  cout << "processing input" << endl;
  for (int line_number = 1; line_number < nlines; ++line_number) {
    if (line_number % 1000 == 0) {
      cout << "processing line " << line_number << endl;
    }
    std::vector<std::string> fields = split(lines[line_number]);
    //    int resp_id = atol(fields[0].c_str());
    //    int version = atol(fields[1].c_str());
    //    int scenario = atol(fields[2].c_str());
    int alternative = atol(fields[3].c_str());
    --alternative;  // Convert from unit-offset to zero-offset.

    //    double intercept = atof(fields[4].c_str());
    for (int pos = 5; strip_white_space(vnames[pos])[0] == 'x'; ++pos) {
      observation_level_choice_matrix(alternative, pos - 5) = atof(fields[pos].c_str());
    }
    if(alternative == 0) ++ observation_number;
    if(alternative == 3) choice_x.push_back(observation_level_choice_matrix);

    // Fill in subject level intercepts.
    (*subject_characteristics)(observation_number, 0) = 1.0;

    int forced_choice = atol(strip_white_space(fields[84]).c_str());
    //    int free_choice = atol(strip_white_space(fields[85]).c_str());

    if (forced_choice == 1) {
      (*choice)[observation_number] = alternative;
    }
  }

  *choice_characteristics = choice_x;
}

int main(int argc, char **argv) {
  cout << "setting up program options" << endl;
//  ProgramOptions options;
//  options.add_option<std::string>("datafile", "The data file to process");
//  options.add_option<int>("niter", "Number of MCMC iterations");
//  cout << "processing command line" << endl;
//  options.process_command_line(argc, argv);
//  cout << "done processing command line" << endl;

  std::vector<uint> choice;                   // A single entry for each observation [0..3]
  std::vector<Matrix> choice_characteristics; // Indexed by observation number, potential choice, variable
  Matrix subject_characteristics;             // Indexed by observation number,

//  std::string datafile = options.get_required_option<std::string>("datafile");
    std::string datafile = "disguised-data-1.txt";  

  std::cout << "reading data" << endl;
  ReadChoiceData(datafile,
                 &choice,
                 &choice_characteristics,
                 &subject_characteristics);
  std::cout << "done reading data" << endl;

  std::cout << "building model" << endl;
  NEW(MultinomialLogitModel, model)(make_catdat_ptrs(choice),
                                    subject_characteristics,
                                    choice_characteristics);
  std::cout << "done building model" << endl;

  // Set the prior
  NEW(VectorParams, beta_prior_mean)(model->beta_size(false), 0.0);
  NEW(UnivParams, beta_prior_sample_size)(1.0);
  std::cout << "setting prior" << endl;
  NEW(MvnGivenXMultinomialLogit, beta_prior)(beta_prior_mean,
                                             beta_prior_sample_size);
  std::cout << "prior built, setting X" << endl;
  beta_prior->set_x(subject_characteristics,
                    choice_characteristics,
                    model->Nchoices());
  NEW(MLAuxMixSampler, sampler)(model.get(), beta_prior);
  std::cout << "all done with prior" << endl;
  model->set_method(sampler);

//  int niter = options.get_with_default<int>("niter", 1000);
  int niter = 1000;

  for (int i = 0; i < niter; ++i) {
    std::cout << "iteration " << i << endl;
    model->sample_posterior();
    cout << model->beta() << endl;
  }

}
