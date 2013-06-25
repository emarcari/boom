# This Makefile assumes that boost and blas libraries and headers are available
# on the local machine.

SHELL = /bin/sh

CXXFLAGS = -Isrc -O2
LDFLAGS = -L.
LIBS = -lboost_filesystem -lboost_system -llapack -lblas

all: examples libboom.a

%.o: %.cpp
	${CXX} ${CXXFLAGS} -c -MD $< -o $@ 

BART_SRCS := $(wildcard src/Models/Bart/*.cpp) \
	$(wildcard src/Models/Bart/PosteriorSamplers/*.cpp)
BART_OBJS = ${BART_SRCS:.cpp=.o}

DISTRIBUTION_SRCS := $(wildcard src/distributions/*.cpp)
DISTRIBUTION_OBJS := ${DISTRIBUTION_SRCS:.cpp=.o}

GLM_SRCS := $(wildcard src/Models/Glm/*.cpp) \
	$(wildcard src/Models/Glm/PosteriorSamplers/*.cpp)
GLM_OBJS = ${GLM_SRCS:.cpp=.o}

HIERARCHICAL_SRCS := $(wildcard src/Models/Hierarchical/*.cpp) \
	$(wildcard src/Models/Hierarchical/PosteriorSamplers/*.cpp)
HIERARCHICAL_OBJS = ${HIERARCHICAL_SRCS:.cpp=.o}

HMM_SRCS := $(wildcard src/Models/HMM/*.cpp) \
	$(wildcard src/Models/HMM/Clickstream/*.cpp) \
	$(wildcard src/Models/HMM/PosteriorSamplers/*.cpp)
HMM_OBJS = ${HMM_SRCS:.cpp=.o}

IRT_SRCS := $(wildcard src/Models/IRT/*.cpp)
IRT_SRCS := $(filter-out \
	src/Models/IRT/ItemSliceSampler.cpp \
	src/Models/IRT/multisubscale_logit_cutpoint_model.cpp \
	src/Models/IRT/PcrNid.cpp \
	src/Models/IRT/Subject_prior.cpp, \
	${IRT_SRCS})
IRT_OBJS = ${IRT_SRCS:.cpp=.o}

LINALG_SRCS := $(wildcard src/LinAlg/*.cpp)
LINALG_OBJS = ${LINALG_SRCS:.cpp=.o}

MATH_SRCS := $(wildcard src/math/cephes/*.cpp)
MATH_OBJS = ${MATH_SRCS:.cpp=.o}

MODELS_SRCS := $(wildcard src/Models/*.cpp) \
	$(wildcard src/Models/Policies/*.cpp) \
	$(wildcard src/Models/PosteriorSamplers/*.cpp) 
MODELS_OBJS = ${MODELS_SRCS:.cpp=.o}

NUMOPT_SRCS := $(wildcard src/numopt/*.cpp)
NUMOPT_OBJS = ${NUMOPT_SRCS:.cpp=.o}

POINTPROCESS_SRCS := $(wildcard src/Models/PointProcess/*.cpp) \
	$(wildcard src/Models/PointProcess/PosteriorSamplers/*.cpp)
POINTPROCESS_OBJS = ${POINTPROCESS_SRCS:.cpp=.o}

RMATH_SRCS := $(wildcard src/Bmath/*.cpp) \
	src/distributions/Rmath_dist.cpp \
	src/distributions/rng.cpp \
	src/distributions/BinomialDistribution.cpp
RMATH_OBJS = ${RMATH_SRCS:.cpp=.o}

SAMPLERS_SRCS := $(wildcard src/Samplers/*.cpp) \
	$(wildcard src/Samplers/Gilks/*.cpp)
SAMPLERS_OBJS = ${SAMPLERS_SRCS:.cpp=.o}

STATESPACE_SRCS := $(wildcard src/Models/StateSpace/*.cpp) \
	$(wildcard src/Models/StateSpace/Filters/*.cpp) \
	$(wildcard src/Models/StateSpace/PosteriorSamplers/*.cpp) \
	$(wildcard src/Models/StateSpace/StateModels/*.cpp)
STATESPACE_OBJS = ${STATESPACE_SRCS:.cpp=.o}

STATS_SRCS := $(wildcard src/stats/*.cpp)
STATS_OBJS = ${STATS_SRCS:.cpp=.o}

TARGETFUN_SRCS := $(wildcard src/TargetFun/*.cpp)
TARGETFUN_OBJS = ${TARGETFUN_SRCS:.cpp=.o}

TIMESERIES_SRCS := $(wildcard src/Models/TimeSeries/*.cpp) \
	$(wildcard src/Models/TimeSeries/PosteriorSamplers/*.cpp)
TIMESERIES_OBJS = ${TIMESERIES_SRCS:.cpp=.o}

UTIL_SRCS := $(wildcard src/cpputil/*.cpp)
UTIL_SRCS := $(filter-out src/cpputil/ProgramOptions.cpp, ${UTIL_SRCS})
UTIL_OBJS = ${UTIL_SRCS:.cpp=.o}

ALL_SRCS = ${BART_SRCS} \
	${DISTRIBUTION_SRCS} \
	${GLM_SRCS} \
	${HIERARCHICAL_SRCS} \
	${HMM_SRCS} \
	${IRT_SRCS} \
	${LINALG_SRCS} \
	${MATH_SRCS} \
	${MODELS_SRCS} \
	${NUMOPT_SRCS} \
	${POINTPROCESS_SRCS} \
	${RMATH_SRCS} \
	${SAMPLERS_SRCS} \
	${STATS_SRCS} \
	${STATESPACE_SRCS} \
	${TARGETFUN_SRCS} \
	${TIMESERIES_SRCS} \
	${UTIL_SRCS}

ALL_OBJS = ${ALL_SRCS:.cpp=.o}

libboom.a: ${ALL_OBJS}
	$(AR) rcs $@ $?

SeasonalStateModel_example: \
    src/Models/StateSpace/StateModels/tests/SeasonalStateModel_example.o \
    libboom.a
	$(CXX) src/Models/StateSpace/StateModels/tests/SeasonalStateModel_example.o $(LDFLAGS) -lboom $(LIBS) -o $@

WeeklyCyclePoissonProcess_example: \
  src/Models/PointProcess/tests/WeeklyCyclePoissonProcess_example.o \
  libboom.a
	$(CXX) src/Models/PointProcess/tests/WeeklyCyclePoissonProcess_example.o  $(LDFLAGS) -lboom $(LIBS) -o $@

binomial_logit_auxmix_sampler_example: \
    src/Models/Glm/tests/binomial_logit_auxmix_sampler_example.o \
    libboom.a
	$(CXX) src/Models/Glm/tests/binomial_logit_auxmix_sampler_example.o $(LDFLAGS) -lboom $(LIBS) -o $@

# TODO(kmillar): enable once the code has been modified to not use Google flags.
# hpoisson_threading_example: \
#   src/Interfaces/R/hpoisson/hpoisson_threading_example.o \
#   libboom.a
# 	$(CXX) src/Interfaces/R/hpoisson/hpoisson_threading_example.o $(LDFLAGS) -lboom $(LIBS) -o $@

examples: \
	SeasonalStateModel_example \
	WeeklyCyclePoissonProcess_example \
	binomial_logit_auxmix_sampler_example
# hpoisson_threading_example

src/Models/tests/zero_inflated_lognormal_test: \
    src/Models/tests/zero_inflated_lognormal_test.o \
    libboom.a
	$(CXX) src/Models/tests/zero_inflated_lognormal_test.o $(LDFLAGS) -lboom $(LIBS) -o $@

# TODO(kmillar): add more tests.
tests: \
	src/Models/tests/zero_inflated_lognormal_test

clean:
	rm `find . -name \*.o` `find . -name \*.d`

-include ${ALL_SRCS:.cpp=.d}
