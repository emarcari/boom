cmake_minimum_required(VERSION 2.6)
project(MLAUX)

if (APPLE)
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
else (APPLE)
     set(CMAKE_C_COMPILER "gcc44")
     set(CMAKE_CXX_COMPILER "g++44")
endif (APPLE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O2 -pg")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O2 -pg")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

set(CMAKE_VERBOSE_MAKEFILE on )

find_package(CUDA)

include_directories(
	${CMAKE_SOURCE_DIR} 
	${CMAKE_SOURCE_DIR}/..
	/System/Library/Frameworks/vecLib.framework/Headers/
	/Users/msuchard/Documents/Projects/boost_1_49_0
	/home/msuchard/boost_1_39_0
	/usr/local/atlas/include
	)
	
link_directories(
	${CMAKE_SOURCE_DIR}/../..
	/opt/local/lib
	/Users/msuchard/Documents/Projects/boost_1_49_0/stage/lib
	/home/msuchard/boost_1_39_0/stage/lib
#	/usr/lib
#	/usr/local/atlas/lib
	/usr/local/lib
	)

set(SOURCE_FILES
    gpu_example.cpp
#	MLAuxMixSampler_test.cpp
#	MersenneTwister_kernel.cu
#	GPU_MDI_worker.cpp
#	GPU_MDI_worker_kernel.cu
	)	
	
if (APPLE)
set(BOOST_LIBRARIES
	boost_program_options
	boost_filesystem
	boost_thread
	boost_signals
	boost_system
	)
else (APPLE)
set(BOOST_LIBRARIES
	boost_program_options-gcc41-mt
	boost_filesystem-gcc41-mt
	boost_thread-gcc41-mt
	boost_signals-gcc41-mt
	boost_system-gcc41-mt
	)
endif (APPLE)
	
set(MATH_LIBRARIES
	m
	)

if (APPLE)
set(MATH_LIBRARIES
	${MATH_LIBRARIES}
	lapack
	blas
	cblas
	atlas
	)
else (APPLE)
set(MATH_LIBRARIES
	${MATH_LIBRARIES}
	gslcblas
	blas
#	cblas
	lapack
#	lapack_atlas
#	atlas
	)
endif (APPLE)
	
if(CUDA_FOUND)	
	add_definitions(-DCUDA -DCUDA_ENABLED)
 	if(APPLE)
# 		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
# 		set(CUDA_64_BIT_DEVICE_CODE OFF)
	else (APPLE)
	     add_definitions(-DGTX280)	
 	endif(APPLE)		
 	set(CUDA_NVCC_FLAGS -arch compute_11)
 	cuda_add_executable(mlaux ${SOURCE_FILES})	
else(CUDA_FOUND)
	add_executable(mlaux ${SOURCE_FILES})
endif(CUDA_FOUND)

target_link_libraries(mlaux BOOM ${BOOST_LIBRARIES} ${MATH_LIBRARIES})


