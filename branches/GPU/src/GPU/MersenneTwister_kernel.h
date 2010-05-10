/*
 * MersenneTwister_kernel.h
 *
 *  Created on: Apr 18, 2010
 *      Author: msuchard
 */

#ifndef MERSENNETWISTER_KERNEL_H_
#define MERSENNETWISTER_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

cudaError_t gpuRandomMT(
		float *rng,
		int nTotal);

void loadMTGPU(const char *fname);
void seedMTGPU(unsigned int seed);

#endif /* MERSENNETWISTER_KERNEL_H_ */
