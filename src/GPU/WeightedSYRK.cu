#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>

#define TILE_DIM_LOG    (5)
#define TILE_DIM        (1 << TILE_DIM_LOG)
#define TILE_SIZE       (TILE_DIM * TILE_DIM)

#define SYRK_LOG_THREAD_COUNT   (9)
#define SYRK_THREAD_COUNT       (1 << SYRK_LOG_THREAD_COUNT)

struct WeightedSYRKParams {
    const float *A;
    const float *B;
    const float *W;   
    float *C;
    int   n;
    int   k;   
    int   lda;
    int   ldb;   
    int   ldc;
};

#if (((TILE_DIM * TILE_DIM) % SYRK_THREAD_COUNT) !=0)
    #error tile size and threadcount do not divide evenly!
#endif

#if ((SYRK_THREAD_COUNT % TILE_DIM) !=0)
    #error TILE_DIM does not divide THREAD_COUNT evenly
#endif

#define COL_INCR               (SYRK_THREAD_COUNT / TILE_DIM)
#define C_ELEMS_PER_THREAD     (TILE_SIZE / SYRK_THREAD_COUNT)
#define B_ELEMS_PER_THREAD     (TILE_SIZE / SYRK_THREAD_COUNT)
#define A_ELEMS_PER_THREAD     (TILE_SIZE / SYRK_THREAD_COUNT)

typedef WeightedSYRKParams Params;

typedef unsigned int uint;

__device__  float atomicAddf(float* address, float val) {
   float old = *address, assumed;

   do {
      assumed = old;
      old = __int_as_float(
            atomicCAS((unsigned int*)address,
      __float_as_int(assumed),
      __float_as_int(val + assumed)));
   } while (assumed != old);

   return old;
}

// Index functions for accessing global memory and cached tiles. All matrices are column-major
inline __device__ int colOfsA(const Params& params) {
    return params.lda;
}

inline __device__ int colOfsB(const Params& params) {
    return params.ldb;
}

inline __device__ int colOfsC(const Params& params) {
    return params.ldc;
}

inline __device__ int colOfsAA() {
    return TILE_DIM + 1;
}

inline __device__ int colOfsBB() {
    return TILE_DIM + 1;
}

inline __device__ int idxA(int row, int col, const Params& params) {
    return colOfsA(params) * col + row;
}

inline __device__ int idxB(int row, int col, const Params& params) {
    return colOfsB(params) * col + row;
}

inline __device__ int idxC(int row, int col, const Params& params) {
    return colOfsC(params) * col + row;
}

inline __device__ int getIdxAA(int row, int col) {
    return __umul24(colOfsAA(), col) + row;
}

inline __device__ int getIdxBB(int row, int col) {
    return __umul24(colOfsBB(), col) + row;
}

inline __device__ void accumulateDotProduct32(float& dp, 
        const float* AA, const float* BB,
        const uint li, const uint lj) {
    dp += (AA[li+ 0] * BB[lj+ 0]);     
    dp += (AA[li+ 1] * BB[lj+ 1]);     
    dp += (AA[li+ 2] * BB[lj+ 2]);     
    dp += (AA[li+ 3] * BB[lj+ 3]);     
    dp += (AA[li+ 4] * BB[lj+ 4]);     
    dp += (AA[li+ 5] * BB[lj+ 5]);     
    dp += (AA[li+ 6] * BB[lj+ 6]);     
    dp += (AA[li+ 7] * BB[lj+ 7]);     
    dp += (AA[li+ 8] * BB[lj+ 8]);     
    dp += (AA[li+ 9] * BB[lj+ 9]);     
    dp += (AA[li+10] * BB[lj+10]);     
    dp += (AA[li+11] * BB[lj+11]);     
    dp += (AA[li+12] * BB[lj+12]);     
    dp += (AA[li+13] * BB[lj+13]);     
    dp += (AA[li+14] * BB[lj+14]);     
    dp += (AA[li+15] * BB[lj+15]);     
    dp += (AA[li+16] * BB[lj+16]);     
    dp += (AA[li+17] * BB[lj+17]);     
    dp += (AA[li+18] * BB[lj+18]);     
    dp += (AA[li+19] * BB[lj+19]);     
    dp += (AA[li+20] * BB[lj+20]);     
    dp += (AA[li+21] * BB[lj+21]);     
    dp += (AA[li+22] * BB[lj+22]);     
    dp += (AA[li+23] * BB[lj+23]);     
    dp += (AA[li+24] * BB[lj+24]);     
    dp += (AA[li+25] * BB[lj+25]);     
    dp += (AA[li+26] * BB[lj+26]);     
    dp += (AA[li+27] * BB[lj+27]);     
    dp += (AA[li+28] * BB[lj+28]);     
    dp += (AA[li+29] * BB[lj+29]);     
    dp += (AA[li+30] * BB[lj+30]);     
    dp += (AA[li+31] * BB[lj+31]);     
}

inline __device__ void accumulate2DotProduct32(float& dp1, float& dp2,
        const float* AA, const float* BB,
        const uint li, const uint lj, const uint ljOfs) {                                                 
    dp1 += (AA[li+ 0] * BB[lj+ 0]);              
    dp2 += (AA[li+ 0] * BB[lj+(ljOfs)+ 0]);      
    dp1 += (AA[li+ 1] * BB[lj+ 1]);              
    dp2 += (AA[li+ 1] * BB[lj+(ljOfs)+ 1]);      
    dp1 += (AA[li+ 2] * BB[lj+ 2]);              
    dp2 += (AA[li+ 2] * BB[lj+(ljOfs)+ 2]);      
    dp1 += (AA[li+ 3] * BB[lj+ 3]);              
    dp2 += (AA[li+ 3] * BB[lj+(ljOfs)+ 3]);      
    dp1 += (AA[li+ 4] * BB[lj+ 4]);              
    dp2 += (AA[li+ 4] * BB[lj+(ljOfs)+ 4]);      
    dp1 += (AA[li+ 5] * BB[lj+ 5]);              
    dp2 += (AA[li+ 5] * BB[lj+(ljOfs)+ 5]);      
    dp1 += (AA[li+ 6] * BB[lj+ 6]);              
    dp2 += (AA[li+ 6] * BB[lj+(ljOfs)+ 6]);      
    dp1 += (AA[li+ 7] * BB[lj+ 7]);              
    dp2 += (AA[li+ 7] * BB[lj+(ljOfs)+ 7]);      
    dp1 += (AA[li+ 8] * BB[lj+ 8]);              
    dp2 += (AA[li+ 8] * BB[lj+(ljOfs)+ 8]);      
    dp1 += (AA[li+ 9] * BB[lj+ 9]);              
    dp2 += (AA[li+ 9] * BB[lj+(ljOfs)+ 9]);      
    dp1 += (AA[li+10] * BB[lj+10]);              
    dp2 += (AA[li+10] * BB[lj+(ljOfs)+10]);      
    dp1 += (AA[li+11] * BB[lj+11]);              
    dp2 += (AA[li+11] * BB[lj+(ljOfs)+11]);      
    dp1 += (AA[li+12] * BB[lj+12]);              
    dp2 += (AA[li+12] * BB[lj+(ljOfs)+12]);      
    dp1 += (AA[li+13] * BB[lj+13]);              
    dp2 += (AA[li+13] * BB[lj+(ljOfs)+13]);      
    dp1 += (AA[li+14] * BB[lj+14]);              
    dp2 += (AA[li+14] * BB[lj+(ljOfs)+14]);      
    dp1 += (AA[li+15] * BB[lj+15]);              
    dp2 += (AA[li+15] * BB[lj+(ljOfs)+15]);      
    dp1 += (AA[li+16] * BB[lj+16]);              
    dp2 += (AA[li+16] * BB[lj+(ljOfs)+16]);      
    dp1 += (AA[li+17] * BB[lj+17]);              
    dp2 += (AA[li+17] * BB[lj+(ljOfs)+17]);      
    dp1 += (AA[li+18] * BB[lj+18]);              
    dp2 += (AA[li+18] * BB[lj+(ljOfs)+18]);      
    dp1 += (AA[li+19] * BB[lj+19]);              
    dp2 += (AA[li+19] * BB[lj+(ljOfs)+19]);      
    dp1 += (AA[li+20] * BB[lj+20]);              
    dp2 += (AA[li+20] * BB[lj+(ljOfs)+20]);      
    dp1 += (AA[li+21] * BB[lj+21]);              
    dp2 += (AA[li+21] * BB[lj+(ljOfs)+21]);      
    dp1 += (AA[li+22] * BB[lj+22]);              
    dp2 += (AA[li+22] * BB[lj+(ljOfs)+22]);      
    dp1 += (AA[li+23] * BB[lj+23]);              
    dp2 += (AA[li+23] * BB[lj+(ljOfs)+23]);      
    dp1 += (AA[li+24] * BB[lj+24]);              
    dp2 += (AA[li+24] * BB[lj+(ljOfs)+24]);      
    dp1 += (AA[li+25] * BB[lj+25]);              
    dp2 += (AA[li+25] * BB[lj+(ljOfs)+25]);      
    dp1 += (AA[li+26] * BB[lj+26]);              
    dp2 += (AA[li+26] * BB[lj+(ljOfs)+26]);      
    dp1 += (AA[li+27] * BB[lj+27]);              
    dp2 += (AA[li+27] * BB[lj+(ljOfs)+27]);      
    dp1 += (AA[li+28] * BB[lj+28]);              
    dp2 += (AA[li+28] * BB[lj+(ljOfs)+28]);      
    dp1 += (AA[li+29] * BB[lj+29]);              
    dp2 += (AA[li+29] * BB[lj+(ljOfs)+29]);      
    dp1 += (AA[li+30] * BB[lj+30]);              
    dp2 += (AA[li+30] * BB[lj+(ljOfs)+30]);      
    dp1 += (AA[li+31] * BB[lj+31]);              
    dp2 += (AA[li+31] * BB[lj+(ljOfs)+31]);  
}

inline __device__ void accumulateDotProductN(float& dp,
        const float* AA, const float* BB, 
        uint& li, uint& lj, uint& ll) {                                       
    while (ll) {                            
        dp += (AA[li++] * BB[lj++]);   
        ll -= 1;                            
    }                                       
}

inline __device__ void accumulate2DotProductN(float& dp1, float& dp2,
        const float* AA, const float* BB,
        uint& li, uint& lj, uint& ll,  
        const uint ljOfs) {                                               
    do {                                             
        dp1 += (AA[li+ 0] * BB[lj+ 0]);         
        dp2 += (AA[li+ 0] * BB[lj+(ljOfs)+ 0]); 
        li++;                                        
        lj++;                                        
        ll--;                                        
    } while(ll);                                     
}

template <bool FullTiles, bool Weighted, typename RealType>
__global__
void weighted_ssyrk (Params params) {
    unsigned int i, j, l, ii, jj, ll, tid = threadIdx.x;

    unsigned int tidLo = (tid & (TILE_DIM - 1));
    unsigned int tidHi = (tid >> TILE_DIM_LOG);  

    i = blockIdx.y * TILE_DIM;
    j = blockIdx.x * TILE_DIM;

    if (j > i) return;
    
    __shared__ RealType AA[(TILE_DIM + 1) * TILE_DIM]; // avoid bank conflicts
    __shared__ RealType BB[(TILE_DIM + 1) * TILE_DIM]; // avoid bank conflicts
    __shared__ RealType WW[TILE_DIM];
        
    RealType dp0 = 0.0f;
    RealType dp1 = 0.0f;

    for (l = 0; l < params.k; l += TILE_DIM) {
        unsigned int llLimit = min ((l + TILE_DIM), params.k);
        // Wait until previous work is done
        __syncthreads ();
        
        // Load from global memory;
        if (Weighted) {        
             if (tidHi == 0) {
                uint ll = l + tidLo;
                if(FullTiles || ll < llLimit) {
                    uint idxWW = tidLo;
                    uint addrW = ll;
                    WW[idxWW] = params.W[addrW];            
                }  
             }
            __syncthreads();
        }         
                            
        // A is transposed, B is not transposed
        ll = l + tidLo;
        if(FullTiles || ll < llLimit) {
            ii = i + tidHi;
            if(FullTiles || ii < params.n) {
                unsigned int idxAA;
                unsigned int addrA;
                idxAA = getIdxAA(tidLo, tidHi);
                addrA = idxA(ll, ii, params);
                if (Weighted) {
                     AA[idxAA] = params.A[addrA] * WW[tidLo];
                } else {
                     AA[idxAA] = params.A[addrA];
                }

                ii += COL_INCR;
                if(FullTiles || ii < params.n) {
                    idxAA += COL_INCR * colOfsAA();
                    addrA += COL_INCR * colOfsA(params);
                    if (Weighted) {
                         AA[idxAA] = params.A[addrA] * WW[tidLo];
                    } else {
                         AA[idxAA] = params.A[addrA];                
                    }
                }
            }
        }
        
        ll = l + tidLo;
        if(FullTiles || ll < llLimit) {
            jj = j + tidHi;
            if(FullTiles || jj < params.n) {
                unsigned int idxBB;
                unsigned int addrB;
                idxBB = getIdxBB(tidLo,tidHi);
                addrB = idxB(ll,jj,params);
                BB[idxBB] = params.B[addrB];

                jj += COL_INCR;
                if(FullTiles || jj < params.n) {
                    idxBB += COL_INCR * colOfsBB();
                    addrB += COL_INCR * colOfsB(params);
                    BB[idxBB] = params.B[addrB];
                }
            }
        }    
        __syncthreads ();           
        
        // Take inner product of elements
        ii = tidLo;
        if(FullTiles || ii < (params.n - i)) {
            unsigned int z = llLimit - l;
            jj = tidHi;
            if(FullTiles || z == 32) {
                if(FullTiles || (jj + COL_INCR) < (params.n - j)) {
                    unsigned int li = getIdxAA(0,ii);
                    unsigned int lj = getIdxBB(0,jj);
                    accumulate2DotProduct32(dp0, dp1, AA, BB, li, lj, colOfsBB() * COL_INCR); 
                } else if(jj < (params.n - j)) {
                    unsigned int li = getIdxAA(0,ii);
                    unsigned int lj = getIdxBB(0,jj);
                    accumulateDotProduct32(dp0, AA, BB, li, lj);
                }

            } else {
                if(FullTiles || (jj + COL_INCR) < (params.n - j)) {
                    unsigned int li = getIdxAA(0,ii);
                    unsigned int lj = getIdxBB(0,jj);
                    ll = z;
                    accumulate2DotProductN(dp0, dp1, AA, BB, li, lj, ll, colOfsBB() * COL_INCR);
                } else if(jj < (params.n - j)) {
                    unsigned int li = getIdxAA(0,ii);
                    unsigned int lj = getIdxBB(0,jj);
                    ll = z;
                    accumulateDotProductN(dp0, AA, BB, li, lj, ll);
                }
            }
        }
    }

    // Store results            
    ii = i + tidLo;
    jj = j + tidHi;
    if(FullTiles || (ii < params.n) && (jj < params.n)) {
        unsigned int addrC = idxC(ii,jj,params);

        if (ii >= jj) {
            params.C[addrC] = dp0; // Handle alpha and beta here
        }

        jj += COL_INCR;
        if(FullTiles || jj < params.n) {

            if (ii >= jj) {
                addrC += COL_INCR * colOfsC(params);
                params.C[addrC] = dp1; // Handle alpha and beta here
            }
        }
    }
}

void ripSsyrk(int n, int k,
              const float *A, int lda,
              const float *W,
              float *C, int ldc) {
    WeightedSYRKParams params;

    dim3 grid (((n+TILE_DIM-1)>>TILE_DIM_LOG),
                    ((n+TILE_DIM-1)>>TILE_DIM_LOG));  

    params.n = n;
    params.k = k;
    params.A = A;
    params.lda = lda;
    params.B = A;
    params.ldb = lda;
    params.C = C;
    params.ldc =ldc;
    params.W = W;
 
    int fullTilesOnly = (((n % TILE_DIM) == 0) &&
                     ((k % TILE_DIM) == 0));
                     
    if (fullTilesOnly) {
        weighted_ssyrk<true, true, float> <<<grid, SYRK_THREAD_COUNT>>>(params);
    } else {
        weighted_ssyrk<false, true, float> <<<grid, SYRK_THREAD_COUNT>>>(params);
    }   
}





