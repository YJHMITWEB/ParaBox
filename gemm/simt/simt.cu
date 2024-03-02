#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// nvcc simt.cu -o simt -arch=compute_80 -code=sm_80 -lcublas -lineinfo --ptxas-options=-v -maxrregcount=128

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}


__global__ void shmemSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    __shared__ float shmA[BM][BK];
    __shared__ float shmB[BK][BN];
    
    float resultC[TM][TN] = {0.0};
    int ldArow_shm = tid >> 1;
    int ldAcol_shm = (tid & 1) << 2;
    int ldBrow_shm = tid >> 5;
    int ldBcol_shm = (tid & 31) << 2;

    int ldArow_gm  = blockIdx.y * BM + ldArow_shm;
    int ldBcol_gm  = blockIdx.x * BN + ldBcol_shm;

    int step = 0;
    int total_step = K / BK;
    for (;step < total_step ; ++step) {

        int ldAcol_gm  = step * BK + ldAcol_shm;
        int ldBrow_gm  = step * BK + ldBrow_shm;
        int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
        FLOAT4(shmA[ldArow_shm][ldAcol_shm]) = FLOAT4(a[ldA_gm]);
        int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
        FLOAT4(shmB[ldBrow_shm][ldBcol_shm]) = FLOAT4(b[ldB_gm]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    int compRow = threadIdx.y * TM + tm;
                    int compCol = threadIdx.x * TN + tn;
                    resultC[tm][tn] += shmA[compRow][k] * shmB[k][compCol];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int stCrow_gm = blockIdx.y * BM + threadIdx.y * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int stCcol_gm = blockIdx.x * BN + threadIdx.x * TN + j;
            int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
            FLOAT4(c[stC_gm]) = FLOAT4(resultC[i][j]);
        }
    }
}


__global__ void shmemNoBCSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    __shared__ float shmA[BK][BM];
    __shared__ float shmB[BK][BN];
    
    float ldA_rg[4];
    float ldA_comp_rg[TM];
    float ldB_comp_rg[TN];

    float resultC[TM][TN] = {0.0};
    int ldArow_shm = tid >> 1;
    int ldAcol_shm = (tid & 1) << 2;
    int ldBrow_shm = tid >> 5;
    int ldBcol_shm = (tid & 31) << 2;

    int ldArow_gm  = blockIdx.y * BM + ldArow_shm;
    int ldBcol_gm  = blockIdx.x * BN + ldBcol_shm;

    int step = 0;
    int total_step = K / BK;
    for (;step < total_step ; ++step) {

        int ldAcol_gm  = step * BK + ldAcol_shm;
        int ldBrow_gm  = step * BK + ldBrow_shm;
        int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
        FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
        shmA[ldAcol_shm][ldArow_shm] = ldA_rg[0];
        shmA[ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
        shmA[ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
        shmA[ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
        
        int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
        FLOAT4(shmB[ldBrow_shm][ldBcol_shm]) = FLOAT4(b[ldB_gm]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[k][threadIdx.y * TM / 2]);
            FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[k][threadIdx.y * TM / 2 + BM / 2]);
            FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[k][threadIdx.x * TN / 2]);
            FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[k][threadIdx.x * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + threadIdx.y * TM / 2 + i;
        int stCcol_gm = blockIdx.x * BN + threadIdx.x * TN / 2;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + threadIdx.y * TM / 2 + i + BM / 2;
        int stCcol_gm = blockIdx.x * BN + threadIdx.x * TN / 2;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i + TM / 2][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i + TM / 2][4]);
    }
}


__global__ void shmemNoBCDoubleBufSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    __shared__ float shmA[2][BK][BM];
    __shared__ float shmB[2][BK][BN];
    
    float ldA_rg[4];
    float ldB_rg[4];
    float ldA_comp_rg[TM];
    float ldB_comp_rg[TN];

    float resultC[TM][TN] = {0.0};
    int ldArow_shm = tid >> 1;
    int ldAcol_shm = (tid & 1) << 2;
    int ldBrow_shm = tid >> 5;
    int ldBcol_shm = (tid & 31) << 2;

    int ldArow_gm  = blockIdx.y * BM + ldArow_shm;
    int ldBcol_gm  = blockIdx.x * BN + ldBcol_shm;

    int step = 1;
    
    int ldAcol_gm  = 0 * BK + ldAcol_shm;
    int ldBrow_gm  = 0 * BK + ldBrow_shm;
    int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
    FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
    shmA[0][ldAcol_shm][ldArow_shm] = ldA_rg[0];
    shmA[0][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
    shmA[0][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
    shmA[0][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
    
    int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
    FLOAT4(shmB[0][ldBrow_shm][ldBcol_shm]) = FLOAT4(b[ldB_gm]);
    __syncthreads();

    int total_step = K / BK;
    for (;step < total_step ; ++step) {
        int doubleBufcomp_ind = (step - 1) & 1;

        int ldAcol_gm  = step * BK + ldAcol_shm;
        int ldBrow_gm  = step * BK + ldBrow_shm;
        int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
        FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
        int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
        FLOAT4(ldB_rg) = FLOAT4(b[ldB_gm]);

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][threadIdx.y * TM / 2]);
            FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][threadIdx.y * TM / 2 + BM / 2]);
            FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][threadIdx.x * TN / 2]);
            FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][threadIdx.x * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
                }
            }
        }

        shmA[1 - doubleBufcomp_ind][ldAcol_shm][ldArow_shm] = ldA_rg[0];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
        FLOAT4(shmB[1 - doubleBufcomp_ind][ldBrow_shm][ldBcol_shm]) = FLOAT4(ldB_rg);
        
        __syncthreads();
    }

    int doubleBufcomp_ind = (total_step - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k){
        FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][threadIdx.y * TM / 2]);
        FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][threadIdx.y * TM / 2 + BM / 2]);
        FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][threadIdx.x * TN / 2]);
        FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][threadIdx.x * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; ++tm){
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn){
                resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + threadIdx.y * TM / 2 + i;
        int stCcol_gm = blockIdx.x * BN + threadIdx.x * TN / 2;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + threadIdx.y * TM / 2 + i + BM / 2;
        int stCcol_gm = blockIdx.x * BN + threadIdx.x * TN / 2;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i + TM / 2][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i + TM / 2][4]);
    }
}



__global__ void shmemNoBCDoubleBufWarpTileSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    int laneId = tid & 31;

    // we use 2x4 warptile
    int ldArow_shm2rg = ((warpId >> 2) << 5) + ((laneId >> 2) << 2);
    int ldBcol_shm2rg = ((warpId & 3) << 4) + ((laneId & 3) << 2);

    __shared__ float shmA[2][BK][BM];
    __shared__ float shmB[2][BK][BN];
    
    float ldA_rg[4];
    float ldB_rg[4];
    float ldA_comp_rg[TM];
    float ldB_comp_rg[TN];

    float resultC[TM][TN] = {0.0};
    int ldArow_shm = tid >> 1;
    int ldAcol_shm = (tid & 1) << 2;
    int ldBrow_shm = tid >> 5;
    int ldBcol_shm = (tid & 31) << 2;

    int ldArow_gm  = blockIdx.y * BM + ldArow_shm;
    int ldBcol_gm  = blockIdx.x * BN + ldBcol_shm;

    int step = 1;
    
    int ldAcol_gm  = 0 * BK + ldAcol_shm;
    int ldBrow_gm  = 0 * BK + ldBrow_shm;
    int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
    FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
    shmA[0][ldAcol_shm][ldArow_shm] = ldA_rg[0];
    shmA[0][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
    shmA[0][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
    shmA[0][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
    
    int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
    FLOAT4(shmB[0][ldBrow_shm][ldBcol_shm]) = FLOAT4(b[ldB_gm]);
    __syncthreads();

    int total_step = K / BK;
    for (;step < total_step ; ++step) {
        int doubleBufcomp_ind = (step - 1) & 1;

        int ldAcol_gm  = step * BK + ldAcol_shm;
        int ldBrow_gm  = step * BK + ldBrow_shm;
        int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
        FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
        int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
        FLOAT4(ldB_rg) = FLOAT4(b[ldB_gm]);

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg]);
            FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg + BM / 2]);
            FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg]);
            FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
                }
            }
        }

        shmA[1 - doubleBufcomp_ind][ldAcol_shm][ldArow_shm] = ldA_rg[0];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
        FLOAT4(shmB[1 - doubleBufcomp_ind][ldBrow_shm][ldBcol_shm]) = FLOAT4(ldB_rg);
        
        __syncthreads();
    }

    int doubleBufcomp_ind = (total_step - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k){
        FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg]);
        FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg + BM / 2]);
        FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg]);
        FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; ++tm){
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn){
                resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + ldArow_shm2rg + i;
        int stCcol_gm = blockIdx.x * BN + ldBcol_shm2rg;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + ldArow_shm2rg + i + BM / 2;
        int stCcol_gm = blockIdx.x * BN + ldBcol_shm2rg;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i + TM / 2][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i + TM / 2][4]);
    }
}



__global__ void shmemNoBCDoubleBufWarpTileLessRgSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // we use 2x4 warptile
    int ldArow_shm2rg = (((tid >> 5) >> 2) << 5) + (((tid & 31) >> 2) << 2);
    int ldBcol_shm2rg = (((tid >> 5) & 3) << 4) + (((tid & 31) & 3) << 2);

    __shared__ float shmA[2][BK][BM];
    __shared__ float shmB[2][BK][BN];
    
    float ldA_rg[4];
    float ldB_rg[4];
    float ldA_comp_rg[TM];
    float ldB_comp_rg[TN];

    float resultC[TM][TN] = {0.0};
    int ldArow_shm = tid >> 1;
    int ldAcol_shm = (tid & 1) << 2;
    int ldBrow_shm = tid >> 5;
    int ldBcol_shm = (tid & 31) << 2;

    int ldArow_gm  = blockIdx.y * BM + ldArow_shm;
    int ldBcol_gm  = blockIdx.x * BN + ldBcol_shm;

    int step = 1;
    
    int ldAcol_gm  = 0 * BK + ldAcol_shm;
    int ldBrow_gm  = 0 * BK + ldBrow_shm;
    int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
    FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
    shmA[0][ldAcol_shm][ldArow_shm] = ldA_rg[0];
    shmA[0][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
    shmA[0][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
    shmA[0][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
    
    int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
    FLOAT4(shmB[0][ldBrow_shm][ldBcol_shm]) = FLOAT4(b[ldB_gm]);
    __syncthreads();

    int total_step = K / BK;
    for (;step < total_step ; ++step) {
        int doubleBufcomp_ind = (step - 1) & 1;

        ldAcol_gm  = step * BK + ldAcol_shm;
        ldBrow_gm  = step * BK + ldBrow_shm;
        ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
        FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
        ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
        FLOAT4(ldB_rg) = FLOAT4(b[ldB_gm]);

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg]);
            FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg + BM / 2]);
            FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg]);
            FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
                }
            }
        }

        shmA[1 - doubleBufcomp_ind][ldAcol_shm][ldArow_shm] = ldA_rg[0];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
        FLOAT4(shmB[1 - doubleBufcomp_ind][ldBrow_shm][ldBcol_shm]) = FLOAT4(ldB_rg);
        
        __syncthreads();
    }

    int doubleBufcomp_ind = (total_step - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k){
        FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg]);
        FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg + BM / 2]);
        FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg]);
        FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; ++tm){
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn){
                resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + ldArow_shm2rg + i;
        int stCcol_gm = blockIdx.x * BN + ldBcol_shm2rg;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + ldArow_shm2rg + i + BM / 2;
        int stCcol_gm = blockIdx.x * BN + ldBcol_shm2rg;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i + TM / 2][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i + TM / 2][4]);
    }
}



__global__ __launch_bounds__(256, 2) void shmemNoBCDoubleBufWarpTileLessRgZSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // we use 2x4 warptile
    int warpId = tid >> 5;
    int laneId = tid & 31;
    int ldArow_shm2rg = ((warpId >> 1) << 4) + ((((laneId >> 4) << 1) + ((laneId & 15) & 1)) << 2);
    int ldBcol_shm2rg = ((warpId & 1) << 5) + (((laneId & 15) >> 1) << 2);

    __shared__ float shmA[2][BK][BM];
    __shared__ float shmB[2][BK][BN];
    
    float ldA_rg[4];
    float ldB_rg[4];
    float ldA_comp_rg[TM];
    float ldB_comp_rg[TN];

    float resultC[TM][TN] = {0.0};
    int ldArow_shm = tid >> 1;
    int ldAcol_shm = (tid & 1) << 2;
    int ldBrow_shm = tid >> 5;
    int ldBcol_shm = (tid & 31) << 2;

    int ldArow_gm  = blockIdx.y * BM + ldArow_shm;
    int ldBcol_gm  = blockIdx.x * BN + ldBcol_shm;

    int step = 1;
    
    int ldAcol_gm  = 0 * BK + ldAcol_shm;
    int ldBrow_gm  = 0 * BK + ldBrow_shm;
    int ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
    FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
    shmA[0][ldAcol_shm][ldArow_shm] = ldA_rg[0];
    shmA[0][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
    shmA[0][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
    shmA[0][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
    
    int ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
    FLOAT4(shmB[0][ldBrow_shm][ldBcol_shm]) = FLOAT4(b[ldB_gm]);
    __syncthreads();

    int total_step = K / BK;
    for (;step < total_step ; ++step) {
        int doubleBufcomp_ind = (step - 1) & 1;

        ldAcol_gm  = step * BK + ldAcol_shm;
        ldBrow_gm  = step * BK + ldBrow_shm;
        ldA_gm     = OFFSET(ldArow_gm, ldAcol_gm, K);
        FLOAT4(ldA_rg) = FLOAT4(a[ldA_gm]);
        ldB_gm     = OFFSET(ldBrow_gm, ldBcol_gm, N);
        FLOAT4(ldB_rg) = FLOAT4(b[ldB_gm]);

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg]);
            FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg + BM / 2]);
            FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg]);
            FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
                }
            }
        }

        shmA[1 - doubleBufcomp_ind][ldAcol_shm][ldArow_shm] = ldA_rg[0];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 1][ldArow_shm] = ldA_rg[1];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 2][ldArow_shm] = ldA_rg[2];
        shmA[1 - doubleBufcomp_ind][ldAcol_shm + 3][ldArow_shm] = ldA_rg[3];
        FLOAT4(shmB[1 - doubleBufcomp_ind][ldBrow_shm][ldBcol_shm]) = FLOAT4(ldB_rg);
        
        __syncthreads();
    }

    int doubleBufcomp_ind = (total_step - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k){
        FLOAT4(ldA_comp_rg[0]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg]);
        FLOAT4(ldA_comp_rg[4]) = FLOAT4(shmA[doubleBufcomp_ind][k][ldArow_shm2rg + BM / 2]);
        FLOAT4(ldB_comp_rg[0]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg]);
        FLOAT4(ldB_comp_rg[4]) = FLOAT4(shmB[doubleBufcomp_ind][k][ldBcol_shm2rg + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; ++tm){
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn){
                resultC[tm][tn] += ldA_comp_rg[tm] * ldB_comp_rg[tn];
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + ldArow_shm2rg + i;
        int stCcol_gm = blockIdx.x * BN + ldBcol_shm2rg;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int stCrow_gm = blockIdx.y * BM + ldArow_shm2rg + i + BM / 2;
        int stCcol_gm = blockIdx.x * BN + ldBcol_shm2rg;
        int stC_gm    = OFFSET(stCrow_gm, stCcol_gm, N);
        FLOAT4(c[stC_gm]) = FLOAT4(resultC[i + TM / 2][0]);
        FLOAT4(c[stC_gm + BN / 2]) = FLOAT4(resultC[i + TM / 2][4]);
    }
}





float testMaxError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testCublasMaxError(const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}


float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}


int main() {

    const int TESTNUM = 15;
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    // const int K_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int outer_repeat = 10, inner_repeat = 1;
    // const int outer_repeat = 1, inner_repeat = 1;

    {
        printf("\nKernal = cublas\n");

        {
            const int M = 512, N = 512, K = 512;
            float max_error = testCublasMaxError(M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testCublasPerformance(M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = naiveSgemm\n");

        const int BM = 32, BN = 32;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            naiveSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN, BM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN, BM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = shmSgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            shmemSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = shmNoBCSgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            shmemNoBCSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = shmNoBCDoubleBufSgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            shmemNoBCDoubleBufSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = shmNoBCDoubleBufWarpTileSgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            shmemNoBCDoubleBufWarpTileSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = shmNoBCDoubleBufWarpTileLessRgSgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            shmemNoBCDoubleBufWarpTileLessRgSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    {
        printf("\nKernal = shmNoBCDoubleBufWarpTileLessRgZSgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            shmemNoBCDoubleBufWarpTileLessRgZSgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    return 0;
}