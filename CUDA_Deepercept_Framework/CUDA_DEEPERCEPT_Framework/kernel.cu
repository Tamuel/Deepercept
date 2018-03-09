#include "tensor.h"
#include <stdexcept>
#include <time.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cublas_api.h>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

// 1. The maximum number of threads in the block is limited to 1024. This is the product of whatever your
//    threadblock dimensions are(x*y*z).For example(32, 32, 1) creates a block of 1024 threads. (33, 32, 1) is not legal, since 33 * 32 * 1 > 1024.
// 2. The maximum x - dimension is 1024. (1024, 1, 1) is legal. (1025, 1, 1) is not legal.
// 3. The maximum y - dimension is 1024. (1, 1024, 1) is legal. (1, 1025, 1) is not legal.
// 4. The maximum z - dimension is 64. (1, 1, 64) is legal. (2, 2, 64) is also legal. (1, 1, 65) is not legal.
// Also, threadblock dimensions of 0 in any position are not legal.

// Size of grid can be more upper most 2^31 - 1. (Where compute capability 3.0 and newer)


using namespace std;

enum OP {SUM, SUBTRACT, ELT_MULT, MAT_MULT, SCALAR_MULT};

__global__ void scalarMultKernel(Tensor* t_out, const Tensor* t1, const dtype scalar) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < t_out->size())
		(*t_out).data[i] = scalar * (*t1).data[i];
}

__global__ void addKernel(Tensor* t_out, const Tensor* t1, const Tensor* t2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < t_out->size())
		(*t_out).data[i] = (*t1).data[i] + (*t2).data[i];
}

__global__ void subKernel(Tensor* t_out, const Tensor* t1, const Tensor* t2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < t_out->size())
		(*t_out).data[i] = (*t1).data[i] - (*t2).data[i];
}

__global__ void multiplyKernel(Tensor* t_out, const Tensor* t1, const Tensor* t2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < t_out->size())
		(*t_out).data[i] = (*t1).data[i] * (*t2).data[i];
}

// Matrix multiplication with tiling
__global__ void matMultKernel(Tensor* tOut, const Tensor* t1, const Tensor* t2,
	const int t1_row, const int t1_t2_bet, const int t2_col) {
	// Use shared memory for speeding up
	__shared__ dtype s_m[BLOCK_DIM][BLOCK_DIM];
	__shared__ dtype s_n[BLOCK_DIM][BLOCK_DIM];

	register int t_c = threadIdx.x, t_r = threadIdx.y;
	register int b_c = blockIdx.x,  b_r = blockIdx.y;

	register int row = b_r * blockDim.y + t_r;
	register int col = b_c * blockDim.x + t_c;

	if (row >= t1_row || col >= t2_col)
		return;

	register int t1_c, t1_r, t2_c, t2_r;
	register dtype pValue = 0;

	for (int m = 0; m < t1_t2_bet / BLOCK_DIM + 1; ++m) {
		t1_r = row;
		t1_c = m * BLOCK_DIM + t_c;
		t2_r = m * BLOCK_DIM + t_r;
		t2_c = col;
		s_m[t_r][t_c] = t1_r < t1_row    && t1_c < t1_t2_bet ? (*t1).data[t1_r * t1_t2_bet + t1_c] : 0;
		s_n[t_r][t_c] = t2_r < t1_t2_bet && t2_c < t2_col    ? (*t2).data[t2_r * t2_col + t2_c] : 0;
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < BLOCK_DIM; k++)
			pValue += s_m[t_r][k] * s_n[k][t_c];

		__syncthreads();
	}
	(*tOut)(row, col) = pValue;
}

// Matrix multiplication with unrolling
__global__ void matrixMul_unroll(Tensor* t_out, const Tensor* t1, const Tensor* t2, int w_t1, int w_t2) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Declaration of the shared memory array As used to
	// store the sub-matrix of t1
	__shared__ dtype s_t1[BLOCK_DIM * BLOCK_DIM];

	dtype cv[BLOCK_DIM] = { 0 };

	// Index of the first sub-matrix of t1 processed by the block
	int aBegin = w_t1 * BLOCK_DIM * by;

	// Index of the last sub-matrix of t1 processed by the block
	int aEnd = aBegin + w_t1;

	// Step size used to iterate through the sub-matrices of t1
	int aStep = BLOCK_DIM;

	// Index of the first sub-matrix of t2 processed by the block
	int bBegin = BLOCK_DIM * VECTOR_SIZE * bx;

	// Step size used to iterate through the sub-matrices of t2
	int bStep = BLOCK_DIM * w_t2;

	int cBegin = w_t2 * BLOCK_DIM * by + VECTOR_SIZE * BLOCK_DIM * bx;

	// Loop over all the sub-matrices of t1 and t2
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) {

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		dtype *Ap = &(*t1).data[a + w_t1 * ty + tx];
		dtype *ap = &s_t1[ty + BLOCK_DIM * tx];
		#pragma unroll
		for (int i = 0; i < BLOCK_DIM; i += VECTOR_SIZE) {
			ap[i] = Ap[w_t1 * i];
		}
		__syncthreads();

		ap = &s_t1[0];
		dtype *bp = &(*t2).data[b + BLOCK_DIM * ty + tx];

		#pragma unroll      
		for (int i = 0; i < BLOCK_DIM; i++) {
			dtype bv = bp[0];
			cv[0] += ap[0] * bv;
			cv[1] += ap[1] * bv;
			cv[2] += ap[2] * bv;
			cv[3] += ap[3] * bv;
			cv[4] += ap[4] * bv;
			cv[5] += ap[5] * bv;
			cv[6] += ap[6] * bv;
			cv[7] += ap[7] * bv;
			cv[8] += ap[8] * bv;
			cv[9] += ap[9] * bv;
			cv[10] += ap[10] * bv;
			cv[11] += ap[11] * bv;
			cv[12] += ap[12] * bv;
			cv[13] += ap[13] * bv;
			cv[14] += ap[14] * bv;
			cv[15] += ap[15] * bv;
			cv[16] += ap[16] * bv;
			cv[17] += ap[17] * bv;
			cv[18] += ap[18] * bv;
			cv[19] += ap[19] * bv;
			cv[20] += ap[20] * bv;
			cv[21] += ap[21] * bv;
			cv[22] += ap[22] * bv;
			cv[23] += ap[23] * bv;
			cv[24] += ap[24] * bv;
			cv[25] += ap[25] * bv;
			cv[26] += ap[26] * bv;
			cv[27] += ap[27] * bv;
			cv[28] += ap[28] * bv;
			cv[29] += ap[29] * bv;
			cv[30] += ap[30] * bv;
			cv[31] += ap[31] * bv;
			ap += BLOCK_DIM;
			bp += w_t2;
		}

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	dtype *Cp = &(*t_out).data[cBegin];
	Cp += BLOCK_DIM * ty + tx;
	int cStep = w_t2;

	#pragma unroll
	for (int i = 0; i < BLOCK_DIM; i++) {
		Cp[0] = cv[i]; Cp += cStep;
	}
}


//Tensor* matrixOperation(OP operation, Tensor* const t1, Tensor* const t2, dtype scalar = 0.0) {
//	int ROW = t1->shape()[0];
//	int COL;
//	if (t2 != NULL && t2->dimension() == 2)
//		COL = t2->shape()[1];
//	else
//		COL = 1;
//	int BETWEEN;
//	if (t1->dimension() == 2)
//		BETWEEN = t1->shape()[1];
//	else
//		BETWEEN = 1;
//	dim3 dimBlock;
//	dim3 dimGrid;
//
//	Tensor* tOut;
//	if (t1->dimension() == 2)
//		tOut = new Tensor({ ROW, COL }, false);
//	else if (t1->dimension() == 1)
//		tOut = new Tensor({ ROW }, false);
//
//	Tensor* dev_t1 = 0;
//	Tensor* dev_t2 = 0;
//	Tensor* dev_tOut = 0;
//
//	time_t start = clock();
//	dev_t1 = t1->sendToDevice();
//	if (t2 != NULL)
//		dev_t2 = t2->sendToDevice();
//	dev_tOut = tOut->sendToDevice();
//	time_t end = clock();
//	printf("Sending Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//
//	// Check availability
//	switch (operation) {
//	case SUM:
//	case SUBTRACT:
//	case ELT_MULT:
//		if (!t1->isSame(*t2)) {
//			cout << "Two matrices shapes are different!" << endl;
//			return NULL;
//		}
//		break;
//	case MAT_MULT:
//		if (t1->shape()[1] != t2->shape()[0]) {
//			cout << "Cannot mutiply two matrices!" << endl;
//			return NULL;
//		}
//		break;
//	case SCALAR_MULT:
//		break;
//	}
//
//	// Set block and grid size
//	switch (operation) {
//	case SUM:
//	case SUBTRACT:
//	case ELT_MULT:
//	case SCALAR_MULT:
//		dimBlock.x = BLOCK_SIZE * BLOCK_SIZE;
//		dimBlock.y = 1;
//		dimGrid.x = tOut->size() / dimBlock.x + 1;
//		dimGrid.y = 1;
//		break;
//	case MAT_MULT:
//		dimBlock.x = BLOCK_SIZE;
//		dimBlock.y = BLOCK_SIZE;
//		dimGrid.x = (COL + BLOCK_SIZE - 1) / dimBlock.x;
//		dimGrid.y = (ROW + BLOCK_SIZE - 1) / dimBlock.y;
//		break;
//	}
//
//	start = clock();
//	// Process operation
//	switch (operation) {
//	case SUM:
//		addKernel <<<dimGrid, dimBlock >>> (dev_tOut, dev_t1, dev_t2);
//		break;
//	case SUBTRACT:
//		subKernel <<<dimGrid, dimBlock >>> (dev_tOut, dev_t1, dev_t2);
//		break;
//	case ELT_MULT:
//		multiplyKernel <<<dimGrid, dimBlock>>> (dev_tOut, dev_t1, dev_t2);
//		break;
//	case MAT_MULT:
//		matMultKernel <<<dimGrid, dimBlock>>> (dev_tOut, dev_t1, dev_t2, ROW, BETWEEN, COL);
//		break;
//	case SCALAR_MULT:
//		scalarMultKernel <<<dimGrid, dimBlock >>> (dev_tOut, dev_t1, scalar);
//		break;
//	}
//	cudaThreadSynchronize();
//	end = clock();
//	printf("Calc Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//
//	start = clock();
//	tOut->retrievDataFromDevice(dev_tOut);
//	end = clock();
//	printf("Retrieving Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//
//	cudaFree(dev_t1);
//	cudaFree(dev_t2);
//	cudaFree(dev_tOut);
//
//	return tOut;
//}

void getGpuInformation() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number : %d\n", i);
		printf("\tDevice Name : %s\n", prop.name);
		printf("\tCompute Capability : %d.%d\n", prop.major, prop.minor);
		printf("\tGPU Clock Rate (GHz) : %f\n", float(prop.clockRate) / (1000.0 * 1000.0));
		printf("\tMemory Clock Rate (GHz) : %f\n", float(prop.memoryClockRate) / (1000.0 * 1000.0));
		printf("\tMemory Size (GB) : %f\n", static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024));
		printf("\tMemory Bus Width (bits) : %b\n", prop.memoryBusWidth);
		printf("\tPeak Memory Bandwitdh (GB/s) : %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6);
		printf("\tNumber of Multi Processor : %d\n", prop.multiProcessorCount);
		printf("\tMax Grid Size : [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\tMax Threads Dimension : [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("\tMax Threads Per Block : %d\n", prop.maxThreadsPerBlock);
		printf("\tMax Threads Per Multi Processor : %d\n", prop.maxThreadsPerMultiProcessor);
	}
}

void print(char* title, float* src, int h, int w)
{
	cout << title << endl;

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			int index = y * w + x;
			printf("%5.0f", src[index]);
		}
		printf("\n");
	}
	printf("\n");
}

//int main(void) {
//	getGpuInformation();
//	//Tensor a({ 3, 3 }, "A");
//	//Tensor b({ 3, 3 }, "B");
//
//	//dtype aData[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//	//dtype bData[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
//	//a.print();
//	//b.print();
//	//a[aData]; // Set data
//	//b[bData]; // Set data
//
//	//Tensor k = a;
//	//k.setName("k");
//	//a.print();
//	//b.print();
//	//k.print();
//	//Tensor* c = matrixOperation(SUM, &a, &b);
//	//c->setName("A + B");
//	//c->print();
//	//Tensor* d = matrixOperation(SUBTRACT, &a, &b);
//	//d->setName("A - B");
//	//d->print();
//	//Tensor* e = matrixOperation(ELT_MULT, &a, &b);
//	//e->setName("A and B Elt Mult");
//	//e->print();
//	//Tensor* e1 = matrixOperation(MAT_MULT, &a, &b);
//	//e1->setName("A and B Mat Mult");
//	//e1->print();
//
//	//Tensor m1({ 3, 2 }, "m1");
//	//Tensor m2({ 2, 3 }, "m2");
//	//m1[{1, 2,
//	//	3, 4,
//	//	5, 6}];
//	//m2[{6, 5, 4,
//	//	3, 2, 1}];
//	//Tensor* m3 = matrixOperation(MAT_MULT, &m1, &m2);
//	//m3->print();
//
//	int size = 4096;
//	int size2 = 4096;
//	Tensor mult1({ size, size2 }, "mult1");
//	Tensor mult2({ size2, size }, "mult2");
//
//	mult1[1];
//	cout << mult1(0) << " " << mult1(1) << " " << mult1(2) << endl;
//	mult2[1];
//	cout << mult2(0) << " " << mult2(1) << " " << mult2(2) << endl;
//
//	time_t start = clock();
//	Tensor* f = matrixOperation(MAT_MULT, &mult1, &mult2);
//	//Tensor* g = matrixOperation(MAT_MULT, f, &mult2);
//	//Tensor* h = matrixOperation(MAT_MULT, g, &mult1);
//	printf("%lf, %lf, %lf\n", (*f)(0), (*f)(1), (*f)(2));
//	time_t end = clock();
//	printf("Whole Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//
//	// cuBLAS
//	const int M = size;
//	const int N = size2;
//	Tensor cu1({ size, size2 });
//	Tensor cu2({ size2, size });
//	Tensor cu3({ size, size }, false);
//	float* cu1_d;
//	float* cu2_d;
//	float* cu3_d;
//
//	cu1[1];
//	cu2[1];
//
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//	start = clock();
//	cudaMalloc((void**)&cu1_d, sizeof(dtype) * cu1.size());
//	cudaMalloc((void**)&cu2_d, sizeof(dtype) * cu2.size());
//	cudaMalloc((void**)&cu3_d, sizeof(dtype) * cu3.size());
//	cudaMemcpy(cu1_d, cu1.data, sizeof(dtype) * cu1.size(), cudaMemcpyHostToDevice);
//	cudaMemcpy(cu2_d, cu2.data, sizeof(dtype) * cu2.size(), cudaMemcpyHostToDevice);
//	end = clock();
//	printf("Cublas prepare Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//	start = clock();
//	float alpha = 1.0;
//	float beta = 0.0;
//	cublasStatus_t stat = cublasSgemm(
//		handle, CUBLAS_OP_N, CUBLAS_OP_N,
//		M, M, N,
//		&alpha,
//		cu1_d, M,
//		cu2_d, M,
//		&beta,
//		cu3_d, N
//	);
//	cudaDeviceSynchronize();
//	cudaThreadSynchronize();
//	end = clock();
//	printf("Cublas calc Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//	start = clock();
//	cu3.retrievDataFromDevice(cu3_d);
//	end = clock();
//	printf("Cublas retrieve Time : %lf\n", double(end - start) / CLOCKS_PER_SEC);
//	//cu3.print();
//
//	cudaFree(cu1_d);
//	cudaFree(cu2_d);
//	cudaFree(cu3_d);
//
//}
