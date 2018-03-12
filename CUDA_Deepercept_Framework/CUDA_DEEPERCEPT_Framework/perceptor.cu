#include "perceptor.h"
#include "srcMeasure.h"

bool Perceptor::gpuUtilization[] = { 0 };

void Perceptor::getGpuInformation() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		string archName = "";
		int cores = 0;
		int mp = prop.multiProcessorCount;
		switch (prop.major) {
		case 2: // Fermi
			if (prop.minor == 1) cores = mp * 48;
			else cores = mp * 32;
			archName = "Fermi";
			break;
		case 3: // Kepler
			cores = mp * 192;
			archName = "Kepler";
			break;
		case 5: // Maxwell
			cores = mp * 128;
			archName = "Maxwell";
			break;
		case 6: // Pascal
			if (prop.minor == 1) cores = mp * 128;
			else if (prop.minor == 0) cores = mp * 64;
			else printf("Unknown device type\n");
			archName = "Pascal";
			break;
		case 7: // Volta
			if (prop.minor == 0) cores = mp * 64;
			else printf("Unknown device type\n");
			archName = "Volta";
			break;
		default:
			printf("Unknown device type\n");
			break;
		}

		printf("Device Number : %d\n", i);
		printf("\tDevice Name : %s [%s]\n", prop.name, archName.c_str());
		printf("\tCompute Capability : %d.%d\n", prop.major, prop.minor);
		printf("\tGPU Clock Rate (GHz) : %f\n", float(prop.clockRate) / (1000.0 * 1000.0));
		printf("\tMemory Clock Rate (GHz) : %f\n", float(prop.memoryClockRate) / (1000.0 * 1000.0));
		printf("\tMemory Size (GB) : %f\n", static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024));
		printf("\tMemory Bus Width (bits) : %d\n", prop.memoryBusWidth);
		printf("\tPeak Memory Bandwitdh (GB/s) : %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6);
		printf("\tShared Memory per Blocks (KB) : %f\n", prop.sharedMemPerBlock);
		printf("\tNumber of Multi Processor : %d\n", prop.multiProcessorCount);
		printf("\tNumber of Cuda Cores : %d\n", cores);
		printf("\tMax Grid Size : [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\tMax Threads Dimension : [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("\tMax Threads Per Block : %d\n", prop.maxThreadsPerBlock);
		printf("\tMax Threads Per Multi Processor : %d\n", prop.maxThreadsPerMultiProcessor);
	}
}

void Perceptor::getCuDnnVersion() {
	cout << "cuDNN version : " << cudnnGetVersion() << endl;
}

void Perceptor::getCuBlasVersion() {
	int version;
	cublasGetVersion(cuBlasHandle, &version);
	cout << "cuBlas version : " << version << endl;
}
void Perceptor::getGpuDriverVersion() {
	int version;
	cuDriverGetVersion(&version);
	cout << "GPU driver version : " << version << endl;
}

void Perceptor::checkDevice(Tensor* tA, Tensor* tB) {
	checkDevice(tA);
	checkDevice(tB);
	if (tA->deviceId() != tB->deviceId()) {
		cout << "Device of " << tA->name() << " = " << tA->deviceId() << 
			" and device of " << tB->name() << " = " << tB->deviceId() << " are different" << endl;
		exit(EXIT_FAILURE);
	}
	setDevice();
}

void Perceptor::checkDevice(Tensor* tA) {
	if (!tA->haveDevicePtr() && !tA->haveDeviceDataPtr()) {
		if (tA->deviceId() != deviceId()) {
			cout << "Device of " << tA->name() << " = " << tA->deviceId() << " is automatically chanaged to " << deviceId()
				 << " because " << tA->name() << " was not allocated at device" << endl;
		}
		tA->setDevice(deviceId());
		setDevice();
	}
	else if (deviceId() != tA->deviceId()) {
		cout << "Device of Perceptor = " << deviceId() <<
			" and device of " << tA->name() << " = " << tA->deviceId() << " are different" << endl;
		exit(EXIT_FAILURE);
	}
}

// Input pointer of tensor A and B
// Output is tensor pointer of result
Tensor* Perceptor::matSgemm(Tensor* tA, Tensor* tB, float alpha, float beta) {
	checkDevice(tA, tB);
	if (tA->shape()[1] != tB->shape()[0]) {
		cout << "Cannot multiply " << tA->name() << " and " << tB->name() << endl;
		cout << "Number of " << tA->name() << " columns and number of " << tB->name() << " rows are different" << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate result tensor
	Tensor* t_out = new Tensor({ tA->row(), tB->col() }, false);
	t_out->setDevice(deviceId());
	sendToDevice(t_out);

	// Allocate to device
	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	// cublasSgemm : C = alpha * (OP(A) * OP(B)) + beta * C
	// With row major!
	CuBLAS_ERROR(
		cublasSgemm(
			cuBlasHandle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			tB->col(), tA->row(), tA->col(), // Rows of OP(A), Columns of OP(B), Rows of C
			&alpha, // alpha
			tB->devDataPtr(), tB->col(), // A, leading dimension of A used to store the matrix A
			tA->devDataPtr(), tA->col(), // B, leading dimension of B used to store the matrix B
			&beta, // beta
			t_out->devDataPtr(), tB->col() // C, leading dimension of C
		)
	);
	syncGpuStream();

	return t_out;
}

void Perceptor::matSgemm(Tensor* tOut, Tensor* tA, Tensor* tB, float alpha, float beta) {
	checkDevice(tOut, tA);
	checkDevice(tA, tB);
	if (tA->shape()[1] != tB->shape()[0]) {
		cout << "Cannot multiply " << tA->name() << " and " << tB->name() << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate to device
	if (!tOut->haveDevicePtr())
		sendToDevice(tOut);
	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	// cublasSgemm : C = alpha * (OP(A) * OP(B)) + beta * C
	// With row major!
	CuBLAS_ERROR(
		cublasSgemm(
			cuBlasHandle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			tB->col(), tA->row(), tA->col(), // Rows of OP(A), Columns of OP(B), Rows of C
			&alpha, // alpha
			tB->devDataPtr(), tB->col(), // A, leading dimension of A used to store the matrix A
			tA->devDataPtr(), tA->col(), // B, leading dimension of B used to store the matrix B
			&beta, // beta
			tOut->devDataPtr(), tB->col() // C, leading dimension of C
		)
	);
	syncGpuStream();
}

Tensor* Perceptor::matMult(Tensor* tA, Tensor* tB) {
	return matSgemm(tA, tB, 1, 0);
}

void Perceptor::matMult(Tensor* tOut, Tensor* tA, Tensor* tB) {
	matSgemm(tOut, tA, tB, 1, 0);
}

void Perceptor::matMult(dtype scalA, Tensor* tB) {
	checkDevice(tB);
	// Allocate result tensor
	Tensor* t_out = new Tensor({ tB->row(), tB->col() }, false);
	t_out->setDevice(deviceId());
	sendToDevice(t_out);

	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	CuBLAS_ERROR(
		cublasSscal(
			cuBlasHandle,
			tB->size(),
			&scalA,
			tB->devDataPtr(),
			1
		)
	);

	syncGpuStream();
}

Tensor* Perceptor::matSgeam(Tensor* tA, Tensor* tB, float alpha, float beta) {
	checkDevice(tA, tB);
	if (!tA->isSame(*tB)) {
		cout << "Cannot sum " << tA->name() << " and " << tB->name() << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate result tensor
	Tensor* t_out = new Tensor({ tA->row(), tB->col() }, false);
	t_out->setDevice(deviceId());
	sendToDevice(t_out);

	// Allocate to device
	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	CuBLAS_ERROR(
		cublasSgeam(
			cuBlasHandle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			tB->col(), tA->row(),
			&alpha,
			tB->devDataPtr(), tB->col(),
			&beta,
			tA->devDataPtr(), tA->col(),
			t_out->devDataPtr(), tB->col()
		)
	);
	syncGpuStream();

	return t_out;
}

void Perceptor::matSgeam(Tensor* tOut, Tensor* tA, Tensor* tB, float alpha, float beta) {
	checkDevice(tOut, tA);
	checkDevice(tA, tB);
	if (!tA->isSame(*tB)) {
		cout << "Cannot sum " << tA->name() << " and " << tB->name() << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate to device
	if (!tOut->haveDevicePtr())
		sendToDevice(tOut);
	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	// cublasSgemm : C = alpha * (OP(A) * OP(B)) + beta * C
	// With row major!
	CuBLAS_ERROR(
		cublasSgeam(
			cuBlasHandle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			tB->col(), tA->row(),
			&alpha,
			tB->devDataPtr(), tB->col(),
			&beta,
			tA->devDataPtr(), tA->col(),
			tOut->devDataPtr(), tB->col()
		)
	);
	syncGpuStream();
}

Tensor* Perceptor::matAdd(Tensor* tA, Tensor* tB) {
	return matSgeam(tA, tB, 1, 1);
}

void Perceptor::matAdd(Tensor* tOut, Tensor* tA, Tensor* tB) {
	matSgeam(tOut, tA, tB, 1, 1);
}

Tensor* Perceptor::matSub(Tensor* tA, Tensor* tB) {
	return matSgeam(tA, tB, -1, 1);
}

void Perceptor::matSub(Tensor* tOut, Tensor* tA, Tensor* tB) {
	matSgeam(tOut, tA, tB, -1, 1);
}

__global__ void cuEltwiseMultiplication(dtype* tOut, dtype* tA, dtype* tB, int row, int col) {
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;

	if (y < row && x < col)
		tOut[y * col + x] = tA[y * col + x] * tB[y * col + x];
}

// Return = tA * tB (Haramard Product)
Tensor* Perceptor::matEltMult(Tensor* tA, Tensor* tB) {
	checkDevice(tA, tB);
	if (!tA->isSame(*tB)) {
		cout << "Cannot element wise multiplication with " << tA->name() << " and " << tB->name() << ", their shape is different" << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate result tensor
	Tensor* t_out = new Tensor({ tA->row(), tA->col() }, false);
	sendToDevice(t_out);

	// Allocate to device
	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	int nBblocks;
	if (tA->row() >= tA->col())
		nBblocks = tA->row() % BLOCK_DIM == 0 ? tA->row() / BLOCK_DIM : tA->row() / BLOCK_DIM + 1;
	else
		nBblocks = tA->col() % BLOCK_DIM == 0 ? tA->col() / BLOCK_DIM : tA->col() / BLOCK_DIM + 1;

	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks(nBblocks, nBblocks);

	cuEltwiseMultiplication <<<blocks, threads >>> (t_out->devDataPtr(), tA->devDataPtr(), tB->devDataPtr(), tA->row(), tA->col());
	syncGpuStream();
}

// tOut = tA * tB (Haramard Product)
void Perceptor::matEltMult(Tensor* tOut, Tensor* tA, Tensor* tB) {
	checkDevice(tOut, tA);
	checkDevice(tA, tB);
	if (!tA->isSame(*tB)) {
		cout << "Cannot element wise multiplication with " << tA->name() << " and " << tB->name() << ", their shape is different" << endl;
		exit(EXIT_FAILURE);
	}

	// Allocate to device
	if (!tOut->haveDevicePtr())
		sendToDevice(tOut);
	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	int nBblocks;
	if (tA->row() >= tA->col())
		nBblocks = tA->row() % BLOCK_DIM == 0 ? tA->row() / BLOCK_DIM : tA->row() / BLOCK_DIM + 1;
	else
		nBblocks = tA->col() % BLOCK_DIM == 0 ? tA->col() / BLOCK_DIM : tA->col() / BLOCK_DIM + 1;

	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks(nBblocks, nBblocks);

	cuEltwiseMultiplication << <blocks, threads >> > (tOut->devDataPtr(), tA->devDataPtr(), tB->devDataPtr(), tA->row(), tA->col());
	syncGpuStream();
}

void Perceptor::matSwap(Tensor* tA, Tensor* tB, bool forceSwap) {
	checkDevice(tA, tB);
	if (!forceSwap && !tA->isSame(*tB)) {
		cout << tA->name() << " and " << tB->name() << " tensors shapes are different!" << endl;
		exit(EXIT_FAILURE);
	}
	if(forceSwap && tA->size() != tB->size()) {
		cout << tA->name() << " and " << tB->name() << " tensors size are different!" << endl;
		exit(EXIT_FAILURE);
	}

	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	CuBLAS_ERROR(
		cublasSswap(
			cuBlasHandle,
			tA->size(),
			tA->devDataPtr(),
			1,
			tB->devDataPtr(),
			1
		)
	);
	syncGpuStream();
}

void Perceptor::matCopy(Tensor* tB, Tensor* tA) {
	checkDevice(tB, tA);
	if (!tA->isSame(*tB)) {
		cout << tA->name() << " and " << tB->name() << " tensors shapes are different!" << endl;
		exit(EXIT_FAILURE);
	}

	if (!tA->haveDevicePtr())
		sendToDevice(tA);
	if (!tB->haveDevicePtr())
		sendToDevice(tB);

	CuBLAS_ERROR(
		cublasScopy(
			cuBlasHandle,
			tA->size(),
			tA->devDataPtr(),
			1,
			tB->devDataPtr(),
			1
		)
	);
	syncGpuStream();
}

int Perceptor::matMaxIndex(Tensor* tA) {
	checkDevice(tA);
	int result = 0;

	if (!tA->haveDevicePtr())
		sendToDevice(tA);

	CuBLAS_ERROR(
		cublasIsamax(
			cuBlasHandle,
			tA->size(),
			tA->devDataPtr(),
			1,
			&result
		)
	);
	syncGpuStream();
	
	return result - 1;
}

int Perceptor::matMinIndex(Tensor* tA) {
	checkDevice(tA);
	int result = 0;

	if (!tA->haveDevicePtr())
		sendToDevice(tA);

	CuBLAS_ERROR(
		cublasIsamin(
			cuBlasHandle,
			tA->size(),
			tA->devDataPtr(),
			1,
			&result
		)
	);
	syncGpuStream();

	return result - 1;
}

dtype Perceptor::matSum(Tensor* tA) {
	checkDevice(tA);
	dtype result = 0;

	if (!tA->haveDevicePtr())
		sendToDevice(tA);

	CuBLAS_ERROR(
		cublasSasum(
			cuBlasHandle,
			tA->size(),
			tA->devDataPtr(),
			1,
			&result
		)
	);
	syncGpuStream();

	return result;
}

__global__ void cuMatrixCopy(const dtype* src, dtype* dst, int src_row, int src_col, int dst_row, int dst_col) {
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	if (y < dst_row && x < dst_col && y < src_row && x < src_col)
		dst[y * dst_col + x] = src[y * src_col + x];
}

__global__ void iptransposeCoalesced(dtype* src_data, dtype* dummy_data, int src_row, int src_col, int dummy_row, int dummy_col)
{
	__shared__ float tile_s[BLOCK_DIM][BLOCK_DIM + 1];
	__shared__ float tile_d[BLOCK_DIM][BLOCK_DIM + 1];

	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
		int dx = blockIdx.y * BLOCK_DIM + threadIdx.x;
		int dy = blockIdx.x * BLOCK_DIM + threadIdx.y;
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if(y + j < src_row && x < src_col)
				tile_s[threadIdx.y + j][threadIdx.x] = src_data[(y + j) * src_col + x];
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (dy + j < src_row && dx < src_col)
				tile_d[threadIdx.y + j][threadIdx.x] = src_data[(dy + j) * src_col + dx];

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			dummy_data[(dy + j) * dummy_col + dx] = tile_s[threadIdx.x][threadIdx.y + j];
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			dummy_data[(y + j) * dummy_col + x] = tile_d[threadIdx.x][threadIdx.y + j];
	}

	else if (blockIdx.y == blockIdx.x) { // handle on-diagonal case
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (y + j < src_row && x < src_col)
				tile_s[threadIdx.y + j][threadIdx.x] = src_data[(y + j)*src_col + x];

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			dummy_data[(y + j) * dummy_col + x] = tile_s[threadIdx.x][threadIdx.y + j];
	}
}

__global__ void iptransposeCoalesced(dtype* src_data, int src_row, int src_col)
{
	__shared__ float tile_s[BLOCK_DIM][BLOCK_DIM + 1];
	__shared__ float tile_d[BLOCK_DIM][BLOCK_DIM + 1];

	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
		int dx = blockIdx.y * BLOCK_DIM + threadIdx.x;
		int dy = blockIdx.x * BLOCK_DIM + threadIdx.y;
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (y + j < src_row && x < src_col)
				tile_s[threadIdx.y + j][threadIdx.x] = src_data[(y + j) * src_col + x];
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (dy + j < src_row && dx < src_col)
				tile_d[threadIdx.y + j][threadIdx.x] = src_data[(dy + j) * src_col + dx];

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (dy + j < src_row && dx < src_col)
				src_data[(dy + j) * src_col + dx] = tile_s[threadIdx.x][threadIdx.y + j];
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (y + j < src_row && x < src_col)
				src_data[(y + j) * src_col + x] = tile_d[threadIdx.x][threadIdx.y + j];
	}

	else if (blockIdx.y == blockIdx.x) { // handle on-diagonal case
		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (y + j < src_row && x < src_col)
				tile_s[threadIdx.y + j][threadIdx.x] = src_data[(y + j)*src_col + x];

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j += VECTOR_SIZE)
			if (y + j < src_row && x < src_col)
				src_data[(y + j) * src_col + x] = tile_s[threadIdx.x][threadIdx.y + j];
	}
}

void Perceptor::matTranspose(Tensor* tA) {
	checkDevice(tA);
	if (tA->dimension() != 2) {
		cout << "Cannot transpose matrix. " << tA->name() << " is " << tA->dimension() << " dimension matrix." << endl;
		exit(EXIT_FAILURE);
	}

	if (!tA->haveDevicePtr())
		sendToDevice(tA);

	int nBblocks;
	if (tA->row() >= tA->col())
		nBblocks = tA->row() % BLOCK_DIM == 0 ? tA->row() / BLOCK_DIM : tA->row() / BLOCK_DIM + 1;
	else
		nBblocks = tA->col() % BLOCK_DIM == 0 ? tA->col() / BLOCK_DIM : tA->col() / BLOCK_DIM + 1;

	dim3 threads(BLOCK_DIM, VECTOR_SIZE);
	dim3 threads2(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks(nBblocks, nBblocks);

	if (tA->shape(0) == tA->shape(1)) {
		iptransposeCoalesced <<< blocks, threads >>> (tA->devDataPtr(), tA->col(), tA->row());
	}
	else {
		iptransposeCoalesced <<< blocks, threads >>>
			(tA->devDataPtr(), dummyTensor->devDataPtr(), tA->row(), tA->col(), MATRIX_DIM_LIMIT, MATRIX_DIM_LIMIT);
		syncGpuStream();

		cuMatrixCopy <<< blocks, threads2 >>>
			(dummyTensor->devDataPtr(), tA->devDataPtr(), MATRIX_DIM_LIMIT, MATRIX_DIM_LIMIT, tA->col(), tA->row());
	}

	syncGpuStream();
	tA->swapDimension(0, 1);
}


void Perceptor::sendToDevice(Tensor* t, bool sendData) {
	Tensor* devPtr = 0;
	t->setDevice(deviceId());
	cudaSetDevice(deviceId());
	CUDA_CHECK(cudaMalloc((void**)&devPtr, sizeof(Tensor)));
	CUDA_CHECK(cudaMemcpy(devPtr, t, sizeof(Tensor), cudaMemcpyHostToDevice));

	int* hostShape;
	CUDA_CHECK(cudaMalloc((void**)&hostShape, sizeof(int) * t->dimension()));
	CUDA_CHECK(cudaMemcpy(hostShape, t->mShape, sizeof(int) * t->dimension(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&(devPtr->mShape), &hostShape, sizeof(int*), cudaMemcpyHostToDevice));

	int* hostCumulatedDimension;
	CUDA_CHECK(cudaMalloc((void**)&hostCumulatedDimension, sizeof(int) * t->dimension()));
	CUDA_CHECK(cudaMemcpy(hostCumulatedDimension, t->cumulatedDimension, sizeof(int) * t->dimension(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&(devPtr->cumulatedDimension), &hostCumulatedDimension, sizeof(int*), cudaMemcpyHostToDevice));

	// Set device tensor as container
	CUDA_CHECK(cudaMemcpy(&devPtr->isContainer, new bool(true), sizeof(bool), cudaMemcpyHostToDevice));

	// Copy host data to device
	dtype* hostData;
	CUDA_CHECK(cudaMalloc((void**)&hostData, sizeof(dtype) * t->size()));
	if (sendData)
		CUDA_CHECK(cudaMemcpy(hostData, t->data, sizeof(dtype) * t->size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&(devPtr->data), &hostData, sizeof(dtype*), cudaMemcpyHostToDevice));
	t->devData = hostData;
	t->dev = devPtr;
	t->devShape = hostShape;
	t->devCumulatedDimension = hostCumulatedDimension;
	t->mHaveDevPtr = true;
	t->mHaveDevDataPtr = true;
}

void Perceptor::sendDataToDevice(Tensor* t) {
	if (t->deviceId() != deviceId()) {
		cout << t->name() << " and  perceptor device ID " << deviceId() << " are different" << endl;
		exit(EXIT_FAILURE);
	}
	if (t->haveDevicePtr()) {
		dtype* devPtr = 0;
		t->setDevice(deviceId());
		cudaSetDevice(deviceId());
		cudaFree(t->devData);
		CUDA_CHECK(cudaMalloc((void**)&devPtr, sizeof(dtype) * t->size()));
		CUDA_CHECK(cudaMemcpy(devPtr, t->data, sizeof(dtype) * t->size(), cudaMemcpyHostToDevice));
		t->devData = devPtr;
		t->mHaveDevDataPtr = true;
	}
	else {
		cout << t->name() << " is not allocated at device" << endl;
		exit(EXIT_FAILURE);
	}
}

void Perceptor::retrievDataFromDevice(Tensor* t, bool retreiveOnlyData) {
	if (t->haveDevicePtr() && t->haveDeviceDataPtr()) {
		cudaSetDevice(deviceId());
		CUDA_CHECK(cudaMemcpy(t->data, t->devData, t->size() * sizeof(dtype), cudaMemcpyDeviceToHost));
		if (!retreiveOnlyData) {
			Tensor temp;
			CUDA_CHECK(cudaMemcpy(&temp, t->dev, sizeof(Tensor), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(t->cumulatedDimension, temp.cumulatedDimension, t->dimension() * sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(t->mShape, temp.mShape, t->dimension() * sizeof(int), cudaMemcpyDeviceToHost));
			temp.setName("");
		}
	}
}