//#include <stdio.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

#include "SimulatorCUDA.cuh"

#include <math.h>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Common.cuh"

using namespace std;
using namespace cinder;

#define BUCKETSIZE 200
#define PROBESIZE 401
#define BUCKETLIMIT 1024
#define PROBETHRESHOLD 200
#define MAXRANGE 20000
#define s_size 700

//=========================================================================================================
__device__ int findGI(int column, int row) {
	return ((column << 8) | row);
}

__device__ int findRange(int value, int *range, int nRanges) {
	int L = 0;
	int R = nRanges - 1;
	while (L <= R) {
		int m = floorf((L + R) / 2);
		if (value > range[m]) L = m + 1;
		else if (value <= range[m]) {
			if (value <= range[m - 1]) {
				R = m - 1;
			}
			else {
				return m - 1;
			}
		}
	}

	return -1;

	//for (int i = 1; i < nRanges + 1; i++) {
	//	if (value <= range[i]) return i - 1;
	//}
}

__global__ void initializeProbe(int *workItem, int *probe, int probeSize, int *count, int *splitter) {
	int probeRange = MAXRANGE / (probeSize - 1);
	int workerID = threadIdx.x + (blockIdx.x*blockDim.x);
	int nWorker = blockDim.x * gridDim.x;

	//resetWorkItem
	for (int i = workerID; i < BUCKETLIMIT*BUCKETSIZE; i += nWorker) {
		workItem[i] = -1;
	}

	for (int i = workerID; i < PROBESIZE; i += nWorker) {
		probe[i] = probeRange*i;
	}
	__syncthreads();
	if (workerID == 0) probe[0] = -1;

	for (int i = workerID; i < BUCKETSIZE; i += nWorker) {
		count[i] = 0;
	}

	for (int i = workerID; i < BUCKETSIZE; i += nWorker) {
		splitter[i] = -1;
	}
}

__global__ void prepareProbe(int *workItem, int *probe, int probeSize, int *count, int *splitter) {
	int probeRange = MAXRANGE / (probeSize - 1);
	int workerID = threadIdx.x + (blockIdx.x*blockDim.x);
	int nWorker = blockDim.x * gridDim.x;

	//resetWorkItem
	for (int i = workerID; i < BUCKETLIMIT*BUCKETSIZE; i += nWorker) {
		workItem[i] = -1;
	}

	for (int i = workerID; i < BUCKETSIZE; i += nWorker) {
		count[i] = 0;
	}

	for (int i = workerID; i < BUCKETSIZE; i += nWorker) {
		splitter[i] = -1;
	}
}

__device__ int assignProbe(int *probe, int probeIndex, int probeCount, int lowBound, int highBound) {
	int range = highBound - lowBound;
	int distance = range / (probeCount + 1);
	int probeused = 0;

	if (distance == 0) {
		distance = 1;
	}

	for (int i = 0; i < probeCount; i++) {
		if (lowBound + (distance*(i + 1)) > highBound + 1) break;

		int index = i + probeIndex;
		if (index >= PROBESIZE - 1) break;
		probe[index] = lowBound + (distance*(i + 1));
		probeused++;
	}

	return probeused;
}

__global__ void tuneProbe2(int *probe, int probeSize, int* splitter) {
	//count non-hit
	__shared__ int count;
	__shared__ float probePerBucket;
	__shared__ int s_lowbound[BUCKETSIZE];
	__shared__ int s_highbound[BUCKETSIZE];
	if (threadIdx.x == 0) count = 0;
	__syncthreads();

	for (int i = threadIdx.x; i < BUCKETSIZE; i += blockDim.x) {
		s_lowbound[i] = splitter[i];
		s_highbound[i] = splitter[i];
		if (splitter[i] == -1) atomicAdd(&count, 1);
	}
	if (threadIdx.x == 0) {
		if (s_lowbound[0] == -1) {
			s_lowbound[0] = 0;
		}
		if (s_highbound[BUCKETSIZE] == -1) {
			s_highbound[BUCKETSIZE] = MAXRANGE;
		}
	}
	__syncthreads();

	//create lowbound prefix
	int partner = 0;
	unsigned int maxstep = (unsigned int)ceilf(log2f(BUCKETSIZE));
	for (unsigned int d = 0; d < maxstep; d++) {
		partner = threadIdx.x + (1 << d);
		if (partner < BUCKETSIZE) {
			if (s_lowbound[partner] == -1) s_lowbound[partner] = s_lowbound[threadIdx.x];
		}
		__syncthreads();
	}
	//create highbound prefix
	partner = 0;
	for (unsigned int d = 0; d < maxstep; d++) {
		partner = threadIdx.x - (1 << d);
		if (partner >= 0) {
			if (s_highbound[partner] == -1) s_highbound[partner] = s_highbound[threadIdx.x];
		}
		__syncthreads();
	}

	//calculate assigned probe per bucket
	if (threadIdx.x == 0) {
		probePerBucket = (PROBESIZE - 1) / count;
		int bucketCount = 0;
		int probeCount = 0;
		int pi = 1;

		int start = 0;
		for (int i = 0; i < BUCKETSIZE; i++) {
			if (splitter[i] == -1) {
				bucketCount++;
			}
			else {
				if (bucketCount == 0) continue;
				else {
					probeCount = floorf(bucketCount * probePerBucket);
					pi += assignProbe(probe, pi, probeCount, s_lowbound[i - 1], s_highbound[i - 1]);
					bucketCount = 0;
				}
			}
		}
		if (bucketCount != 0) {
			probeCount = floorf(bucketCount * probePerBucket);
			pi += assignProbe(probe, pi, probeCount, s_lowbound[BUCKETSIZE - 1], s_highbound[BUCKETSIZE - 1]);
			bucketCount = 0;
		}

		int remaining = PROBESIZE - pi;
		assignProbe(probe, pi, remaining, probe[pi - 1], MAXRANGE);
	}
	__syncthreads();
}

__global__ void tuneProbeTest(int *probe, int probeSize, int* splitter) {
	for (int i = threadIdx.x + 1; i < BUCKETSIZE; i += blockDim.x) {
		probe[i] = splitter[i - 1];
	}
	if(threadIdx.x == 0)probe[BUCKETSIZE] = MAXRANGE;
}

__global__ void checkHistogram(int *prefixsum, int *probe, int probeSize, bool *probecheck, int* splitter) {
	if (splitter[blockIdx.x] != -1) return;

	int d;
	int ideal = (blockIdx.x + 1) * 512;
	__shared__ int candidate[PROBESIZE];
	__shared__ int candidateindex[PROBESIZE];

	for (int i = threadIdx.x; i < PROBESIZE - 1; i += blockDim.x) {
		int currentTotal = prefixsum[i + 1];
		int accuracy = abs(currentTotal - ideal);

		candidate[i] = accuracy;
		candidateindex[i] = i + 1;
		__syncthreads();

		//do reduction to find accurate
		int partner = 0;
		int distance = 0;
		unsigned int sum = 0;
		unsigned int maxstep = (unsigned int)ceilf(log2f(PROBESIZE));
		for (unsigned int d = 0; d < maxstep; d++) {
			partner = threadIdx.x - (1 << d);
			distance = 2 << d;
			if (partner%distance == 0) {
				candidateindex[partner] = (candidate[threadIdx.x] < candidate[partner]) ? candidateindex[threadIdx.x] : candidateindex[partner];
				candidate[partner] = (candidate[threadIdx.x] < candidate[partner]) ? candidate[threadIdx.x] : candidate[partner];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			if (candidate[0] < PROBETHRESHOLD)splitter[blockIdx.x] = probe[candidateindex[0]];
			else {
				splitter[blockIdx.x] = -1;
				probecheck[0] = false;
			}
		}
	}

	//splitter[threadIdx.x + (blockIdx.x*blockDim.x)] = 999;
}

__global__ void resetHistogram(int *histogram, int probeSize) {
	int nWorker = gridDim.x * blockDim.x;
	int workerID = blockIdx.x*blockDim.x + threadIdx.x;
	for (int i = workerID; i < PROBESIZE; i += nWorker) {
		histogram[i] = 0;
	}
}

__global__ void fillBucket(Particle* particles, int nParticles, int *histogram, int *probe, int probeSize, bool *probecheck) {
	if (threadIdx.x + blockIdx.x == 0)probecheck[0] = true;
	int nWorker = gridDim.x * blockDim.x;
	int workerID = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int localHistogram[PROBESIZE];

	//zero out local
	for (int i = threadIdx.x; i < PROBESIZE; i += blockDim.x) {
		localHistogram[i] = 0;
	}
	__syncthreads();

	//fill local bucket
	for (int i = workerID; i < nParticles; i += nWorker) {
		int bucketID = findRange(particles[i].gi, probe, probeSize + 1);
		atomicAdd(&localHistogram[bucketID], 1);
	}
	__syncthreads();

	//fill global bucket
	for (int i = threadIdx.x; i < PROBESIZE; i += blockDim.x) {
		atomicAdd(&histogram[i], localHistogram[i]);
	}
	__syncthreads();
}

__global__ void createPrefixSum(int *histogram, int *prefixsum, int probeSize) {
	//copy histogram to shared
	int nWorker = gridDim.x * blockDim.x;
	int workerID = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int s_prefix[PROBESIZE + 1];
	for (int i = workerID; i < PROBESIZE; i += nWorker) {
		s_prefix[i] = histogram[i];
	}
	__syncthreads();

	//create prefixsum from histogram
	int partner = 0;
	unsigned int sum = 0;
	unsigned int maxstep = (unsigned int)ceilf(log2f(PROBESIZE));
	for (unsigned int d = 0; d < maxstep; d++) {
		partner = threadIdx.x - (1 << d);
		if (partner >= 0) {
			sum = s_prefix[threadIdx.x] + s_prefix[partner];
		}
		else {
			sum = s_prefix[threadIdx.x];
		}
		__syncthreads();
		s_prefix[threadIdx.x] = sum;
		__syncthreads();
	}

	//Shift elements to produce the same effect as exclusive scan
	unsigned int cpy_val = 0;
	cpy_val = s_prefix[threadIdx.x];
	__syncthreads();
	s_prefix[threadIdx.x + 1] = cpy_val;
	if (workerID == 0) s_prefix[0] = 0;
	__syncthreads();

	for (int i = workerID; i < PROBESIZE; i += nWorker) {
		atomicExch(&prefixsum[i], s_prefix[i]);
	}
}

__global__ void fillWorkItem2(Particle* particles, int* workItem, int *histogram, int *probe, int probeSize, int *count) {
	__shared__ int s_histogram[BUCKETSIZE];
	__shared__ int s_offset[BUCKETSIZE];
	int workerID = threadIdx.x + (blockDim.x*blockIdx.x);

	//zero out local histogram
	for (int i = threadIdx.x; i < BUCKETSIZE; i += blockDim.x) {
		s_histogram[i] = 0;
	}
	__syncthreads();

	//fill local histogram
	int bucketID = findRange(particles[workerID].gi, probe, BUCKETSIZE + 1);
	atomicAdd(&s_histogram[bucketID], 1);
	__syncthreads();

	//fill global bucket
	for (int i = threadIdx.x; i < PROBESIZE; i += blockDim.x) {
		atomicAdd(&histogram[i], s_histogram[i]);
	}

	//atomicAdd count to reserve space
	for (int i = threadIdx.x; i < BUCKETSIZE; i += blockDim.x) {
		s_offset[i] = atomicAdd(&count[i], s_histogram[i]);
	}
	__syncthreads();

	//localCount???
	int index = atomicAdd(&s_offset[bucketID], 1);
	index += bucketID*BUCKETLIMIT;
	workItem[index] = workerID;
	//__syncthreads();
}

__global__ void localBitonicSort(Particle* particles, Particle* particles2, int* workItem, int *prefixsum) {
	int workstart = blockIdx.x*BUCKETLIMIT;
	int type = 1;
	__shared__ int s_data[BUCKETLIMIT];
	__shared__ int s_index[BUCKETLIMIT];
	__shared__ bool done;

	int wi = -1;
	int partnerwi = -1;
	int loopCount = 0;

	//copy data to shared
	for (int i = threadIdx.x; i < BUCKETLIMIT; i += blockDim.x) {
		int index = i + workstart;
		if (workItem[index] != -1) {
			s_data[i] = particles[workItem[index]].gi;
			s_index[i] = workItem[index];
		}
		else {
			s_data[i] = 9999999;
			s_index[i] = -1;
		}
	}
	if (threadIdx.x == 0) done = false;
	__syncthreads();

	//bitonic-sort loop per stage
	int maxstage = log2f(blockDim.x);
	for (int i = 0; i < maxstage + 1; i++) {

		//sort until distance = 0
		int initialDistance = 1 << (i);
		bool ascending = (threadIdx.x / initialDistance) % 2 == 0;
		for (int d = initialDistance; d > 0; d /= 2) {
			//assign work per thread
			int leap = d * 2;
			int nleap = threadIdx.x / d;
			int offset = threadIdx.x%d;

			wi = (leap*nleap) + offset;
			partnerwi = wi + d;
			if ((wi > BUCKETLIMIT) || (partnerwi > BUCKETLIMIT)) continue;

			//compare and swap
			if (ascending) {
				if (s_data[wi] > s_data[partnerwi]) {
					s_data[wi] = s_data[wi] + s_data[partnerwi];
					s_data[partnerwi] = s_data[wi] - s_data[partnerwi];
					s_data[wi] = s_data[wi] - s_data[partnerwi];

					s_index[wi] = s_index[wi] + s_index[partnerwi];
					s_index[partnerwi] = s_index[wi] - s_index[partnerwi];
					s_index[wi] = s_index[wi] - s_index[partnerwi];
				}
			}
			else {
				if (s_data[wi] < s_data[partnerwi]) {
					s_data[wi] = s_data[wi] + s_data[partnerwi];
					s_data[partnerwi] = s_data[wi] - s_data[partnerwi];
					s_data[wi] = s_data[wi] - s_data[partnerwi];

					s_index[wi] = s_index[wi] + s_index[partnerwi];
					s_index[partnerwi] = s_index[wi] - s_index[partnerwi];
					s_index[wi] = s_index[wi] - s_index[partnerwi];
				}
			}

			__syncthreads();
		}

		__syncthreads();
	}

	//output data from shared to global
	for (int i = threadIdx.x; i < BUCKETLIMIT; i += blockDim.x) {
		//int index = i + workstart;
		//workItem[index] = s_index[i];
		if (s_index[i] != -1) {
			//particles[s_index[i]].writei = prefixsum[blockIdx.x] + i;
			Particle &p = particles[s_index[i]];
			Particle &p2 = particles2[prefixsum[blockIdx.x] + i];

			p2.x = p.x;
			p2.y = p.y;
			p2.u = p.u;
			p2.v = p.v;
			p2.gu = p.gu;
			p2.gv = p.gv;
			p2.T00 = p.T00;
			p2.T01 = p.T01;
			p2.T11 = p.T11;
			p2.gi = p.gi;

			p2.px[0] = p.px[0];
			p2.py[0] = p.py[0];
			p2.gx[0] = p.gx[0];
			p2.gy[0] = p.gy[0];

			p2.px[1] = p.px[1];
			p2.py[1] = p.py[1];
			p2.gx[1] = p.gx[1];
			p2.gy[1] = p.gy[1];

			p2.px[2] = p.px[2];
			p2.py[2] = p.py[2];
			p2.gx[2] = p.gx[2];
			p2.gy[2] = p.gy[2];

			p2.color = p.color;
			p2.pos = p.pos;
			p2.trail = p.trail;
			p2.new_color = p.new_color;
			p2.cx = p.cx;
			p2.cy = p.cy;
			p2.writei = p.writei;
		}

	}
	__syncthreads();
}

string histogramSort(Particle* particles, Particle* particles2, int nParticles, int *histogram, int *probe, int probeSize, bool *probecheck, int *workitem, int *count, int *prefixsum, int *splitter, int order) {
	int change = 3200;
	bool end = false;
	prepareProbe << <BUCKETSIZE, 512 >> > (workitem, probe, probeSize, count, splitter); cudaDeviceSynchronize();

	int i = 0;
	for (i = 0; i < 10; i++) {
		//new probe
		if (i > 0) {
			tuneProbe2 << <1, PROBESIZE >> > (probe, probeSize, splitter);
		}

		resetHistogram << <1, PROBESIZE >> > (histogram, probeSize); cudaDeviceSynchronize(); cudaDeviceSynchronize();

		//kernel: fill bucket
		fillBucket << <BUCKETSIZE, 512 >> > (particles, nParticles, histogram, probe, probeSize, probecheck); cudaDeviceSynchronize();

		//createprefixsum
		createPrefixSum << <1, PROBESIZE >> > (histogram, prefixsum, probeSize); cudaDeviceSynchronize();

		checkHistogram << <BUCKETSIZE, 512 >> > (prefixsum, probe, probeSize, probecheck, splitter); cudaDeviceSynchronize();

		cudaMemcpy(&end, probecheck, sizeof(bool), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
		if (end) break;
	}

	if (!end) {
		return "fail";
	}

	if (i > 0) {
		//kernel: fill each block working item
		tuneProbeTest << <1, PROBESIZE >> > (probe, probeSize, splitter); cudaDeviceSynchronize();
		resetHistogram << <1, PROBESIZE >> > (histogram, probeSize); cudaDeviceSynchronize(); cudaDeviceSynchronize();
		fillWorkItem2 << <BUCKETSIZE, 512 >> > (particles, workitem, histogram, probe, BUCKETSIZE, count); cudaDeviceSynchronize();
		createPrefixSum << <1, PROBESIZE >> >  (histogram, prefixsum, probeSize); cudaDeviceSynchronize();

		//kernel: each block assign moving index
		localBitonicSort << <BUCKETSIZE, 512 >> > (particles, particles2, workitem, prefixsum); cudaDeviceSynchronize();
	}
	return "wa";

	//int hist[PROBESIZE];
	//int pro[PROBESIZE];
	//int pre[PROBESIZE];
	//int split[BUCKETSIZE];
	//int work[BUCKETSIZE*BUCKETLIMIT];
	//std::stringstream ss;
	//////print-------------------------------------------------------
	//cudaMemcpy(hist, histogram, sizeof(int)* PROBESIZE, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
	//cudaMemcpy(pro, probe, sizeof(int)*PROBESIZE, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
	//cudaMemcpy(pre, prefixsum, sizeof(int)*PROBESIZE, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
	//cudaMemcpy(split, splitter, sizeof(int)*BUCKETSIZE, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
	//cudaMemcpy(work, workitem, sizeof(int)*BUCKETSIZE*BUCKETLIMIT, cudaMemcpyDeviceToHost); cudaDeviceSynchronize();

	//for (int i = 0; i < PROBESIZE-1; i++) {
	//	ss << i << ") \t" << hist[i] << " \t | " << pro[i] << "-" << pro[i + 1] << "\t \t | " << pre[i + 1] << endl;
	//	//ss << i << ") \t" << hist[i] << " \t | " << pro[i*2] << "-" << pro[(i*2) + 1] << "\t \t | " << pre[i + 1] << endl;
	//}
	//ss << endl;

	//ss << "splitter=================" << endl;
	//for (int i = 0; i < BUCKETSIZE; i++) {
	//	ss << i << ") \t" << split[i];
	//	ss << endl;
	//}

	//ss << "workitems=================" << endl;
	//int cnt = 0;
	//for (int i = 0; i < BUCKETLIMIT*200; i++) {
	//	if (i%BUCKETLIMIT == 0) {
	//		ss << endl << cnt << "-------------" << endl;
	//		cnt = 0;
	//	}
	//	//if (work[i] == 9999999) continue;
	//	cnt++;
	//	ss << work[i] << ",";
	//	//if (i%BUCKETLIMIT == 10) i += BUCKETLIMIT - 11;
	//}
	//ss << endl;
	//-------------------------------------------------------------
	//return ss.str();
}
//=========================================================================================================

__device__ float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v)
{
	float dx = x00 - x01;
	float dy = y00 - y10;
	float a = p01 - p00;
	float b = p11 - p10 - a;
	float c = p10 - p00;
	float d = y11 - y01;
	return ((((d - 2 * b - dy) * u - 2 * a + y00 + y01) * v +
		((3 * b + 2 * dy - d) * u + 3 * a - 2 * y00 - y01)) * v +
		((((2 * c - x00 - x10) * u + (3 * b + 2 * dx + x10 - x11)) * u - b - dy - dx) * u + y00)) * v +
		(((x11 - 2 * (p11 - p01 + c) + x10 + x00 + x01) * u +
		(3 * c - 2 * x00 - x10)) * u +
			x00) * u + p00;
}

SimulatorCUDA::SimulatorCUDA() :scale(1.0f) {
	//default
	materials[0].materialIndex = 0;
	materials[0].mass = 1.0f;
	materials[0].viscosity = 0.04f;

	materials[1].materialIndex = 1;
	materials[1].mass = 1.0f;
	materials[1].restDensity = 10.0f;
	materials[1].viscosity = 1.0f;
	materials[1].bulkViscosity = 3.0f;
	materials[1].stiffness = 1.0f;
	materials[1].meltRate = 1.0f;
	materials[1].kElastic = 1.0f;



	materials[2].materialIndex = 2;
	materials[2].mass = 0.7f;
	materials[2].viscosity = 0.03f;

	materials[3].materialIndex = 3;

	const int size = numMaterials * sizeof(Material);
	cudaMalloc((void**)&d_materials, size);
	cudaMemcpy(d_materials, materials, size, cudaMemcpyHostToDevice);
	app::console() << "Materials intialized" << endl;
}

void SimulatorCUDA::initializeHelper() {
	probeSize = PROBESIZE;
	cudaMalloc(&d_probe, sizeof(int)* PROBESIZE);
	cudaMalloc(&d_splitter, sizeof(int)* (BUCKETSIZE + 1));
	cudaMalloc(&d_histogram, sizeof(int)*PROBESIZE);
	cudaMalloc(&d_prefixsum, sizeof(int)*PROBESIZE);
	cudaMalloc(&d_workitem, sizeof(int)*BUCKETSIZE*BUCKETLIMIT);
	cudaMalloc(&d_count, sizeof(int)*BUCKETSIZE);
	cudaMalloc(&d_probecheck, sizeof(bool));

	initializeProbe << <BUCKETSIZE, 512 >> > (d_workitem, d_probe, PROBESIZE, d_count, d_splitter); cudaDeviceSynchronize();
}

void SimulatorCUDA::initializeGrid(int sizeX, int sizeY) {
	gSizeX = sizeX;
	gSizeY = sizeY;
	gSizeY_3 = sizeY - 3;
	int gSize = gSizeX*gSizeY;
	int sizeOfGrid = gSize * sizeof(Node);
	int sizeOfGridAtt = gSize * sizeof(float) * numMaterials * 2;

	grid = new Node[gSize];
	cudaMalloc(&Pindex, sizeof(int)*MAXPARTICLE*NPATCH);
	cudaMalloc(&startIndex, sizeof(int)*NPATCH*NPATCH);
	gpuErrchk(cudaMalloc((void**)&d_grid, sizeOfGrid));
	gpuErrchk(cudaMalloc((void**)&d_gridAtt, sizeOfGridAtt));

	for (int i = 0; i < gSize; i++)
	{
		grid[i].initAttArrays(d_gridAtt + i * numMaterials * 2);
	}

	gpuErrchk(cudaMemcpy(d_grid, grid, sizeOfGrid, cudaMemcpyHostToDevice));
	//size = gSizeX*gSizeY * sizeof(Node*);
	//gpuErrchk(cudaMalloc((void**)&d_active, size));
	//gpuErrchk(cudaMalloc((void**)&d_nActive, sizeof(int)));

	app::console() << "Grid initialized" << endl;
}

void SimulatorCUDA::addParticles(int n) {
	app::console() << "Adding particles..." << endl;
	int pn = (int)sqrt((float)(n / 3));
	int dn = n - (pn*pn * 3);
	float bw = 40.0f;
	float bh = 16.0f;
	float mx = bw / (float)pn;
	float my = bh / (float)pn;
	float offset = 10.0f;

	this->particleCount = n;
	const int sizeOfParticles = particleCount * sizeof(Particle);
	const int sizeOfParticlesAtt = particleCount * sizeof(float) * 12;

	gpuErrchk(cudaMalloc((void**)&d_particles, sizeOfParticles));
	gpuErrchk(cudaMalloc((void**)&d_particles2, sizeOfParticles));
	gpuErrchk(cudaMalloc((void**)&d_particleAtt, sizeOfParticlesAtt));
	gpuErrchk(cudaMalloc((void**)&d_particleAtt2, sizeOfParticlesAtt));
	gpuErrchk(cudaMemset(d_particleAtt, 0.0f, sizeOfParticlesAtt));
	gpuErrchk(cudaMemset(d_particleAtt2, 0.0f, sizeOfParticlesAtt));
	workarray = d_particles;
	workarray2 = d_particles2;

	int i, j;
	int pCount = 0;
	// Material 1
	for (i = 0; i < pn; i++) {
		for (j = 0; j < pn; j++) {
			float px = i*mx + offset;
			float py = j*my + offset;
			Particle p(&d_materials[0], px, py, ColorA(1, 0.5, 0.5, 1));
			//p.initAttArrays(d_particleAtt + particles.size() * 12);
			p.initializeWeights(gSizeY);
			p.writei = pCount;
			particles.push_back(p);

			//int position = py + (px*gSizeY);
			//int gi = grid[position].particleCount;
			//if (gi < 10) {
			//	grid[position].particleIndex[gi] = pCount;
			//	grid[position].particleCount++;
			//}
			pCount++;
		}
	}

	// Material 2
	for (i = 0; i < pn; i++) {
		for (j = 0; j < pn; j++) {
			float px = i*mx + (this->gSizeX - bw) / 2;
			float py = j*my + offset;
			Particle p(&d_materials[0], px, py, ColorA(0.5, 1, 0.5, 1));
			//p.initAttArrays(d_particleAtt + particles.size() * 12);
			p.initializeWeights(gSizeY);
			p.writei = pCount;
			particles.push_back(p);

			//int position = py + (px*gSizeY);
			//int gi = grid[position].particleCount;
			//if (gi < 10) {
			//	grid[position].particleIndex[gi] = pCount;
			//	grid[position].particleCount++;
			//}
			pCount++;
		}
	}

	// Material 2
	for (i = 0; i < pn; i++) {
		for (j = 0; j < pn; j++) {
			float px = i*mx + this->gSizeX - bw - offset;
			float py = j*my + offset;
			Particle p(&d_materials[0], px, py, ColorA(0.5, 0.5, 1, 1));
			//p.initAttArrays(d_particleAtt + particles.size() * 12);
			p.initializeWeights(gSizeY);
			p.writei = pCount;
			particles.push_back(p);

			//int position = py + (px*gSizeY);
			//int gi = grid[position].particleCount;
			//if (gi < 10) {
			//	grid[position].particleIndex[gi] = pCount;
			//	grid[position].particleCount++;
			//}
			pCount++;
		}
	}
	for (int x = 0; x < dn; x++)
	{
		float px = i*mx + this->gSizeX - bw - offset;
		float py = (j++)*my + offset;
		Particle p(&d_materials[0], px, py, ColorA(0.5, 0.5, 1, 1));
		//p.initAttArrays(d_particleAtt + particles.size() * 12);
		p.initializeWeights(gSizeY);
		p.writei = pCount;
		particles.push_back(p);

		//int position = py + (px*gSizeY);
		//int gi = grid[position].particleCount;
		//if (gi < 10) {
		//	grid[position].particleIndex[gi] = pCount;
		//	grid[position].particleCount++;
		//}
		pCount++;
	}

	app::console() << "Copying particles to device..." << endl;

	gpuErrchk(cudaMemcpy(d_particles, particles.data(), sizeOfParticles, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particles2, particles.data(), sizeOfParticles, cudaMemcpyHostToDevice));

	app::console() << sizeof(Particle) << ", " << sizeOfParticles << endl;
	app::console() << "Particles added" << endl;
}

__device__ unsigned int hashCustom(unsigned int x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

__global__ void PrepareGrid4(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3, int order)
{
	int nWorker = blockDim.x * gridDim.x;
	int workerId = threadIdx.x + (blockIdx.x*blockDim.x);
	for (int loop = workerId; loop < nParticles; loop += nWorker) {
		int pi = loop;
		if (pi < nParticles) {
			Particle &p = particles[pi];
			Material& mat = *p.mat;

			float m = p.mat->mass;
			float mu = m * p.u;
			float mv = m * p.v;
			int mi = p.mat->materialIndex;
			float *px = p.px;
			float *gx = p.gx;
			float *py = p.py;
			float *gy = p.gy;
			Node* n = &grid[p.gi];
			int gi = p.gi;

			__syncthreads();

			for (int i = 0; i < 3; i++, n += gSizeY_3, gi += gSizeY_3) {
				float pxi = px[i];
				float gxi = gx[i];
				for (int j = 0; j < 3; j++, n++, gi++) {
					float pyj = py[j];
					float gyj = gy[j];
					float phi = pxi * pyj;

					atomicAdd(&(grid[gi].mass), phi * m);
					atomicAdd(&(grid[gi].particleDensity), phi);
					atomicAdd(&(grid[gi].u), phi * mu);
					atomicAdd(&(grid[gi].v), phi * mv);
					//atomicAdd(&(grid[gi].cgx[mi]), gxi * pyj);
					//atomicAdd(&(grid[gi].cgy[mi]), pxi * gyj);

					grid[gi].active = true;
				}
			}
		}
	}
}

__global__ void PrepareGrid4Sort(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3, int order)
{
	__shared__ float localGridMass[s_size];
	__shared__ float localGridDensity[s_size];
	__shared__ float localGridU[s_size];
	__shared__ float localGridV[s_size];
	__shared__ int localStart;
	__shared__ int localEnd;

	int nWorker = blockDim.x * gridDim.x;
	int workerId = threadIdx.x + (blockIdx.x*blockDim.x);
	int pi = workerId;
	if (pi >= nParticles) return;
	Particle &p = particles[pi];
	Material& mat = *p.mat;

	float m = p.mat->mass;
	float mu = m * p.u;
	float mv = m * p.v;
	int mi = p.mat->materialIndex;
	float *px = p.px;
	float *gx = p.gx;
	float *py = p.py;
	float *gy = p.gy;
	Node* n = &grid[p.gi];

	if (threadIdx.x == 0) {
		localStart = p.gi;
	}
	if (threadIdx.x == blockDim.x - 1) {
		localEnd = p.gi;
		localEnd += (gSizeY_3 * 2);
		localEnd += (10);
	}
	__syncthreads();

	while(localStart < localEnd) {
		int gi = p.gi;
		for (int i = threadIdx.x; i < s_size; i += blockDim.x) {
			localGridMass[i] = 0;
			localGridDensity[i] = 0;
			localGridV[i] = 0;
			localGridU[i] = 0;
			//localGridGi[i] = -999;
		}
		__syncthreads();
		for (int i = 0; i < 3; i++, n += gSizeY_3, gi += gSizeY_3) {
			for (int j = 0; j < 3; j++, n++, gi++) {
				int index = gi - localStart;
				if ((index >= s_size) || (index < 0))continue;

				float pxi = px[i];
				float gxi = gx[i];
				float pyj = py[j];
				float gyj = gy[j];
				float phi = pxi * pyj;

				atomicAdd(&(localGridMass[index]), phi*m);
				atomicAdd(&(localGridDensity[index]), phi);
				atomicAdd(&(localGridU[index]), phi * mu);
				atomicAdd(&(localGridV[index]), phi * mv);

				grid[gi].active = true;
			}
		}
	
		__syncthreads();
		for (int i = threadIdx.x; i < s_size; i += blockDim.x) {
			int index = i+localStart;
			if ((index >= gSizeX*gSizeY) || (index < 0)) continue;

			atomicAdd(&grid[index].mass, localGridMass[i]);
			atomicAdd(&grid[index].particleDensity, localGridDensity[i]);
			atomicAdd(&grid[index].u, localGridU[i]);
			atomicAdd(&grid[index].v, localGridV[i]);
		}
		__syncthreads();
		if (threadIdx.x == 0) localStart += s_size;
		__syncthreads();
	}
}

__global__ void PrepareGridTest(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3)
{
	int nWorker = blockDim.x * gridDim.x;
	int workerId = threadIdx.x + (blockIdx.x*blockDim.x);
	for (int loop = workerId; loop < nParticles; loop += nWorker) {
		int pi = loop;
		if (pi < nParticles) {
			Particle &p = particles[pi];
			Material& mat = *p.mat;

			float m = p.mat->mass;
			float mu = m * p.u;
			float mv = m * p.v;
			int mi = p.mat->materialIndex;
			float *px = p.px;
			float *gx = p.gx;
			float *py = p.py;
			float *gy = p.gy;
			Node* n = &grid[p.gi];
			int gi = p.gi;

			__syncthreads();

			for (int i = 0; i < 3; i++, n += gSizeY_3, gi += gSizeY_3) {
				float pxi = px[i];
				float gxi = gx[i];
				for (int j = 0; j < 3; j++, n++, gi++) {
					float pyj = py[j];
					float gyj = gy[j];
					float phi = pxi * pyj;

					atomicAdd(&(grid[gi].mass), 1);

					grid[gi].active = true;
				}
			}
		}
	}
}

__global__ void ApplyNewParticleAttribute(Particle* particles, Particle* particles2, int nParticles) {
	int pi = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (pi < nParticles) {
		Particle &p = particles[pi];
		Particle &p2 = particles2[pi];

		p.x = p2.x;
		p.y = p2.y;
		p.u = p2.u;
		p.v = p2.v;
		p.gu = p2.gu;
		p.gv = p2.gv;
		p.T00 = p2.T00;
		p.T01 = p2.T01;
		p.T11 = p2.T11;
		p.gi = p2.gi;
		p.color = p2.color;

		p.px[0] = p2.px[0];
		p.py[0] = p2.py[0];
		p.gx[0] = p2.gx[0];
		p.gy[0] = p2.gy[0];

		p.px[1] = p2.px[1];
		p.py[1] = p2.py[1];
		p.gx[1] = p2.gx[1];
		p.gy[1] = p2.gy[1];

		p.px[2] = p2.px[2];
		p.py[2] = p2.py[2];
		p.gx[2] = p2.gx[2];
		p.gy[2] = p2.gy[2];
	}
}

__global__ void PrepareGridLite(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3, int order)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi < nParticles) {
		Particle &p = particles[pi];

		Material& mat = *p.mat;

		float gu = 0, gv = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
		Node* n = &grid[p.gi];
		float* ppx = p.px;
		float* ppy = p.py;
		float* pgx = p.gx;
		float* pgy = p.gy;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				gu += phi * n->u2;
				gv += phi * n->v2;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u2 * gx;
				dudy += n->u2 * gy;
				dvdx += n->v2 * gx;
				dvdy += n->v2 * gy;
			}
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);

		p.T00 += .5f * (-wT0 + (D00 - trace) - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + (D11 - trace) - mat.meltRate * p.T11);

		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		p.x += gu;
		p.y += gv;
		p.gu = gu;
		p.gv = gv;
		p.u += mat.smoothing*(gu - p.u);
		p.v += mat.smoothing*(gv - p.v);

		// Hard boundary correction (Random numbers keep it from clustering)
		curandState_t cstate;
		curand_init(0, 0, 0, &cstate);

		if (p.x < 1) {
			p.x = 1 + .01*curand_uniform(&cstate);
			//p.x = 1;
		}
		else if (p.x > gSizeX - 2) {
			p.x = gSizeX - 2 - .01*curand_uniform(&cstate);
			//p.x = gSizeX - 2;
		}
		if (p.y < 1) {
			p.y = 1 + .01*curand_uniform(&cstate);
			//p.y = 1;
		}
		else if (p.y > gSizeY - 2) {
			p.y = gSizeY - 2 - .01*curand_uniform(&cstate);
			//p.y = gSizeY - 2;
		}

		// Update grid cell index and kernel weights
		int cx = p.cx = (int)(p.x - .5f);
		int cy = p.cy = (int)(p.y - .5f);
		p.gi = cx * gSizeY + cy;

		int gi = grid[p.gi].particleCount;
		if (gi < 10) {
			grid[p.gi].particleIndex[gi] = pi;
			grid[p.gi].particleCount++;
		}

		float x = cx - p.x;
		float y = cy - p.y;

		p.px[0] = .5f * x * x + 1.5f * x + 1.125f;
		p.gx[0] = x + 1.5f;
		x++;
		p.px[1] = -x * x + .75f;
		p.gx[1] = -2 * x;
		x++;
		p.px[2] = .5f * x * x - 1.5f * x + 1.125f;
		p.gx[2] = x - 1.5f;

		p.py[0] = .5f * y * y + 1.5f * y + 1.125f;
		p.gy[0] = y + 1.5f;
		y++;
		p.py[1] = -y * y + .75f;
		p.gy[1] = -2 * y;
		y++;
		p.py[2] = .5f * y * y - 1.5f * y + 1.125f;
		p.gy[2] = y - 1.5f;
	}
}

__global__ void PrepareGridLiteSort(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi < nParticles) {
		Particle &p = particles[pi];
		Particle &p2 = particles[p.writei];
		Material& mat = *p.mat;

		float gu = 0, gv = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
		Node* n = &grid[p.gi];
		float* ppx = p.px;
		float* ppy = p.py;
		float* pgx = p.gx;
		float* pgy = p.gy;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				gu += phi * n->u2;
				gv += phi * n->v2;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u2 * gx;
				dudy += n->u2 * gy;
				dvdx += n->v2 * gx;
				dvdy += n->v2 * gy;
			}
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);

		p2.new_T00 = p.T00 + .5f * (-wT0 + (D00 - trace) - mat.meltRate * p.T00);
		p2.new_T01 = p.T01 + .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p2.new_T11 = p.T11 + .5f * (wT0 + (D11 - trace) - mat.meltRate * p.T11);
		//p.T00 += .5f * (-wT0 + (D00 - trace) - mat.meltRate * p.T00);
		//p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		//p.T11 += .5f * (wT0 + (D11 - trace) - mat.meltRate * p.T11);

		float norm = p2.new_T00 * p2.new_T00 + 2 * p2.new_T01 * p2.new_T01 + p2.new_T11 * p2.new_T11;
		//float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p2.new_T00 = p2.new_T01 = p2.new_T11 = 0;
			//p.T00 = p.T01 = p.T11 = 0;
		}

		p2.new_x = p.x + gu;
		p2.new_y = p.y + gv;
		p2.new_gu = gu;
		p2.new_gv = gv;
		p2.new_u += mat.smoothing*(gu - p2.new_u);
		p2.new_v += mat.smoothing*(gv - p2.new_v);

		//p.x += gu;
		//p.y += gv;
		//p.gu = gu;
		//p.gv = gv;
		//p.u += mat.smoothing*(gu - p.u);
		//p.v += mat.smoothing*(gv - p.v);

		// Hard boundary correction (Random numbers keep it from clustering)
		curandState_t cstate;
		curand_init(0, 0, 0, &cstate);


		if (p2.new_x < 1) {
			p2.new_x = 1 + .01*curand_uniform(&cstate);
			//p.x = 1;
		}
		else if (p2.new_x > gSizeX - 2) {
			p2.new_x = gSizeX - 2 - .01*curand_uniform(&cstate);
			//p.x = gSizeX - 2;
		}
		if (p2.new_y < 1) {
			p2.new_y = 1 + .01*curand_uniform(&cstate);
			//p.y = 1;
		}
		else if (p2.new_y > gSizeY - 2) {
			p2.new_y = gSizeY - 2 - .01*curand_uniform(&cstate);
			//p.y = gSizeY - 2;
		}

		//if (p.x < 1) {
		//	p.x = 1 + .01*curand_uniform(&cstate);
		//	//p.x = 1;
		//}
		//else if (p.x > gSizeX - 2) {
		//	p.x = gSizeX - 2 - .01*curand_uniform(&cstate);
		//	//p.x = gSizeX - 2;
		//}
		//if (p.y < 1) {
		//	p.y = 1 + .01*curand_uniform(&cstate);
		//	//p.y = 1;
		//}
		//else if (p.y > gSizeY - 2) {
		//	p.y = gSizeY - 2 - .01*curand_uniform(&cstate);
		//	//p.y = gSizeY - 2;
		//}

		// Update grid cell index and kernel weights
		int cx = p2.cx = (int)(p2.new_x - .5f);
		int cy = p2.cy = (int)(p2.new_y - .5f);
		p2.new_gi = cx * gSizeY + cy;

		int gi = grid[p2.new_gi].particleCount;
		if (gi < 10) {
			grid[p2.new_gi].particleIndex[gi] = p.writei;
			grid[p2.new_gi].particleCount++;
		}

		float x = cx - p2.new_x;
		float y = cy - p2.new_y;

		// Quadratic interpolation kernel weights - Not meant to be changed
		p2.new_px[0] = .5f * x * x + 1.5f * x + 1.125f;
		p2.new_gx[0] = x + 1.5f;
		x++;
		p2.new_px[1] = -x * x + .75f;
		p2.new_gx[1] = -2 * x;
		x++;
		p2.new_px[2] = .5f * x * x - 1.5f * x + 1.125f;
		p2.new_gx[2] = x - 1.5f;

		p2.new_py[0] = .5f * y * y + 1.5f * y + 1.125f;
		p2.new_gy[0] = y + 1.5f;
		y++;
		p2.new_py[1] = -y * y + .75f;
		p2.new_gy[1] = -2 * y;
		y++;
		p2.new_py[2] = .5f * y * y - 1.5f * y + 1.125f;
		p2.new_gy[2] = y - 1.5f;

		//int cx = p.cx = (int)(p.x - .5f);
		//int cy = p.cy = (int)(p.y - .5f);
		//p.gi = cx * gSizeY + cy;

		//int gi = grid[p.gi].particleCount;
		//if (gi < 10) {
		//	grid[p.gi].particleIndex[gi] = pi;
		//	grid[p.gi].particleCount++;
		//}

		//float x = cx - p.x;
		//float y = cy - p.y;

		//p.px[0] = .5f * x * x + 1.5f * x + 1.125f;
		//p.gx[0] = x + 1.5f;
		//x++;
		//p.px[1] = -x * x + .75f;
		//p.gx[1] = -2 * x;
		//x++;
		//p.px[2] = .5f * x * x - 1.5f * x + 1.125f;
		//p.gx[2] = x - 1.5f;

		//p.py[0] = .5f * y * y + 1.5f * y + 1.125f;
		//p.gy[0] = y + 1.5f;
		//y++;
		//p.py[1] = -y * y + .75f;
		//p.gy[1] = -2 * y;
		//y++;
		//p.py[2] = .5f * y * y - 1.5f * y + 1.125f;
		//p.gy[2] = y - 1.5f;
	}
}
//===================================================================================
__global__ void FinalizeGrid(Node* grid, int  gSizeXY)
{
	Node* gi = grid;
	//int gSizeXY = gSizeX * gSizeY;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < gSizeXY) {
		Node* n = &gi[i];
		n->particleCount = 0;
		if (n->active && n->mass > 0) {
			//atomicAdd(nActive, 1);
			//active[*nActive] = n;
			//n->active = false;
			n->ax = n->ay = 0;
			n->gx = 0;
			n->gy = 0;
			n->u /= n->mass;
			n->v /= n->mass;
			for (int j = 0; j < numMaterials; j++) {
				n->gx += n->cgx[j];
				n->gy += n->cgy[j];
			}
			for (int j = 0; j < numMaterials; j++) {
				n->cgx[j] -= n->gx - n->cgx[j];
				n->cgy[j] -= n->gy - n->cgy[j];
			}
		}
	}
}

__global__ void CalculateForces(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3, float scale, int order)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi < nParticles) {
		Particle& p = particles[pi];
		Material& mat = *p.mat;

		float fx = 0, fy = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0, sx = 0, sy = 0;
		Node* n = &grid[p.gi];
		float *ppx = p.px;
		float *pgx = p.gx;
		float *ppy = p.py;
		float *pgy = p.gy;

		int materialId = mat.materialIndex;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u * gx;
				dudy += n->u * gy;
				dvdx += n->v * gx;
				dvdy += n->v * gy;

				// Surface tension
				sx += phi * n->cgx[materialId];
				sy += phi * n->cgy[materialId];
			}
		}

		int cx = (int)p.x;
		int cy = (int)p.y;
		int gi = cx * gSizeY + cy;

		Node& n1 = grid[gi];
		Node& n2 = grid[gi + 1];
		Node& n3 = grid[gi + gSizeY];
		Node& n4 = grid[gi + gSizeY + 1];
		float density = uscip(n1.particleDensity, n1.gx, n1.gy,
			n2.particleDensity, n2.gx, n2.gy,
			n3.particleDensity, n3.gx, n3.gy,
			n4.particleDensity, n4.gx, n4.gy,
			p.x - cx, p.y - cy);

		float pressure = mat.stiffness / mat.restDensity * (density - mat.restDensity);
		if (pressure > 2) {
			pressure = 2;
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		D00 -= trace;
		D11 -= trace;
		p.T00 += .5f * (-wT0 + D00 - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + D11 - mat.meltRate * p.T11);

		// Stress tensor fracture
		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		float T00 = mat.mass * (mat.kElastic * p.T00 + mat.viscosity * D00 + pressure + trace * mat.bulkViscosity);
		float T01 = mat.mass * (mat.kElastic * p.T01 + mat.viscosity * D01);
		float T11 = mat.mass * (mat.kElastic * p.T11 + mat.viscosity * D11 + pressure + trace * mat.bulkViscosity);

		// Surface tension
		float lenSq = sx * sx + sy * sy;
		if (lenSq > 0)
		{
			float len = sqrtf(lenSq);
			float a = mat.mass * mat.surfaceTension / len;
			T00 -= a * (.5f * lenSq - sx * sx);
			T01 -= a * (-sx * sy);
			T11 -= a * (.5f * lenSq - sy * sy);
		}

		// Wall force
		if (p.x < 4) {
			fx += (4 - p.x);
		}
		else if (p.x > gSizeX - 5) {
			fx += (gSizeX - 5 - p.x);
		}
		if (p.y < 4) {
			fy += (4 - p.y);
		}
		else if (p.y > gSizeY - 5) {
			fy += (gSizeY - 5 - p.y);
		}


		// Add forces to grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;

				float gx = gxi * pyj;
				float gy = pxi * gyj;
				//n->ax += -(gx * T00 + gy * T01) + fx * phi;
				//n->ay += -(gx * T01 + gy * T11) + fy * phi;
				atomicAdd(&(n->ax), -(gx * T00 + gy * T01) + fx * phi);
				atomicAdd(&(n->ay), -(gx * T01 + gy * T11) + fy * phi);
			}
		}

		//Assign final particle Position
		p.pos.x = p.x*scale;
		p.pos.y = p.y*scale;
		p.trail.x = (p.x - p.gu)*scale;
		p.trail.y = (p.y - p.gv)*scale;
	}
}

__global__ void ResetGrid(Node* grid, int gSizeXY)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<gSizeXY) {
		Node* n = &grid[i];
		if (n->active && n->mass > 0) {
			n->u2 = 0;
			n->v2 = 0;
			n->ax /= n->mass;
			n->ay /= n->mass;
		}
	}
}

__global__ void UpdateParticle(Particle* particles, Node* grid, int particleCount, int gSizeY_3, Particle* r_particles, int order)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi<particleCount) {
		Particle& p = particles[pi];
		Material& mat = *p.mat;
		// Update particle velocities
		Node* n = &grid[p.gi];
		float *px = p.px;
		float *py = p.py;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;

				p.u += phi * n->ax;
				p.v += phi * n->ay;
			}
		}

		p.v += mat.gravity;
		p.u *= 1 - mat.damping;
		p.v *= 1 - mat.damping;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;

		// Add particle velocities back to the grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				//n->u2 += phi * mu;
				//n->v2 += phi * mv;
				atomicAdd(&(n->u2), phi*mu);
				atomicAdd(&(n->v2), phi*mv);
			}
		}
		r_particles[pi] = p;
	}
}

__global__ void UpdateParticleSort(Particle* particles, Node* grid, int particleCount, int gSizeY_3, Particle* r_particles, int order)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi<particleCount) {
		Particle& p = particles[pi];
		Particle& p2 = particles[p.writei];
		Material& mat = *p.mat;
		// Update particle velocities
		Node* n = &grid[p.gi];
		float *px = p.px;
		float *py = p.py;
		p2.new_u = p.u;
		p2.new_v = p.v;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;

				p2.new_u = p2.new_u + (phi * n->ax);
				p2.new_v = p2.new_v + (phi * n->ay);
				//p.u += phi * n->ax;
				//p.v += phi * n->ay;
			}
		}


		p2.new_v += mat.gravity;
		p2.new_u *= 1 - mat.damping;
		p2.new_v *= 1 - mat.damping;
		//p.v += mat.gravity;
		//p.u *= 1 - mat.damping;
		//p.v *= 1 - mat.damping;

		float m = p.mat->mass;
		float mu = m * p2.new_u;
		float mv = m * p2.new_v;
		//float mu = m * p.u;
		//float mv = m * p.v;

		// Add particle velocities back to the grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				//n->u2 += phi * mu;
				//n->v2 += phi * mv;
				atomicAdd(&(n->u2), phi*mu);
				atomicAdd(&(n->v2), phi*mv);
			}
		}
		r_particles[pi] = p;
	}
}

__global__ void ResetGrid2(Node* grid, int gSizeXY)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<gSizeXY) {
		Node* n = &grid[i];
		if (n->active && n->mass > 0) {
			n->u2 /= n->mass;
			n->v2 /= n->mass;

			n->mass = 0;
			n->particleDensity = 0;
			n->u = 0;
			n->v = 0;
			memset(n->cgx, 0, numMaterials * sizeof(float));
			memset(n->cgy, 0, numMaterials * sizeof(float));
			n->active = false;
		}
	}
}

string SimulatorCUDA::updateCUDA() {
	//initialize----------------------------------------
	dim3 dimBlockP(1024, 1, 1);
	dim3 dimGridP((particleCount + 1023) / 1024, 1, 1);
	dim3 dimBlockN(gSizeX, 1, 1);
	dim3 dimGridN(gSizeY, 1, 1);
	int gSizeXY = gSizeX*gSizeY;

	Particle* mappedParticles;
	size_t num_bytes;

	//start updating-----------------------------------
	PrepareGridLite << <200, 512 >> >(workarray, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, order); gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
	string output = histogramSort(workarray, workarray2, particleCount, d_histogram, d_probe, PROBESIZE, d_probecheck, d_workitem, d_count, d_prefixsum, d_splitter, order); cudaDeviceSynchronize();
	if (output!="fail") {
		const clock_t begin = clock();
		//ApplyNewParticleAttribute << <200, 512 >> > (workarray, workarray2, particleCount); gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
		if (workarray == d_particles) {
			workarray = d_particles2;
			workarray2 = d_particles;
		}
		else {
			workarray = d_particles;
			workarray2 = d_particles2;
		}
		PrepareGrid4Sort << <200, 512 >> >(workarray, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, order);gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize());
		//PrepareGrid4 << <200, 512 >> >(workarray, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, order); gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
		const clock_t end = clock();
		sout << "sort: " << float(end - begin) / CLOCKS_PER_SEC << endl;
	}
	else {
		const clock_t begin = clock();
		PrepareGrid4 << <200, 512 >> >(workarray, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, order); gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
		const clock_t end = clock();
		sout << "normal: " << float(end - begin) / CLOCKS_PER_SEC << endl;
	}
	//PrepareGrid4Sort << <200, 512 >> >(d_particles, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3); gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
	//PrepareGrid4 << <200, 512 >> >(workarray, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, order); gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

	////partikel atribut
	//if (countdown >= 30) {
	//	//return output;
	//	return sout.str();
	//	////preparegridtest << <200, 512 >> >(d_particles, d_grid, particlecount, gsizex, gsizey, gsizey_3); gpuerrchk(cudapeekatlasterror()); gpuerrchk(cudadevicesynchronize());

	//	//float gr[20000];
	//	//for (int i = 0; i < 128 * 128; i++) {
	//	//	cudaMemcpy(&gr[i], &(d_grid[i].mass), sizeof(float), cudaMemcpyDeviceToHost);
	//	//	cudaMemcpy(&gr[i], &(d_grid[i].particleDensity), sizeof(float), cudaMemcpyDeviceToHost);
	//	//	cudaMemcpy(&gr[i], &(d_grid[i].u), sizeof(float), cudaMemcpyDeviceToHost);
	//	//	cudaMemcpy(&gr[i], &(d_grid[i].v), sizeof(float), cudaMemcpyDeviceToHost);
	//	//}
	//	Particle *ps;
	//	ps = (Particle*)malloc(sizeof(Particle) * particleCount);
	//	cudaMemcpy(ps, workarray, sizeof(Particle) * particleCount, cudaMemcpyDeviceToHost);
	//	//cudaMemcpy(gr, d_grid, sizeof(Node) * 200 * 100, cudaMemcpyDeviceToHost);
	//	gpuErrchk(cudaDeviceSynchronize());

	//	std::stringstream ss;
	//	for (int i = 0; i < particleCount; i++) {
	//		ss << i << ") " << ps[i].writei << "\t " << ps[i].gi << endl;
	//		//if(gr[i]>0)		ss << i << ") " << gr[i] << "\t " << endl;
	//	}
	//	return ss.str();
	//}

	FinalizeGrid << <200, 400 >> >(d_grid, gSizeXY);gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize());
	CalculateForces << <200, 512 >> >(workarray, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, scale, order);gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize());
	ResetGrid << <200, 400 >> >(d_grid, gSizeXY);gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize());

	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);gpuErrchk(cudaPeekAtLastError());
	cudaGraphicsResourceGetMappedPointer((void **)&mappedParticles, &num_bytes, cuda_vbo_resource);gpuErrchk(cudaPeekAtLastError());

	UpdateParticle << <200, 512 >> >(workarray, d_grid, particleCount, gSizeY_3, mappedParticles, order);gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize())
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);gpuErrchk(cudaPeekAtLastError());
	ResetGrid2 << <200, 400 >> >(d_grid, gSizeXY);gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize());

	//end-----------------------------------------------
	countdown++;
	return "wa";
	//return output;
	//app::console() << particles[0].pos.x << ","<< particles[0].pos.y << endl;
}



