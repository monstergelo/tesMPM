#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "cinder/app/App.h"
#include "Material.cuh"
#include "Node.cuh"
#include "Particle.cuh"

class SimulatorCUDA{
	const int NPATCH = 2178;
	const int NPATCH_V = 33;
	const int NPATCH_H = 66;
	const int MAXPARTICLE = 12000;

	int *Pindex;
	int *startIndex;

	int gSizeX, gSizeY, gSizeY_3;

	Node* grid;
	Node* d_grid;
	int countdown = 0;

	std::vector<Node*> active;

	float* d_gridAtt;
	float* d_particleAtt;
	float* d_particleAtt2;

	Material materials[numMaterials];
	Material* d_materials;


	//sort stuff
	int *d_histogram;
	int *d_probe;
	int probeSize;
	bool *d_probecheck;
	int *d_workitem;
	int *d_count;
	int *d_prefixsum;
	int *d_splitter;

	float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);
public:
	std::vector<Particle> particles;
	Particle* d_particles;
	Particle* d_particles2;
	int particleCount;
	struct cudaGraphicsResource *cuda_vbo_resource;

	float scale;

	SimulatorCUDA();
	void initializeHelper();
	void initializeGrid(int sizeX, int sizeY);
	void addParticles(int n);
	void update();
	std::string updateCUDA();

	Particle* workarray;
	Particle* workarray2;

	std::stringstream sout;
	int order = 0;
};