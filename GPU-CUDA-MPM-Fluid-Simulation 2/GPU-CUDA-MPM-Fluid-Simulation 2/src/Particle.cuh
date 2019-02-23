#ifndef PARTICLE_H
#define PARTICLE_H

#include "cinder/app/App.h"
#include "Material.cuh"

struct Particle {
	cinder::vec3		pos;
	cinder::vec3        trail;
	cinder::ColorA		color;
	cinder::ColorA		new_color;

	Material* mat;
	float x, y, u, v, gu, gv, T00, T01, T11;
	float new_x, new_y, new_u, new_v, new_gu, new_gv, new_T00, new_T01, new_T11;
	
	int cx, cy, gi;
	int new_gi, writei;
	float px[3];
	float py[3];
	float gx[3];
	float gy[3];
	float new_px[3];
	float new_py[3];
	float new_gx[3];
	float new_gy[3];

	Particle(Material* mat);

	Particle(Material* mat, float x, float y);

	Particle(Material* mat, float x, float y, cinder::ColorA c);

	Particle(Material* mat, float x, float y, float u, float v);

	~Particle();

	void initAttArrays(float* px);

	void initializeWeights(int gSizeY);
};

#endif