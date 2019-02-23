#ifndef NODE_H
#define NODE_H

#define numMaterials 4

struct Node {
	float mass, particleDensity, gx, gy, u, v, u2, v2, ax, ay;
	float* cgx;
	float* cgy;
	int particleIndex[10];
	int particleCount = 0;
	bool active;
	Node();
	~Node();
	void initAttArrays(float* cgx);
};

#endif