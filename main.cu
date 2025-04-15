#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<math.h>

#include <sys/time.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

struct Planet {
   double mass;
   double x;
   double y;
   double vx;
   double vy;
};

unsigned long long seed = 100;

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

double randomDouble()
{
   unsigned long long next = randomU64();
   next >>= (64 - 26);
   unsigned long long next2 = randomU64();
   next2 >>= (64 - 26);
   return ((next << 27) + next2) / (double)(1LL << 53);
}

int nplanets;
int timesteps;
double dt;
double G;

// Planet* next(Planet* planets) {
//    Planet* nextplanets = (Planet*)malloc(sizeof(Planet) * nplanets);
//    #pragma omp parallel for
//    for (int i=0; i<nplanets; i++) {
//       nextplanets[i].vx = planets[i].vx;
//       nextplanets[i].vy = planets[i].vy;
//       nextplanets[i].mass = planets[i].mass;
//       nextplanets[i].x = planets[i].x;
//       nextplanets[i].y = planets[i].y;
//    }

//    #pragma omp parallel for
//    for (int i=0; i<nplanets; i++) {
//       for (int j=i + 1; j<nplanets; j++) {
//          double dx = planets[j].x - planets[i].x;
//          double dy = planets[j].y - planets[i].y;
//          double distSqr = dx*dx + dy*dy + 0.0001;
//          double invDist = planets[i].mass * planets[j].mass / sqrt(distSqr);
//          double invDist3 = invDist * invDist * invDist;
//          nextplanets[i].vx += dt * dx * invDist3;
//          nextplanets[i].vy += dt * dy * invDist3;
//       }
//       nextplanets[i].x += dt * nextplanets[i].vx;
//       nextplanets[i].y += dt * nextplanets[i].vy;
//    }
//    free(planets);
//    return nextplanets;
// }

// __global__ void next_kernel(Planet* planets, Planet* nextplanets, int nplanets, double dt) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= nplanets) return;

//     nextplanets[i].vx = planets[i].vx;
//     nextplanets[i].vy = planets[i].vy;
//     nextplanets[i].mass = planets[i].mass;
//     nextplanets[i].x = planets[i].x;
//     nextplanets[i].y = planets[i].y;

//     for (int j = 0; j < nplanets; j++) {
//         if (i == j) continue;
//         double dx = planets[j].x - planets[i].x;
//         double dy = planets[j].y - planets[i].y;
//         double distSqr = dx*dx + dy*dy + 0.0001;
//         double invDist = planets[i].mass * planets[j].mass / sqrt(distSqr);
//         double invDist3 = invDist * invDist * invDist;
//         nextplanets[i].vx += dt * dx * invDist3;
//         nextplanets[i].vy += dt * dy * invDist3;
//     }
//     nextplanets[i].x += dt * nextplanets[i].vx;
//     nextplanets[i].y += dt * nextplanets[i].vy;
// }

__device__ double dev_G = 6.6743;
__global__ void next_kernel(Planet* planets, Planet* nextplanets, int nplanets, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nplanets) return;

    double fx = 0.0, fy = 0.0;
    double xi = planets[i].x;
    double yi = planets[i].y;
    double mi = planets[i].mass;

    for (int j = 0; j < nplanets; j++) {
        if (i == j) continue;
        double dx = planets[j].x - xi;
        double dy = planets[j].y - yi;
        double distSqr = dx * dx + dy * dy + 0.0001;
        double dist = sqrt(distSqr);
        // diff is dev_G multiplication here
        double force = dev_G * mi * planets[j].mass / distSqr;
        fx += force * dx / dist;
        fy += force * dy / dist;
    }

    nextplanets[i].vx = planets[i].vx + dt * fx / mi;
    nextplanets[i].vy = planets[i].vy + dt * fy / mi;
    nextplanets[i].x = xi + dt * nextplanets[i].vx;
    nextplanets[i].y = yi + dt * nextplanets[i].vy;
    nextplanets[i].mass = mi;
}




int main(int argc, const char** argv){
//    if (argc < 2) {
//       printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
//       return 1;
//    }
//    nplanets = atoi(argv[1]);
//    timesteps = atoi(argv[2]);
    nplanets = 1000;
    timesteps = 5000;

    dt = 0.001;
    G = 6.6743;

    struct timeval start, end;
    // gettimeofday(&start, NULL);

    Planet* planets = (Planet*)malloc(sizeof(Planet) * nplanets);
    for (int i = 0; i < nplanets; i++) {
        planets[i].mass = randomDouble() * 10 + 0.2;
        planets[i].x = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
        planets[i].y = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
        planets[i].vx = randomDouble() * 5 - 2.5;
        planets[i].vy = randomDouble() * 5 - 2.5;
    }

    // gettimeofday(&end, NULL);
    // printf("Total time to initalize %0.6f seconds\n", tdiff(&start, &end));

    Planet *d_planets, *d_nextplanets;
    cudaMalloc(&d_planets, sizeof(Planet) * nplanets);
    cudaMalloc(&d_nextplanets, sizeof(Planet) * nplanets);

    cudaMemcpy(d_planets, planets, sizeof(Planet) * nplanets, cudaMemcpyHostToDevice);

    int blockSize = 16;
    int numBlocks = (nplanets + blockSize - 1) / blockSize;

   gettimeofday(&start, NULL);
//    for (int i=0; i<timesteps; i++) {
//       planets = next(planets);
//       // printf("x=%f y=%f vx=%f vy=%f\n", planets[nplanets-1].x, planets[nplanets-1].y, planets[nplanets-1].vx, planets[nplanets-1].vy);
//    }


    for (int i = 0; i < timesteps; i++) {
        // gettimeofday(&start, NULL);
        next_kernel<<<numBlocks, blockSize>>>(d_planets, d_nextplanets, nplanets, dt);
        cudaDeviceSynchronize();
        // gettimeofday(&end, NULL);
        // printf("Kernel execution time %0.6f seconds\n", tdiff(&start, &end));
        Planet* tmp = d_planets;
        d_planets = d_nextplanets;
        d_nextplanets = tmp;
    }

    cudaMemcpy(planets, d_nextplanets, sizeof(Planet) * nplanets, cudaMemcpyDeviceToHost);

    cudaFree(d_planets);
    cudaFree(d_nextplanets);




   gettimeofday(&end, NULL);
   printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets[nplanets-1].x, planets[nplanets-1].y);

   return 0;   
}
