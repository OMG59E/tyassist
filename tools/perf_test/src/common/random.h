#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED
 
#include <iostream>
#include <time.h>
#include <limits.h>
#include <math.h>
 
using namespace std;
 
class Random {
 
public:
    Random(bool pseudo = true);
    double random_real();
    int random_integer(int low, int high);
    int poisson(double mean);
 
private:
    int reseed(); //  Re-randomize the seed.
    int seed,multiplier,add_on;   //  constants for use in arithmetic operations
 
};
 
#endif // RANDOM_H_INCLUDED