#include <iostream>

float response(const float *x, const float *w, int size) {

    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += w[i] * x[i];
    }

    return sum;
}

int sign(float x) {
    if (x > 0) {
        return 1;
    }
    return -1;
}

void printWeights(float *w, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << w[i] << ", ";
    }
    std::cout << std::endl;
}

void updateWeights(int yHat, int y, float mu, const float *x, float *w, int size) {
    for (int i = 0; i < size; i++) {
        int wOld = w[i];
        w[i] += mu * (y - yHat) * x[i];
        //std::cout << w[i] <<" = "<< wOld << " + " << mu << " * (" << yHat << "-" << y << ") * " << x[i] << std::endl;
    }
    //printWeights(w, size);
}

int classify(float *x, float *w, int size) {
    return sign(response(x, w, size));
}

int main() {
    int inputSize = 3; //{bias, x1, x2}
    int dataSize = 6;
    float x[6][3] = {{1, 0,  0},
                     {1, 1,  2},
                     {1, 2,  1},
                     {1, 7,  8},
                     {1, 10, 15},
                     {1, 5,  10}};
    int y[6] = {-1, -1, -1, 1, 1, 1};
    float weights[3] = {0, 0, 0};
    int epochs = 20;

    int yHat;
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < dataSize; j++) {
            yHat = classify(x[j], weights, inputSize);
            updateWeights(yHat, y[j], 0.01, x[j], weights, inputSize);
            //printWeights(weights, inputSize);
        }
        float classRate = 0;
        for (int j = 0; j < dataSize; j++) {
            yHat = classify(x[j], weights, inputSize);
            classRate += (yHat == y[j]);
        }
        std::cout << " Epoch: " << i << " - " << classRate << "/" << dataSize << std::endl;
        if(classRate == dataSize){
            break;
        }
    }
    for (int j = 0; j < dataSize; j++) {
        yHat = classify(x[j], weights, inputSize);
        std::cout << yHat << ", " << y[j] << " -> ";
        for(int k = 0 ; k < inputSize; k++){
            std::cout << x[j][k] << " ";
        }
        std::cout << std::endl;
    }
    printWeights(weights, inputSize);
    return 0;
}
