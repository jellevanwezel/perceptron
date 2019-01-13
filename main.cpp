#include <iostream>

float response(const float *x, const float *w, const int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += w[i] * x[i];
    }
    return sum;
}

int sign(const float x) {
    if (x > 0) {
        return 1;
    }
    return -1;
}

void printWeights(const float *w, const int size) {
    std::cout << "Weights: [";
    for (int i = 0; i < size; i++) {
        std::cout << w[i];
        if(i != size - 1){
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void updateWeights(const int yHat, const int y, const float mu, const float *x, float *w, const int size) {
    for (int i = 0; i < size; i++) {
        w[i] += mu * (y - yHat) * x[i];
    }
}

int classify(const float *x, const float *w, const int size) {
    return sign(response(x, w, size));
}

float test(float **x, const int *y, const float *w, const int dataSize, const int inputSize){
    float classRate = 0;
    for(int i=0;i<dataSize; i++){
        int yHat = classify(x[i], w, inputSize);
        classRate += (yHat == y[i]);
    }
    return 1 - (classRate / dataSize);
}

float* train(int epochs, float eta, float **x, const int *y, float *w, const int dataSize, const int inputSize){
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < dataSize; j++) {
            int yHat = classify(x[j], w, inputSize);
            updateWeights(yHat, y[j], eta, x[j], w, inputSize);
        }
        float errorRate = test(x, y, w, dataSize, inputSize);
        std::cout << "Epoch: " << i << " - Error rate: " << errorRate << std::endl;
        if(errorRate == 0){
            break;
        }
    }
    return w;
}

int main() {
    int inputSize = 3; //{bias, x1, x2}
    int dataSize = 6;

    float **x;
    x = new float*[dataSize];
    x[0] = new float[inputSize]{1,0,0};
    x[1] = new float[inputSize]{1,1,2};
    x[2] = new float[inputSize]{1,2,1};
    x[3] = new float[inputSize]{1,7,8};
    x[4] = new float[inputSize]{1,10,15};
    x[5] = new float[inputSize]{1,5,10};

    int y[6] = {-1, -1, -1, 1, 1, 1};
    float weights[3] = {0, 0, 0};
    int epochs = 50;
    float eta = 0.01;

    train(epochs, eta, x, y, weights, dataSize, inputSize);
    printWeights(weights, inputSize);
    return 0;
}
