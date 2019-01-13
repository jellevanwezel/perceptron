#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

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
        if (i != size - 1) {
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

void onEpochEnd(int *indexes, int size) {
    std::random_shuffle(indexes, indexes + size);
}

float test(float **x, const int *y, const float *w, const int dataSize, const int inputSize) {
    float classRate = 0;
    for (int i = 0; i < dataSize; i++) {
        int yHat = classify(x[i], w, inputSize);
        classRate += (yHat == y[i]);
    }
    return 1 - (classRate / dataSize);
}

float *train(int epochs, float eta, float **x, const int *y, float *w, const int dataSize, const int inputSize) {
    int *dataIndexes = new int[dataSize];
    for (int i = 0; i < dataSize; i++) {
        dataIndexes[i] = i;
    }
    onEpochEnd(dataIndexes, dataSize);

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < dataSize; j++) {
            int index = dataIndexes[j];
            int yHat = classify(x[index], w, inputSize);
            updateWeights(yHat, y[index], eta, x[index], w, inputSize);
        }
        float errorRate = test(x, y, w, dataSize, inputSize);
        std::cout << "Epoch: " << i << " - Error rate: " << errorRate << std::endl;
        if (errorRate == 0) {
            break;
        }
        onEpochEnd(dataIndexes, dataSize);
    }

    delete[] dataIndexes;
    return w;
}


std::vector<std::vector<float>> parseCSV(const std::string &filePath) {
    std::ifstream data(filePath);
    std::string line;
    std::vector<std::vector<float>> parsedCsv;
    while (std::getline(data, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> parsedRow;
        while (std::getline(lineStream, cell, ',')) {
            float f = std::stof(cell);
            parsedRow.push_back(f);
        }
        parsedCsv.push_back(parsedRow);
    }
    return parsedCsv;
};

float **getData(vector<vector<float>> &vals, int N, int M) {
    float **temp;
    temp = new float *[N];
    for (unsigned i = 0; (i < N); i++) {
        temp[i] = new float[M];
        temp[i][0] = 1; // add the bias
        for (unsigned j = 0; (j < M - 1); j++) { // the last value is the label
            temp[i][j + 1] = vals[i][j];
        }
    }
    return temp;
}

int *getLabels(vector<vector<float>> &vals, int N, int M) {
    int *temp;
    temp = new int[N];
    for (unsigned i = 0; (i < N); i++) {
        temp[i] = static_cast<int>(vals[i][M-1]); //only take last value
    }
    return temp;
}

int main(int argc, char **argv) {

    ifstream infile(argv[1]);
    std::string line;
    vector<double> constants;
    std::cout << argv[1] << std::endl;
    vector<vector<float>> data = parseCSV(argv[1]);
    vector<vector<float>> labels = parseCSV(argv[1]);

    int inputSize = static_cast<int>(data[0].size());
    int dataSize = static_cast<int>(data.size());
    float **x = getData(data, dataSize, inputSize);
    int *y = getLabels(data, dataSize, inputSize);

    auto *weights = new float[inputSize]{0, 0, 0};
    int epochs = 50;
    float eta = 0.01;

    train(epochs, eta, x, y, weights, dataSize, inputSize); // +1 for the bias
    printWeights(weights, inputSize);

    for (int i = 0; i < dataSize; i++) { delete[] x[i]; }
    delete[] x;
    delete[] y;
    delete[] weights;
    return 0;
}
