#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include "Eigen/Dense"

struct DataSample 
{
    Eigen::VectorXf input;
    Eigen::VectorXf target;
    int label;              // Display
};

class Dataset 
{
private:
    std::vector<DataSample> m_Samples;
    std::vector<size_t> m_Indices;      // For shuffling without moving data
    size_t m_CurrentIndex = 0;
    std::random_device m_Rd;            // Random number gen
    std::mt19937 m_Gen;                 // Random number gen

public:
    Dataset() : m_Gen(m_Rd()) {}

    // MNIST from CSV (Kaggle version)
    bool loadMNIST_CSV(const std::string& filepath, int maxSamples = -1);

    // Data access
    const DataSample& getSample(size_t index) const { return m_Samples[m_Indices[index]]; }
    const DataSample& getRandomSample();
    const DataSample& getNextSample();     // Sequential access

    // Batch operations
    std::vector<DataSample> getBatch(size_t batchSize);
    void shuffle();
    void reset() { m_CurrentIndex = 0; } // Reset sequential access

    // Info
    size_t size() const { return m_Samples.size(); }
    bool empty() const { return m_Samples.empty(); }
    int getInputSize() const { return m_Samples.empty() ? 0 : m_Samples[0].input.size(); }
    int getOutputSize() const { return m_Samples.empty() ? 0 : m_Samples[0].target.size(); }

    // Get number of lables for MNIST
    std::vector<int> getLabelCounts() const;

    // Utility
    static Eigen::VectorXf oneHotEncode(int label, int numClasses = 10);
    static Eigen::VectorXf normalizePixels(const Eigen::VectorXf& pixels) { return pixels / 255.0f; }
};