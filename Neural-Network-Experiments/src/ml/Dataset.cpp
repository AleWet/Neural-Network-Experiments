#include "Dataset.h"
#include <algorithm>
#include <numeric>

bool Dataset::loadMNIST_CSV(const std::string& filepath, int maxSamples) 
{
    std::ifstream file(filepath);
    if (!file.is_open()) 
    {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    m_samples.clear();
    std::string line;
    int samplesLoaded = 0;

    std::cout << "Loading MNIST data from " << filepath << "..." << std::endl;

    // Skip header if exists (check first line)
    std::getline(file, line);
    if (line.find("label") != std::string::npos) 
    {
        // This was a header, continue
    }
    else 
    {
        // This was data, process it
        file.seekg(0); // Go back to beginning
    }

    while (std::getline(file, line) && (maxSamples == -1 || samplesLoaded < maxSamples)) 
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> values;

        // Parse CSV line
        while (std::getline(ss, cell, ',')) 
        {
            try 
            {
                values.push_back(std::stof(cell));
            }
            catch (const std::exception& e) 
            {
                std::cerr << "Error parsing value: " << cell << std::endl;
                continue;
            }
        }

        if (values.size() != 785) 
        { // 1 label + 784 pixels
            std::cerr << "Invalid data format. Expected 785 values, got " << values.size() << std::endl;
            continue;
        }

        DataSample sample;
        sample.label = static_cast<int>(values[0]);

        // Create input vector (784 pixels)
        sample.input = Eigen::VectorXf(784);
        for (int i = 0; i < 784; i++) 
            sample.input[i] = values[i + 1] / 255.0f; // Normalize to [0,1]

        // Create one-hot encoded target
        sample.target = oneHotEncode(sample.label, 10);

        m_samples.push_back(sample);
        samplesLoaded++;

        if (samplesLoaded % 1000 == 0) 
            std::cout << "Loaded " << samplesLoaded << " samples..." << std::endl;
    }

    file.close();

    // Initialize indices for shuffling
    m_indices.resize(m_samples.size());
    std::iota(m_indices.begin(), m_indices.end(), 0);

    std::cout << "Successfully loaded " << m_samples.size() << " MNIST samples." << std::endl;
    return true;
}

const DataSample& Dataset::getRandomSample() 
{
    std::uniform_int_distribution<size_t> dis(0, m_samples.size() - 1);
    return m_samples[m_indices[dis(m_gen)]];
}

const DataSample& Dataset::getNextSample() 
{
    const DataSample& sample = m_samples[m_indices[m_currentIndex]];
    m_currentIndex = (m_currentIndex + 1) % m_samples.size();
    return sample;
}

std::vector<DataSample> Dataset::getBatch(size_t batchSize) 
{
    std::vector<DataSample> batch;
    batch.reserve(batchSize);

    for (size_t i = 0; i < batchSize && i < m_samples.size(); i++) 
    {
        batch.push_back(getNextSample());
    }

    return batch;
}

void Dataset::shuffle() 
{
    std::shuffle(m_indices.begin(), m_indices.end(), m_gen);
    m_currentIndex = 0;
}

std::vector<int> Dataset::getLabelCounts() const 
{
    std::vector<int> counts(10, 0); // Max 10 classes (what?)
    for (const auto& sample : m_samples) 
    {
        if (sample.label >= 0 && sample.label < 10) 
            counts[sample.label]++;
    }
    return counts;
}

Eigen::VectorXf Dataset::oneHotEncode(int label, int numClasses) 
{
    Eigen::VectorXf encoded = Eigen::VectorXf::Zero(numClasses);
    if (label >= 0 && label < numClasses) 
        encoded[label] = 1.0f;
    return encoded;
}