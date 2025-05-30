#include "Utils.h"

bool IsShaderPathOk(std::string shaderPath)
{
    std::ifstream fileCheck(shaderPath);
    if (!fileCheck.good()) {
        std::cerr << "Error: Cannot open shader file: " << shaderPath << std::endl;
        // Handle the error (set a flag or throw an exception)
        return false;
    }
    return true;
}

void UpdateTrainingMetrics(int epoch, float loss, float accuracy,
    std::vector<float>& lossHist,
    std::vector<float>& accuracyHist,
    std::vector<float>& epochNums,
    int maxSize) 
{
    epochNums.push_back(static_cast<float>(epoch));
    lossHist.push_back(loss);
    accuracyHist.push_back(accuracy);

    // Keep history size manageable
    if (epochNums.size() > maxSize) 
    {
        epochNums.erase(epochNums.begin());
        lossHist.erase(lossHist.begin());
        accuracyHist.erase(accuracyHist.begin());
    }
}