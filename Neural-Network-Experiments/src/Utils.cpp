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

void UpdateMNISTTexture(GLuint& texture, const std::vector<float>& imageData) {
    // Create texture if it doesn't exist
    if (texture == 0) 
        glGenTextures(1, &texture);
    

    // Convert float data to unsigned char for OpenGL
    std::vector<unsigned char> pixelData(784);
    for (int i = 0; i < 784; i++) 
        pixelData[i] = static_cast<unsigned char>(imageData[i] * 255.0f);
    

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 28, 28, 0, GL_RED, GL_UNSIGNED_BYTE, pixelData.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
}