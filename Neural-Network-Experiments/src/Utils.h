#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

bool IsShaderPathOk(std::string shaderPath);

void UpdateTrainingMetrics(int epoch, float loss, float accuracy,
    std::vector<float>& lossHist,
    std::vector<float>& accuracyHist,
    std::vector<float>& epochNums,
    int maxSize);

void UpdateMNISTTexture(GLuint& texture, const std::vector<float>& imageData);