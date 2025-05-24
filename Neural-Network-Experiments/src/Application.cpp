#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include "ml/Network.h"
#include "ml/Dataset.h"
#include "Utils.h"

#include "graphics/VertexBuffer.h"
#include "graphics/IndexBuffer.h"
#include "graphics/VertexArray.h"
#include "graphics/VertexBufferLayout.h"
#include "graphics/Shader.h"
#include "graphics/Texture.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "vendor/imgui/imgui.h"
#include "vendor/imgui/imgui_impl_glfw.h"
#include "vendor/imgui/imgui_impl_opengl3.h"

int main(void)
{
    #pragma region Initialize libraries

    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set OpenGL version hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //core profile ==> no standard VA 

    // GLFW stuff
    GLFWwindow* window = glfwCreateWindow(1280, 960, "", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // GLEW stuff
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    GLCall(glViewport(0, 0, 1280, 960));

    // ImGui stuff
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    // DEBUG
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "ImGui Version : " << ImGui::GetVersion() << std::endl;

    #pragma endregion

    { //additional scope to avoid memory leaks

        #pragma region Initialize variables 

        // Initialize 
        std::string Basic = "res/shaders/Basic.shader";
        if (!IsShaderPathOk(Basic)) return 0;
        Shader BasicShader(Basic);

        // MISSING RENDERER

        Dataset dataset;
        bool datasetLoaded = false;
        std::string datasetPath = "path/to/mnist_train.csv";
        char pathBuffer[256] = "path/to/mnist_train.csv";
        int maxSamples = 1000;                // Limit for testing
        int currentSampleIndex = 0;
        bool showSampleViewer = false;
        int selectedDatasetType = 0;          // 0=MNIST for now only this
        float learningRate = 0.01f;

        int numberOfLayers = 4;               // Total layers including input and output
        std::vector<int> layerSizes(4);       // Vector to store nodes for each layer
        layerSizes[0] = 784;                  // Default input size for MNIST
        layerSizes[1] = 128;                  // Default first hidden layer
        layerSizes[2] = 64;                   // Default second hidden layer
        layerSizes[3] = 10;                   // Default output size for MNIST

        bool autoConfigureInputOutput = true; // Auto-set input/output based on dataset
        bool networkCreated = false;
        std::vector<int> sizes = { 1, 1 };
        Network network(sizes);

        //Eigen::setNbThreads(4); // (?)

        #pragma endregion 

        // Main loop
        while (!glfwWindowShouldClose(window))
        {
            // Pre-Rendering 
            GLCall(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
            GLCall(glClear(GL_COLOR_BUFFER_BIT));
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            #pragma region ImGUI UI

            ImGui::Begin("Neural Network Trainer");

            // Network Dataset configuration
            if (ImGui::CollapsingHeader("Dataset", ImGuiTreeNodeFlags_DefaultOpen))
            {
                // Dataset type selection
                const char* datasetTypes[] = { "MNIST" }; // ADD MORE IN THE FUTURE
                ImGui::Combo("Dataset Type", &selectedDatasetType, datasetTypes, 1);

                if (selectedDatasetType == 0)  // MNIST
                {
                    ImGui::InputText("Dataset Path", pathBuffer, sizeof(pathBuffer));
                    ImGui::InputInt("Max Samples (testing)", &maxSamples);

                    if (ImGui::Button("Load MNIST Dataset"))
                    {
                        datasetPath = std::string(pathBuffer);
                        if (dataset.loadMNIST_CSV(datasetPath, maxSamples))
                        {
                            datasetLoaded = true;
                            currentSampleIndex = 0;

                            // Auto-configure input/output layer sizes if enabled
                            if (autoConfigureInputOutput)
                            {
                                layerSizes[0] = dataset.getInputSize();
                                layerSizes[numberOfLayers - 1] = dataset.getOutputSize();
                            }

                            std::cout << "Dataset loaded successfully!" << std::endl;
                        }
                        else
                        {
                            datasetLoaded = false;
                            std::cout << "Failed to load dataset!" << std::endl;
                        }
                    }
                }
                // ADD MORE DATA LOADING OPTIONS

                // Dataset info
                if (datasetLoaded)
                {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Dataset loaded: %zu samples", dataset.size());
                    ImGui::Text("Input size: %d", dataset.getInputSize());
                    ImGui::Text("Output size: %d", dataset.getOutputSize());

                    if (selectedDatasetType == 0) // Show label distribution for MNIST
                    {
                        auto labelCounts = dataset.getLabelCounts();
                        ImGui::Text("Label distribution:");
                        for (int i = 0; i < 10; i++)
                        {
                            if (labelCounts[i] > 0)
                                ImGui::Text("  %d: %d samples", i, labelCounts[i]);
                        }
                    }

                    ImGui::Checkbox("Show Sample Viewer", &showSampleViewer);
                }
                else
                {
                    ImGui::TextColored(ImVec4(1, 0, 0, 1), "No dataset loaded");
                }
            }

            // Network Architecture section
            if (ImGui::CollapsingHeader("Network Architecture", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Checkbox("Auto-configure input/output layers", &autoConfigureInputOutput);
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Automatically set input/output layer sizes based on loaded dataset");
                }

                // Number of layers input with constraints
                int prevLayers = numberOfLayers;
                ImGui::InputInt("Number of Layers", &numberOfLayers);

                if (numberOfLayers < 2) numberOfLayers = 2;   // minimum 2: input + output
                if (numberOfLayers > 20) numberOfLayers = 20; // Reasonable upper limit

                // Resize layerSizes vector if number of layers changed
                if (numberOfLayers != prevLayers)
                {
                    int oldSize = layerSizes.size();
                    layerSizes.resize(numberOfLayers);

                    // Initialize new layers with reasonable defaults
                    for (int i = oldSize; i < numberOfLayers; i++)
                    {
                        if (i == 0)                       // Input layer
                            layerSizes[i] = datasetLoaded ? dataset.getInputSize() : 784;
                        else if (i == numberOfLayers - 1) // Output layer
                            layerSizes[i] = datasetLoaded ? dataset.getOutputSize() : 10;
                        else                              // Hidden layers
                            layerSizes[i] = 64;           // Default hidden layer size
                    }
                }

                ImGui::Separator();
                ImGui::Text("Layer Configuration:");

                // Display input fields for each layer
                for (int i = 0; i < numberOfLayers; i++)
                {
                    std::string label;
                    ImVec4 textColor = ImVec4(1, 1, 1, 1); // Default white

                    if (i == 0)
                    {
                        label = "Input Layer             ";
                        textColor = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); // Light green
                    }
                    else if (i == numberOfLayers - 1)
                    {
                        label = "Output Layer            ";
                        textColor = ImVec4(1.0f, 0.5f, 0.5f, 1.0f); // Light red
                    }
                    else
                    {
                        label = "Hidden Layer " + std::to_string(i) + " (Layer " + std::to_string(i) + ")";
                        textColor = ImVec4(0.7f, 0.7f, 1.0f, 1.0f); // Light blue
                    }

                    ImGui::TextColored(textColor, "%s", label.c_str());
                    ImGui::SameLine();

                    std::string inputLabel = "##nodes" + std::to_string(i);

                    // Disable input/output layer editing if auto-configure is enabled and dataset is loaded
                    bool isDisabled = autoConfigureInputOutput && datasetLoaded && (i == 0 || i == numberOfLayers - 1);

                    if (isDisabled)
                    {
                        ImGui::BeginDisabled();
                    }

                    ImGui::InputInt(inputLabel.c_str(), &layerSizes[i]);

                    // Constrain node counts to reasonable values
                    if (layerSizes[i] < 1) layerSizes[i] = 1;
                    if (layerSizes[i] > 10000) layerSizes[i] = 10000;

                    if (isDisabled)
                    {
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        {
                            ImGui::SetTooltip("Disabled due to auto-configuration. Uncheck 'Auto-configure input/output layers' to edit manually.");
                        }
                    }
                }

                // Calculate and display total parameters
                int totalParams = 0;
                for (int i = 1; i < numberOfLayers; i++)
                    totalParams += layerSizes[i - 1] * layerSizes[i] + layerSizes[i]; // weights + biases
                
                ImGui::Text("Total Parameters: %d", totalParams);

                // Preset buttons for common architectures
                ImGui::Text("Quick Presets:");

                if (ImGui::Button("Simple (3 layers)"))
                {
                    numberOfLayers = 3;
                    layerSizes.resize(3);
                    layerSizes[0] = datasetLoaded ? dataset.getInputSize() : 784;
                    layerSizes[1] = 128;
                    layerSizes[2] = datasetLoaded ? dataset.getOutputSize() : 10;
                }
                ImGui::SameLine();

                if (ImGui::Button("Deep (5 layers)"))
                {
                    numberOfLayers = 5;
                    layerSizes.resize(5);
                    layerSizes[0] = datasetLoaded ? dataset.getInputSize() : 784;
                    layerSizes[1] = 256;
                    layerSizes[2] = 128;
                    layerSizes[3] = 64;
                    layerSizes[4] = datasetLoaded ? dataset.getOutputSize() : 10;
                }
                ImGui::SameLine();

                if (ImGui::Button("Wide (4 layers)"))
                {
                    numberOfLayers = 4;
                    layerSizes.resize(4);
                    layerSizes[0] = datasetLoaded ? dataset.getInputSize() : 784;
                    layerSizes[1] = 512;
                    layerSizes[2] = 256;
                    layerSizes[3] = datasetLoaded ? dataset.getOutputSize() : 10;
                }

                if (ImGui::Button("Create Network"))
                {
                    // Auto-configure input/output if enabled and dataset is loaded
                    if (autoConfigureInputOutput && datasetLoaded)
                    {
                        layerSizes[0] = dataset.getInputSize();
                        layerSizes[numberOfLayers - 1] = dataset.getOutputSize();
                    }

                    // Create network with current layer configuration
                    network = Network(layerSizes);
                    networkCreated = true;

                    std::cout << "Network created with architecture: ";
                    for (int i = 0; i < numberOfLayers; i++)
                    {
                        std::cout << layerSizes[i];
                        if (i < numberOfLayers - 1) std::cout << " -> ";
                    }
                    std::cout << std::endl;
                    std::cout << "Total parameters: " << totalParams << std::endl;
                }

                if (networkCreated)
                {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Network Ready");
                }

                
            }

            // Training section TO BE REVISED 
            if (ImGui::CollapsingHeader("Training"))
            {
                ImGui::Button("Select activation function (TBD)");

                ImGui::SliderFloat("Learning Rate", &learningRate, 0.0f, 1.0f);

                if (datasetLoaded && networkCreated)
                {
                    if (ImGui::Button("Start Learning"))
                    {
                        std::cout << "Training would start here..." << std::endl;
                        // TODO: Implement backpropagation training
                    }

                    if (ImGui::Button("Test Single Sample"))
                    {
                        const DataSample& sample = dataset.getRandomSample();
                        Eigen::VectorXf output = network.forward(sample.input);

                        std::cout << "Input label: " << sample.label << std::endl;
                        std::cout << "Network output: " << std::endl << output.transpose() << std::endl;
                        std::cout << "Expected output: " << std::endl << sample.target.transpose() << std::endl;
                    }

                    if (ImGui::Button("Shuffle Dataset"))
                    {
                        dataset.shuffle();
                        std::cout << "Dataset shuffled." << std::endl;
                    }
                }
                else if (!datasetLoaded)
                {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Load a dataset first");
                }
                else if (!networkCreated)
                {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Create a network first");
                }
            }

            ImGui::End(); 

            // Sample Viewer (separate window)
            if (showSampleViewer && datasetLoaded && selectedDatasetType == 0) // Only for MNIST
            {
                ImGui::Begin("MNIST Sample Viewer", &showSampleViewer);

                ImGui::SliderInt("Sample Index", &currentSampleIndex, 0, dataset.size() - 1);

                const DataSample& sample = dataset.getSample(currentSampleIndex);
                ImGui::Text("Label: %d", sample.label);

                // Display 28x28 
                static std::vector<float> imageData(784);
                for (int i = 0; i < 784; i++)
                    imageData[i] = sample.input[i];

                ImGui::Text("28x28 Image (ASCII representation):");
                for (int row = 0; row < 28; row++)
                {
                    std::string line = "";
                    for (int col = 0; col < 28; col++)
                    {
                        float pixel = imageData[row * 28 + col];
                        if (pixel > 0.7f) line += "O";
                        else if (pixel > 0.4f) line += "o";
                        else if (pixel > 0.1f) line += ".";
                        else line += "  ";
                    }
                    ImGui::Text("%s", line.c_str());
                }

                if (ImGui::Button("Next Sample"))
                    currentSampleIndex = (currentSampleIndex + 1) % dataset.size();

                ImGui::SameLine();
                if (ImGui::Button("Previous Sample"))
                    currentSampleIndex = (currentSampleIndex - 1 + dataset.size()) % dataset.size();

                ImGui::End(); 
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            #pragma endregion 

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}