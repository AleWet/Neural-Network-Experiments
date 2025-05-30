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
#include "vendor/imgui/implot.h"
#include "vendor/imgui/implot_internal.h"


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
    ImPlot::CreateContext();
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
        bool showTrainingSampleViewer = true; // DEBUGGING, will be false and have a separate setter in the UI
        int selectedDatasetType = 0;          // 0=MNIST for now only this
        float learningRate = 0.01f;

        int numberOfLayers = 4;               // Total layers including input and output
        std::vector<int> layerSizes(4);       // Vector to store nodes for each layer
        layerSizes[0] = 784;                  // Default input size for MNIST
        layerSizes[1] = 128;                  // Default first hidden layer
        layerSizes[2] = 64;                   // Default second hidden layer
        layerSizes[3] = 10;                   // Default output size for MNIST

        // Training parameters
        static int batchSize = 32;
        static int epochs = 10;
        static bool isTraining = false;
        static int currentEpoch = 0;
        static float currentLoss = 0.0f;
        static float currentAccuracy = 0.0f;

        // Metrics
        std::vector<float> lossHistory;
        std::vector<float> accuracyHistory;
        std::vector<float> epochNumbers;
        bool showMetricsWindow = false;
        int maxHistorySize = 1000; // Limit history to prevent memory issues

        // Sample rendering
        unsigned int mnistTexture = 0;
        bool textureNeedsUpdate = true;
        int lastDisplayedSample = -1;

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

            #pragma region MAIN WINDOW

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
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No dataset loaded");
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
                ImGui::SliderFloat("Learning Rate", &learningRate, 0.001f, 5.0f);

                ImGui::InputInt("Batch Size", &batchSize);
                if (batchSize < 1) batchSize = 1;
                if (batchSize > maxSamples) batchSize = maxSamples;

                ImGui::InputInt("Epochs", &epochs);
                if (epochs < 1) epochs = 1;

                // Show Metrics Checkbox
                if (datasetLoaded && networkCreated) {
                    ImGui::Checkbox("Show Training Metrics", &showMetricsWindow);

                    if (ImGui::Button("Clear Metrics History")) {
                        lossHistory.clear();
                        accuracyHistory.clear();
                        epochNumbers.clear();
                    }
                }

                if (datasetLoaded && networkCreated && !isTraining)
                {
                    if (ImGui::Button("Start Training"))
                    {
                        isTraining = true;
                        currentEpoch = 0;
                        std::cout << "Starting training with " << epochs << " epochs, batch size " << batchSize << std::endl;
                    }
                }
                else if (isTraining)
                {
                    // TO BE REVISED IN THE FUTURE -> make it train on the first x% of the samples and then present new samples that it has never seen before
                    if (currentEpoch < epochs)
                    {
                        dataset.shuffle();

                        // Batches training
                        int numBatches = (dataset.size() + batchSize - 1) / batchSize;
                        float epochLoss = 0.0f;
                        float epochAccuracy = 0.0f;

                        for (int batch = 0; batch < numBatches; batch++)
                        {
                            auto batchData = dataset.getBatch(batchSize);
                            network.TrainBatch(batchData, learningRate);

                            epochLoss += network.CalculateAverageLoss(batchData);
                            epochAccuracy += network.CalculateAccuracy(batchData);
                        }

                        currentLoss = epochLoss / numBatches;
                        currentAccuracy = epochAccuracy / numBatches;
                        currentEpoch++;

                        UpdateTrainingMetrics(currentEpoch, currentLoss, currentAccuracy,
                            lossHistory, accuracyHistory, epochNumbers, maxHistorySize);

                        std::cout << "Epoch " << currentEpoch << "/" << epochs
                            << " - Loss: " << currentLoss
                            << ", Accuracy: " << (currentAccuracy * 100.0f) << "%" << std::endl;

                        if (currentEpoch >= epochs)
                        {
                            isTraining = false;
                            std::cout << "Training completed!" << std::endl;
                        }
                    }

                    if (ImGui::Button("Stop Training"))
                    {
                        isTraining = false;
                        std::cout << "Training stopped by user." << std::endl;
                    }
                }

                if (datasetLoaded && networkCreated)
                {
                    
                    // ADD BUTTON FOR USER INPUT (CANVAS)

                    if (ImGui::Button("Shuffle Dataset"))
                    {
                        dataset.shuffle();
                        std::cout << "Dataset shuffled." << std::endl;
                    }

                    // Display current metrics
                    ImGui::Separator();
                    ImGui::Text("Training Progress:");
                    if (isTraining)
                    {
                        ImGui::Text("Epoch: %d/%d", currentEpoch, epochs);
                        ImGui::ProgressBar(static_cast<float>(currentEpoch) / static_cast<float>(epochs));
                    }
                    ImGui::Text("Current Loss: %.6f", currentLoss);
                    ImGui::Text("Current Accuracy: %.2f%%", currentAccuracy * 100.0f);
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

            #pragma endregion

            #pragma region SECONDARY WINDOWS

            // Sample Viewer (separate window) I FORGOT WHY IT'S RED I HAVE TO FIND OUT AGAIN
            if (showSampleViewer && datasetLoaded && selectedDatasetType == 0) 
            {
                ImGui::Begin("MNIST Sample Viewer", &showSampleViewer);
                ImGui::SliderInt("Sample Index", &currentSampleIndex, 0, dataset.size() - 1);

                const DataSample& sample = dataset.getSample(currentSampleIndex);
                ImGui::Text("Label: %d", sample.label);

                // Update texture if sample changed
                if (currentSampleIndex != lastDisplayedSample) 
                {
                    std::vector<float> imageData(784);
                    for (int i = 0; i < 784; i++) 
                        imageData[i] = sample.input[i];

                    UpdateMNISTTexture(mnistTexture, imageData);
                    lastDisplayedSample = currentSampleIndex;
                }

                // Display image
                if (mnistTexture != 0) 
                {
                    // Calculate display size 
                    float imageScale = 8.0f; // Adjust this value to change size
                    ImVec2 imageSize(28 * imageScale, 28 * imageScale);

                    ImGui::Text("MNIST Image (28x28 pixels):");
                    ImGui::Image((ImTextureID)mnistTexture, imageSize);
                }

                // Network prediction display 
                if (networkCreated && !isTraining) 
                {
                    ImGui::Separator();
                    ImGui::Text("Network Prediction:");

                    // Predicted class
                    auto prediction = network.Forward(sample.input);
                    int predictedClass = 0;
                    float maxProb = prediction[0];
                    for (int i = 1; i < prediction.size(); i++) 
                    {
                        if (prediction[i] > maxProb) 
                        {
                            maxProb = prediction[i];
                            predictedClass = i;
                        }
                    }

                    // Display 
                    ImVec4 predictionColor = (predictedClass == sample.label) ?
                        ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1); // Green if correct, red if wrong

                    ImGui::TextColored(predictionColor, "Predicted: %d (%.2f%% confidence)",
                        predictedClass, maxProb * 100.0f);

                    if (predictedClass == sample.label) {
                        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Correct");
                    }
                    else {
                        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Incorrect (actual: %d)", sample.label);
                    }

                    // Show all class probabilities as a bar chart
                    ImGui::Text("Class Probabilities:");
                    for (int i = 0; i < 10; i++) 
                    {
                        float prob = prediction[i];
                        ImVec4 barColor = (i == sample.label) ? ImVec4(0, 1, 0, 0.7f) : ImVec4(0.5f, 0.5f, 0.5f, 0.7f);

                        ImGui::Text("%d:", i);
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
                        ImGui::ProgressBar(prob, ImVec2(-1, 0), "");
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::Text("%.3f", prob);
                    }
                }

                // Buttons Nav
                if (ImGui::Button("Previous Sample")) 
                    currentSampleIndex = (currentSampleIndex - 1 + dataset.size()) % dataset.size();
               
                ImGui::SameLine();
                if (ImGui::Button("Next Sample")) 
                    currentSampleIndex = (currentSampleIndex + 1) % dataset.size();
                
                ImGui::SameLine();
                if (ImGui::Button("Random Sample")) 
                    currentSampleIndex = rand() % dataset.size();
                
                ImGui::End();
            }

            // Training batch viewer (separate window) TBD
            if (showTrainingSampleViewer && datasetLoaded && selectedDatasetType == 0) // Only for MNIST
            {
                //ImGui::Begin("MNIST Training Sample Viewer", &showSampleViewer);

                //// RENDER CURRENT SAMPLE BATCH AND LET USER CHOOSE ONE TRAINING SAMPLE TO TEST THE NETWORK 

                //ImGui::End();
            }

            // Training Metrics Window WILL BE REVISED IN THE FUTURE JUST A TEST FOR THE MOMENT
            if (showMetricsWindow && (!lossHistory.empty() || !accuracyHistory.empty())) {
                ImGui::Begin("Training Metrics", &showMetricsWindow);

                // Styling
                ImPlotStyle& style = ImPlot::GetStyle();
                style.LineWeight = 2.0f;
                style.MarkerSize = 4.0f;
                style.ErrorBarSize = 5.0f;
                style.ErrorBarWeight = 1.5f;
                style.DigitalBitHeight = 8.0f;
                style.DigitalBitGap = 4.0f;

                ImVec4 lossColor = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);    // Red for loss
                ImVec4 accuracyColor = ImVec4(0.4f, 1.0f, 0.4f, 1.0f); // Green for accuracy

                // Check data before plotting
                if (!lossHistory.empty() && !epochNumbers.empty()) 
                {
                    // Loss Plot
                    if (ImPlot::BeginPlot("Training Loss", ImVec2(-1, 250)))
                    {
                        ImPlot::SetupAxes("Epoch", "Loss");

                        // X
                        float maxEpoch = epochNumbers.empty() ? 1.0f : *std::max_element(epochNumbers.begin(), epochNumbers.end());
                        ImPlot::SetupAxisLimits(ImAxis_X1, 0, maxEpoch * 1.05f, ImGuiCond_Always);

                        // Y
                        float minLoss = *std::min_element(lossHistory.begin(), lossHistory.end());
                        float maxLoss = *std::max_element(lossHistory.begin(), lossHistory.end());
                        float lossRange = maxLoss - minLoss;
                        float lowerBound = std::max(0.0f, minLoss - lossRange * 0.1f);
                        ImPlot::SetupAxisLimits(ImAxis_Y1, lowerBound, maxLoss + lossRange * 0.1f, ImGuiCond_Always);

                        // Color
                        ImPlot::SetNextLineStyle(lossColor);
                        ImPlot::PlotLine("Loss", epochNumbers.data(), lossHistory.data(),
                            static_cast<int>(lossHistory.size()));

                        // Add markers for recent points
                        if (lossHistory.size() > 0) 
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, lossColor, 1, lossColor);
                            int markerStart = std::max(0, static_cast<int>(lossHistory.size()) - 5);
                            ImPlot::PlotScatter("Recent",
                                epochNumbers.data() + markerStart,
                                lossHistory.data() + markerStart,
                                static_cast<int>(lossHistory.size()) - markerStart);
                        }
                        ImPlot::EndPlot();
                    }
                }
                else 
                {
                    ImGui::TextColored(ImVec4(1, 0.7f, 0, 1), "No loss data available yet.");
                }

                if (!accuracyHistory.empty() && !epochNumbers.empty()) 
                {
                    // Accuracy Plot
                    if (ImPlot::BeginPlot("Training Accuracy", ImVec2(-1, 250))) 
                    {
                        // Set up axes
                        ImPlot::SetupAxes("Epoch", "Accuracy");

                        // Configure X-axis to start from 0
                        float maxEpoch = epochNumbers.empty() ? 1.0f : *std::max_element(epochNumbers.begin(), epochNumbers.end());
                        ImPlot::SetupAxisLimits(ImAxis_X1, 0, maxEpoch * 1.05f, ImGuiCond_Always);

                        // Configure Y-axis for accuracy (0 to 1 or slightly above max)
                        float maxAccuracy = *std::max_element(accuracyHistory.begin(), accuracyHistory.end());
                        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, std::max(1.0f, maxAccuracy * 1.05f), ImGuiCond_Always);

                        // Plot the line with custom color
                        ImPlot::SetNextLineStyle(accuracyColor);
                        ImPlot::PlotLine("Accuracy", epochNumbers.data(), accuracyHistory.data(),
                            static_cast<int>(accuracyHistory.size()));

                        // Add markers for recent points
                        if (accuracyHistory.size() > 0) 
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, accuracyColor, 1, accuracyColor);
                            // Show marker only for the last few points
                            int markerStart = std::max(0, static_cast<int>(accuracyHistory.size()) - 5);
                            ImPlot::PlotScatter("Recent",
                                epochNumbers.data() + markerStart,
                                accuracyHistory.data() + markerStart,
                                static_cast<int>(accuracyHistory.size()) - markerStart);
                        }

                        ImPlot::EndPlot();
                    }
                }
                else
                {
                    ImGui::TextColored(ImVec4(1, 0.7f, 0, 1), "No accuracy data available yet.");
                }
                if (!lossHistory.empty() && !accuracyHistory.empty()) 
                {
                    ImGui::Separator();
                    ImGui::Text("Training Statistics");

                    // Table for statistics
                    if (ImGui::BeginTable("StatsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) 
                    {
                        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 150);
                        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        // LOSS
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("Current Loss");
                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextColored(lossColor, "%.6f", lossHistory.back());

                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("Lowest Loss");
                        ImGui::TableSetColumnIndex(1);
                        float lowestLoss = *std::min_element(lossHistory.begin(), lossHistory.end());
                        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "%.6f", lowestLoss);

                        // ACCURACY
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("Current Accuracy");
                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextColored(accuracyColor, "%.2f%%", accuracyHistory.back() * 100.0f);

                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("Best Accuracy");
                        ImGui::TableSetColumnIndex(1);
                        float bestAccuracy = *std::max_element(accuracyHistory.begin(), accuracyHistory.end());
                        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "%.2f%%", bestAccuracy * 100.0f);

                        // Progress info
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);

                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("Training Progress");
                        ImGui::TableSetColumnIndex(1);
                        if (isTraining) 
                        {
                            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.2f, 1.0f), "Training... (%d/%d epochs)", currentEpoch, epochs);
                        }
                        else 
                        {
                            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Idle");
                        }

                        ImGui::EndTable();
                    }

                    // Performance Indicators
                    if (lossHistory.size() > 1) 
                    {
                        ImGui::Separator();
                        ImGui::Text("Recent Trends:");

                        // Loss 
                        float recentLossChange = lossHistory.back() - lossHistory[lossHistory.size() - 2];
                        if (recentLossChange < 0) {
                            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "Loss: (%.6f)", recentLossChange);
                        }
                        else 
                        {
                            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Loss: (+%.6f)", recentLossChange);
                        }

                        // Accuracy trend
                        float recentAccuracyChange = accuracyHistory.back() - accuracyHistory[accuracyHistory.size() - 2];
                        if (recentAccuracyChange > 0) {
                            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "Accuracy: (+%.2f%%)", recentAccuracyChange * 100.0f);
                        }
                        else 
                        {
                            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Accuracy: (%.2f%%)", recentAccuracyChange * 100.0f);
                        }
                    }
                }

                ImGui::End();
            }

            #pragma endregion
            
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            #pragma endregion 

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        if (mnistTexture != 0)
            glDeleteTextures(1, &mnistTexture);
    }

    // Cleanup
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}