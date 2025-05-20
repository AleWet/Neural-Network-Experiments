#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include "Renderer.h"
#include "core/Time.h"

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "VertexBufferLayout.h"
#include "Shader.h"
#include "Texture.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "vendor/imgui/imgui.h"
#include "vendor/imgui/imgui_impl_glfw.h"
#include "vendor/imgui/imgui_impl_opengl3.h"

const float fixedDeltaTime = 1.0f / 60.0f;

// Will be in Utils.h
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

        // Initialize STUFF
        std::string Basic = "res/shaders/Basic.shader";
        if (!IsShaderPathOk(Basic)) return 0;
        Shader BasicShader(Basic);
        // MISSING RENDERER

        Time timeManager(fixedDeltaTime);
        int FPScounter = 0;

        #pragma endregion


        int numberOfNodes = 0;
        int numberOfLayers = 0;
        int numberOfInputNodes = 0;
        int numberOfOutputNodes = 0;

        // Main loop
        while (!glfwWindowShouldClose(window))
        {
            #pragma region Rendering / ImGui / Metrics

            // Pre-Rendering 
            GLCall(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
            GLCall(glClear(GL_COLOR_BUFFER_BIT));


            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();


            ImGui::Begin("Settings");   
            ImGui::InputScalar("Number of Nodes", ImGuiDataType_U32, &numberOfNodes, NULL, NULL, "%u");
            ImGui::InputScalar("Number of Layers", ImGuiDataType_U32, &numberOfLayers, NULL, NULL, "%u");
            ImGui::InputScalar("Number of Input Nodes", ImGuiDataType_U32, &numberOfInputNodes, NULL, NULL, "%u");
            ImGui::InputScalar("Number of Output Nodes", ImGuiDataType_U32, &numberOfOutputNodes, NULL, NULL, "%u");
            ImGui::Button("Select activation function");
            ImGui::Button("Start Learning");

            ImGui::End();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


            // Display fps and mspf


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