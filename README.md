# Neural Network Trainer

## Overview
A C++ neural network trainer with OpenGL-based GUI for creating, training, and testing neural networks. Currently only supports MNIST handwritten digit recognition.

![Screenshot 2025-05-31 220656](https://github.com/user-attachments/assets/9e2e8f7f-4149-458e-8d5a-70d8efe3ef59)


## Features
- **Flexible Neural Network Architecture**
- **MNIST Dataset Support** - Load and train on MNIST CSV format datasets (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Sample Viewer**
- **Drawing Canvas** - Hand-draw digits for testing
- **Training Metrics Dashboard** 
- **Batch Training**

## Dependencies
The project requires the following libraries, all included in the `Dependencies` folder or are included with header files from the src/vendor folder:
- [GLFW 3.4](https://www.glfw.org/)
- [GLEW 2.1.0](http://glew.sourceforge.net/)
- GLM (src/vendor)
- ImGui (src/vendor)
- ImPlot (src/vendor)
- Eigen (src/vendor)

## Installation & Setup

### 1. Configure Project Properties
Before building, ensure the following settings are correctly configured:

- **Include Directories:**
  - `$(SolutionDir)Dependencies\glfw-3.4.bin.WIN32\include`
  - `$(SolutionDir)Dependencies\glew-2.1.0\include`
  - `src\vendor`

- **Library Directories:**
  - `$(SolutionDir)Dependencies\glfw-3.4.bin.WIN32\lib-vc2022`
  - `$(SolutionDir)Dependencies\glew-2.1.0\lib\Release\Win32`

- **Additional Dependencies:**
  - `glfw3.lib`
  - `glew32s.lib`
  - `opengl32.lib`
  - `user32.lib`
  - `gdi32.lib`
  - `shell32.lib`

- **Preprocessor Definitions:**
  - `GLEW_STATIC`

### 2. Build & Run
**Current Supported Configuration**:
- **Platform**: Windows (x86)
- **OpenGL**: 4.3 compatible graphics card required

## Usage

### Dataset Loading
1. Prepare MNIST dataset in CSV format (label in first column, 784 pixel values following)
2. Enter dataset path in the application
3. Set maximum samples for testing
4. Click "Load MNIST Dataset"

### Network Creation
1. Configure network architecture (layers and nodes per layer)
2. Click "Create Network"

### Training
1. Set learning rate, batch size, and epochs
2. Click "Start Training" to begin

### Testing
- **Sample Viewer**: Browse dataset samples and see network predictions
- **Drawing Canvas**: Draw digits with mouse and get predictions
  - Left click/drag to draw
  - Right click/drag to erase
  - Adjust brush size with slider

## Dataset Format

**MNIST CSV Format Expected:**
- First column: digit label (0-9)
- Following 784 columns: pixel values (0-255, automatically normalized)

## Known Issues & Limitations

- **Dataset Support**: Currently limited to MNIST format only
- **Platform**: Primarily tested on Windows
- **Memory Usage**: Large datasets may require significant RAM
- **Model Persistence**: No save/load functionality for trained networks

## Contribution
This project is open for contributions. Feel free to submit pull requests to improve performance, fix issues, or add features.

## License
This project is licensed under the [MIT License](LICENSE).
