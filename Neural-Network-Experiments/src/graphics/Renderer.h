#pragma once
#include <GL/glew.h>

#include "IndexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"

#define ASSERT(x) if (!(x)) __debugbreak();
#define GLCall(x) GLClearError();\
	x;\
	ASSERT(GLLogCall(#x, __FILE__, __LINE__ ))

void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);

class Renderer
{
private:

public:
	void Clear() const;
	void Draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader) const; //this is goind to draw the entire input IndexBuffer 
};