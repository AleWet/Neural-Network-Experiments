#shader vertex
#version 430 core

layout(location = 0) in vec2 startPos;
layout(location = 1) in vec2 endPos;
layout(location = 2) in float weight;
layout(location = 3) in vec3 color;

uniform mat4 u_MVP;

out vec3 v_Color;
out float v_Weight;
out float v_Progress;

void main()
{
    // Pass data to fragment shader
    v_Color = color;
    v_Weight = weight;
    
    // Determine if this is the start or end point of the line
    v_Progress = (gl_VertexID % 2 == 0) ? 0.0 : 1.0;
    
    // Choose position based on vertex ID
    vec2 position = (gl_VertexID % 2 == 0) ? startPos : endPos;
    
    // Apply MVP transformation
    gl_Position = u_MVP * vec4(position, 0.0, 1.0);
}

#shader fragment
#version 430 core

layout(location = 0) out vec4 color;

in vec3 v_Color;
in float v_Weight;
in float v_Progress;

void main()
{
    // Calculate line intensity based on weight
    float intensity = clamp(abs(v_Weight) * 0.7 + 0.3, 0.1, 1.0);
    
    // Add some animation effect (pulsing) based on weight
    float pulse = (sin(v_Progress * 3.14159 * 2.0) * 0.5 + 0.5) * abs(v_Weight) * 0.2;
    
    // Adjust alpha for visual effect
    float alpha = clamp(intensity + pulse, 0.1, 0.9);
    
    // Add gradient along the connection
    vec3 finalColor = mix(v_Color * 0.7, v_Color, v_Progress);
    
    // Output final color
    color = vec4(finalColor, alpha);
}