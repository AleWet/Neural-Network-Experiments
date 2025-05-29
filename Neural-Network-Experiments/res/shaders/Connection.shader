#shader vertex
#version 430 core

layout(location = 0) in vec2 position;        // Line vertex position (0,0) or (1,0)
layout(location = 1) in vec2 startPos;        // Connection start position (instanced)
layout(location = 2) in vec2 endPos;          // Connection end position (instanced)
layout(location = 3) in float weight;         // Connection weight (instanced)
layout(location = 4) in float thickness;      // Connection thickness (instanced)

uniform mat4 u_MVP;

out float v_Weight;
out float v_Alpha;

void main()
{
    // Interpolate between start and end position based on vertex position
    vec2 worldPos = mix(startPos, endPos, position.x);
    
    gl_Position = u_MVP * vec4(worldPos, 0.0, 1.0);
    
    v_Weight = weight;
    
    // Calculate alpha based on weight magnitude (stronger connections more visible)
    float weightMagnitude = abs(weight);
    v_Alpha = min(1.0, weightMagnitude * 2.0 + 0.3); // Ensure minimum visibility
}

#shader fragment
#version 430 core

in float v_Weight;
in float v_Alpha;

out vec4 color;

void main()
{
    vec3 baseColor;
    
    // Color based on weight: red for negative, blue for positive
    if (v_Weight > 0.0)
    {
        baseColor = vec3(0.2, 0.6, 1.0); // Blue for positive weights
    }
    else
    {
        baseColor = vec3(1.0, 0.3, 0.2); // Red for negative weights
    }
    
    // Intensity based on weight magnitude
    float intensity = min(1.0, abs(v_Weight) + 0.3);
    baseColor *= intensity;
    
    color = vec4(baseColor, v_Alpha);
}