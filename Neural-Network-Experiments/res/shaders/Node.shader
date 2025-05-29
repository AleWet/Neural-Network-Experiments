#shader vertex
#version 430 core

layout(location = 0) in vec2 position;      // Quad vertex position
layout(location = 1) in vec2 texCoord;      // Texture coordinates
layout(location = 2) in vec2 nodePos;       // Node world position (instanced)
layout(location = 3) in float activation;   // Node activation level (instanced)
layout(location = 4) in float size;         // Node size (instanced)

uniform mat4 u_MVP;

out vec2 v_TexCoord;
out float v_Activation;
out vec2 v_LocalPos;

void main()
{
    // Scale the quad by node size and translate to node position
    vec2 scaledPos = position * size + nodePos;
    
    gl_Position = u_MVP * vec4(scaledPos, 0.0, 1.0);
    
    v_TexCoord = texCoord;
    v_Activation = activation;
    v_LocalPos = position; // Local position for circle calculation
}

#shader fragment
#version 430 core

in vec2 v_TexCoord;
in float v_Activation;
in vec2 v_LocalPos;

uniform int u_ShowActivations;

out vec4 color;

void main()
{
    // Calculate distance from center for circular shape
    float dist = length(v_LocalPos);
    
    // Discard pixels outside the circle
    if (dist > 1.0)
        discard;
    
    // Create a smooth edge for the circle
    float alpha = 1.0 - smoothstep(0.9, 1.0, dist);
    
    vec3 baseColor;
    if (u_ShowActivations == 1)
    {
        // Color based on activation: blue (low) to red (high)
        baseColor = mix(vec3(0.2, 0.4, 1.0), vec3(1.0, 0.3, 0.2), v_Activation);
        
        // Add some brightness based on activation
        baseColor *= (0.5 + 0.5 * v_Activation);
    }
    else
    {
        // Default neutral color
        baseColor = vec3(0.7, 0.7, 0.7);
    }
    
    // Add a subtle border
    float border = smoothstep(0.7, 0.8, dist);
    baseColor = mix(baseColor, vec3(0.9, 0.9, 0.9), border * 0.3);
    
    color = vec4(baseColor, alpha);
}