#shader vertex
#version 430 core

layout(location = 0) in vec2 position;
layout(location = 1) in float size;
layout(location = 2) in vec4 color;
layout(location = 3) in float activation;

uniform mat4 u_MVP;

out vec4 v_Color;
out float v_Activation;
out vec2 v_TexCoord;

void main()
{
    // Basic quad vertices (centered at origin)
    vec2 vertices[4] = vec2[4](
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.5)
    );
    
    // Texture coordinates for the quad
    vec2 texCoords[4] = vec2[4](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0)
    );
    
    // Pass vertex data to fragment shader
    v_Color = color;
    v_Activation = activation;
    v_TexCoord = texCoords[gl_VertexID];
    
    // Calculate final position
    vec4 pos = u_MVP * vec4(vertices[gl_VertexID], 0.0, 1.0);
    gl_Position = pos;
}

#shader fragment
#version 430 core

layout(location = 0) out vec4 color;

in vec4 v_Color;
in float v_Activation;
in vec2 v_TexCoord;

uniform float u_Size;

void main()
{
    // Calculate distance from center for circular node
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(v_TexCoord, center);
    
    // Set alpha based on distance (creates circle)
    float alpha = 1.0;
    if (dist > 0.5) {
        discard; // Outside circle
    } else if (dist > 0.45) {
        // Smooth edge
        alpha = 1.0 - smoothstep(0.45, 0.5, dist);
    }
    
    // Calculate inner circle size based on activation
    float innerRadius = 0.35 * v_Activation;
    
    // Base color for the node
    vec4 baseColor = v_Color;
    
    // If inside the inner circle, brighten based on activation
    if (dist < innerRadius) {
        // Brighter center based on activation
        float brightness = mix(1.0, 2.0, v_Activation);
        baseColor.rgb *= brightness;
    }
    
    // Apply glow effect based on activation
    float glowStrength = v_Activation * 0.5;
    float glow = smoothstep(innerRadius - 0.1, innerRadius, dist) * 
                 (1.0 - smoothstep(innerRadius, innerRadius + 0.2, dist));
    
    vec3 glowColor = mix(baseColor.rgb, vec3(1.0), 0.7);
    baseColor.rgb = mix(baseColor.rgb, glowColor, glow * glowStrength);
    
    // Apply alpha for edge smoothing
    baseColor.a *= alpha;
    
    // Output final color
    color = baseColor;
}