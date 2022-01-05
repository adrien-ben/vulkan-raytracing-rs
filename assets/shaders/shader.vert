#version 450

layout(location = 0) in vec2 vPosition;
layout(location = 1) in vec3 vColor;

layout(location = 0) out vec3 oColor;

void main() {
    oColor = vColor;

    gl_Position = vec4(vPosition.x, vPosition.y, 0.0, 1.0);
}
