#version 330

#define POSITION 0

layout(location = POSITION) in vec3 in_Position;

uniform mat4 mat_MVP;

out vec3 out_Position;


void main() {
    gl_Position = mat_MVP * vec4(in_Position, 1.);
    out_Position = in_Position;
}