#version 330

#define POSITION 0

layout(location = POSITION) in vec2 in_Position;

uniform mat4 mat_MVP;


void main() {
    gl_Position = mat_MVP * vec4(in_Position, 0., 1.);
}