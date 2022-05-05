#version 330

in vec3 out_Position;

out vec4 FragColor;

void main() {
    FragColor = vec4(out_Position, 1.);
}