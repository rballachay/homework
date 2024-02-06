#version 330

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

in vec3 in_position; 
in vec3 in_normal;

out vec3 frag_normal;
out vec3 frag_position;

void main() {
    vec4 view_position = V * M * vec4(in_position, 1.0);
    frag_position = view_position.xyz;
    frag_normal = normalize(mat3(transpose(inverse(M))) * in_normal);
    gl_Position = P * view_position;
}
