#version 330 core

uniform bool enable_lighting;
uniform vec3 light_position; 
uniform vec3 diffuse_color; 
uniform float ambient_intensity; 

in vec3 frag_normal;
in vec3 frag_position;

out vec4 frag_color;

void main() {

    if (!enable_lighting) {
        frag_color = vec4(diffuse_color, 1.0);
        return;
    }

    vec3 ambient = ambient_intensity * diffuse_color;

    vec3 light_dir = normalize(light_position - frag_position);
    float distance = length(light_position - frag_position);
    
    float diffuse_intensity = max(dot(frag_normal, light_dir), 0.0);
    vec3 diffuse = diffuse_color * diffuse_intensity;

    vec3 view_dir = normalize(-frag_position);
    vec3 half_vector = normalize(light_dir + view_dir);

    float specular_intensity = pow(max(dot(frag_normal, half_vector), 0.0), 32.0);
    vec3 specular = vec3(diffuse_color) * specular_intensity;

    vec3 final_color = ambient + diffuse + specular;

    frag_color = vec4(final_color, 1.0);
}
