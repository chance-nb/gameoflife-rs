in vec3 position;
in vec2 uv;
uniform uvec2 texture_size;
out vec2 uv_frag;

void main() {
    gl_Position = vec4(position, 1.0);
    uv_frag = uv;
}
