in vec3 position;
in vec2 uv;
uniform vec3 zoom;
out vec2 uv_frag;

void main() {
    gl_Position = vec4((position.x+zoom.x)*zoom.z, ( position.y + zoom.y )*zoom.z, 0, 1.0);
    uv_frag = uv;
}
