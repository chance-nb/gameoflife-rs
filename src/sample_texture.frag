in vec2 uv_frag;

uniform sampler2D conway;

out vec4 FragColor;

void main() {
    float alive = texture(conway, uv_frag).r;
    FragColor = vec4(1.0 * alive, 0.0 * alive, 0.0 * alive, 1.0);
    // FragColor = texture(conway, uv_frag);
}
