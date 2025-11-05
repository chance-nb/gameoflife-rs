in vec2 uv_frag;

uniform sampler2D conway;
uniform uvec2 texture_size;

out vec4 FragColor;

void main() {
    float threshold = 0.8;
    vec2 current_pos = uv_frag * texture_size;

    // Sample the texture with the transformed UVs
    vec4 sampled_color = texture(conway, uv_frag);
    vec4 color;

    bool is_alive = sampled_color.r > threshold;

    int alive_neighbours = 0;
    for (int x_offset = -1; x_offset <= 1; x_offset++) {
        // Loop over Y offsets from -1 to 1
        for (int y_offset = -1; y_offset <= 1; y_offset++) {
            vec2 neighbour_pos = current_pos + vec2(x_offset, y_offset);
            if (texture(conway, neighbour_pos / texture_size).r > threshold) {
                alive_neighbours += 1;
            }
        }
    }
    if (is_alive) {
        alive_neighbours -= 1;
    }

    if ((is_alive && alive_neighbours == 2) || (alive_neighbours == 3)) {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    }

    FragColor = color;
}
