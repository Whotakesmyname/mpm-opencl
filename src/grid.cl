
const int PARTICLE_N = 8192;
const int GRID_SIZE = 128;
const float GRID_SPAN = 1.f / GRID_SIZE;

const float P_RHO = 1;
const float P_VOL = 0.25f * GRID_SPAN * GRID_SPAN;
const float P_MASS = P_VOL * P_RHO;
const float G = 9.8f;
const float BOUND = 3;
const float E = 400;


int coord2index(int2 coord) {
    return coord.y * GRID_SIZE + coord.x;
}


kernel void grid_operation(
    const float time_delta,
    global float2 *grid_v,
    global float *grid_m
) {
    size_t linear_index = get_global_linear_id();
    float mass = grid_m[linear_index];
    float2 velocity = grid_v[linear_index];
    if (mass > 0) {
        velocity /= mass;
    }
    // gravity
    velocity.y -= time_delta * G;
    // boundary condition
    size_t x = get_global_id(0), y = get_global_id(1);
    if (x < BOUND && velocity.x < 0) {
        velocity.x = 0;
    }
    if (x > GRID_SIZE - BOUND && velocity.x > 0) {
        velocity.x = 0;
    }
    if (y < BOUND && velocity.y < 0) {
        velocity.y = 0;
    }
    if (y > GRID_SIZE - BOUND && velocity.y > 0) {
        velocity.y = 0;
    }
    grid_v[linear_index] = velocity;
}

