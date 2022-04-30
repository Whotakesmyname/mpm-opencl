
const int PARTICLE_N = 8192;
const int GRID_SIZE = 128;
const float GRID_SPAN = 1.f / GRID_SIZE;
const float GRID_SPAN_INVSQ = GRID_SIZE * GRID_SIZE;
const float TIME_DELTA = 2e-4f;

const float P_RHO = 1;
const float P_VOL = 0.25f * GRID_SPAN * GRID_SPAN;
const float P_MASS = P_VOL * P_RHO;
const float G = 9.8f;
const float BOUND = 3;
const float E = 400;



int coord2index(int2 coord) {
    return coord.y * GRID_SIZE + coord.x;
}


kernel void grid2particle(
    global float2 *position, // particle property
    global float2 *next_position,
    global float2 *velocity, // particle property
    global float4 *Cmat, // particle property; 2x2 mat in row major
    global float *J, // particle property
    global float2 *grid_v, // grid velocity
    global float *grid_m // grid mass
) {
    size_t pid = get_global_linear_id();
    float2 grid_coord = position[pid] / GRID_SPAN;
    float2 fx, bx;
    // fx = fract(grid_coord, &bx);
    bx = grid_coord - 0.5f;
    int2 coord = convert_int2(bx);
    fx = grid_coord - convert_float2(coord);
    float2 weights[3] = {0.5f * pown(1.5f - fx, 2), 0.75f - pown(fx - 1.f, 2), 0.5f * pown(fx - 0.5f, 2)};

    float2 next_v = (float2)(0.f);
    float4 next_C = (float4)(0.f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int2 offset = (int2)(i, j);
            float2 pos_diff = (convert_float2(offset) - fx) * GRID_SPAN;
            float weight = weights[i].x * weights[j].y;
            int linear_coord = coord2index(coord + offset);
            float2 grid_velocity = grid_v[linear_coord];
            // printf("gcord: %v2d\n", coord + offset);
            next_v += weight * grid_velocity;
            next_C += 4.f * weight * (float4)(grid_velocity*pos_diff.x, grid_velocity*pos_diff.y) * GRID_SPAN_INVSQ;
        }
    }
    velocity[pid] = next_v;
    next_position[pid] = /* clamp( */position[pid] + next_v * TIME_DELTA/* , 0.f, 1.f) */;
    // printf("v: %v2f, p: %v2f\n", next_v, next_position[pid]);
    // position[pid] += next_v * TIME_DELTA;
    J[pid] *= 1.f + TIME_DELTA * (next_C.x + next_C.w);
    Cmat[pid] = next_C;
}