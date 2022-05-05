
const int PARTICLE_N = 8192;
const int GRID_SIZE = 32;
const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
const float GRID_SPAN = 1.f / GRID_SIZE;
const float GRID_SPAN_INVSQ = GRID_SIZE * GRID_SIZE;

const float P_RHO = 1;
const float P_VOL = 0.25f * GRID_SPAN * GRID_SPAN;
const float P_MASS = P_VOL * P_RHO;
const float G = 9.8f;
const float E = 400;


typedef struct {
    float3 r1;
    float3 r2;
    float3 r3;
} Mat3;

Mat3 make_outer_product(float3 a, float3 b) {
    Mat3 result = {a * b.x, a * b.y, a * b.z};
    return result;
}

Mat3 mat3_add(Mat3 a, Mat3 b) {
    return (Mat3){.r1 = a.r1 + b.r1, .r2 = a.r2 + b.r2, .r3 = a.r3 + b.r3};
}

Mat3 mat3_mul(float scalar, Mat3 mat) {
    return (Mat3){.r1 = mat.r1 * scalar, .r2 = mat.r2 * scalar, .r3 = mat.r3 * scalar};
}

float mat3_trace(Mat3 mat) {
    return mat.r1.x + mat.r2.y + mat.r3.z;
}

int coord2index(int3 coord) {
    return coord.z * GRID_SIZE_SQ + coord.y * GRID_SIZE + coord.x;
}


kernel void grid2particle(
    const float time_delta,
    global const float3 *position, // particle property
    global float3 *next_position,
    global float3 *velocity, // particle property
    global Mat3 *Cmat, // particle property; 2x2 mat in row major
    global float *J, // particle property
    global float3 *grid_v, // grid velocity
    global float *grid_m // grid mass
) {
    size_t pid = get_global_linear_id();
    float3 grid_coord = position[pid] / GRID_SPAN;
    float3 fx, bx;
    // fx = fract(grid_coord, &bx);
    bx = grid_coord - 0.5f;
    int3 coord = convert_int3(bx);
    fx = grid_coord - convert_float3(coord);
    float3 weights[3] = {0.5f * pown(1.5f - fx, 2), 0.75f - pown(fx - 1.f, 2), 0.5f * pown(fx - 0.5f, 2)};

    float3 next_v = (float3)(0.f);
    Mat3 next_C = {0., 0., 0.};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                int3 offset = (int3)(i, j, k);
                float3 pos_diff = (convert_float3(offset) - fx) * GRID_SPAN;
                float weight = weights[i].x * weights[j].y * weights[k].z;
                int linear_coord = coord2index(coord + offset);
                float3 grid_velocity = grid_v[linear_coord];
                // printf("gcord: %v3d\n", coord + offset);
                next_v += weight * grid_velocity;
                // printf("v: %v3f\n", grid_velocity);
                next_C = mat3_add(next_C, mat3_mul(4.f * weight * GRID_SPAN_INVSQ, make_outer_product(grid_velocity, pos_diff)));
            }
        }
    }
    next_position[pid] = /* clamp( */position[pid] + next_v * time_delta/* , 0.1f, 0.9f) */;
    // printf("pv: %v3f\n, pp: %v3f\n, v: %v3f\n, p: %v3f\n", velocity[pid], position[pid], next_v, next_position[pid]);
    velocity[pid] = next_v;
    // position[pid] += next_v * time_delta;
    J[pid] *= 1.f + time_delta * mat3_trace(next_C);
    Cmat[pid] = next_C;
}