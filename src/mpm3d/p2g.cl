#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

const int PARTICLE_N = 8192;
const int GRID_SIZE = 128;
const float GRID_SPAN = 1.f / GRID_SIZE;
const float GRID_SPAN_INVSQ = GRID_SIZE * GRID_SIZE;

const float P_RHO = 1;
const float P_VOL = 0.25f * GRID_SPAN * GRID_SPAN;
const float P_MASS = P_VOL * P_RHO;
const float E = 400;

/** Atomic float add
See https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/ 
*/
void atomic_add_g_f(volatile global float *addr, float val) {
    union {
    unsigned int u32;
    float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
    expected.f32 = current.f32;
    next.f32 = expected.f32 + val;
    current.u32 = atomic_cmpxchg((volatile global unsigned int *)addr,
                                    expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

void atomic_add_g_f2(volatile global float2 *addr, float2 val) {
    union {
        unsigned long u64;
        float2 f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        expected.f64 = current.f64;
        next.f64 = expected.f64 + val;
        current.u64 = atom_cmpxchg((volatile global unsigned long *)addr,
                                    expected.u64, next.u64);
    } while (current.u64 != expected.u64);
}

int coord2index(int2 coord) {
    int result = coord.y * GRID_SIZE + coord.x;
    return result;
}

float2 mat2x2_mul_float2(float4 mat, float2 vec) {
    return (float2)(dot(mat.xy, vec), dot(mat.zw, vec));
}

kernel void particle2grid(
    const float time_delta, // time step length
    global const float2 *position, // particle property
    global float2 *velocity, // particle property
    global float4 *Cmat, // particle property; 2x2 mat in row major
    global float *J, // particle property
    global float2 *grid_v, // grid velocity
    global float *grid_m // grid mass
    ) {
    size_t pid = get_global_linear_id();
    float2 grid_coord = position[pid] / GRID_SPAN;
    float2 fx, bx; // fraction and integer part of the position
    // fx = fract(grid_coord, &bx);
    bx = grid_coord - 0.5f;
    int2 coord = convert_int2(bx);
    fx = grid_coord - convert_float2(coord);
    // if (coord.x < 0 || coord.x >= 128 || coord.y < 0 || coord.y >=128)
    //     printf("wrong coord: %d, %d, from %f, %f\n", coord.x, coord.y, grid_coord.x, grid_coord.y);
    // printf("coord: %d, %d, from %f, %f\n", coord.x, coord.y, grid_coord.x, grid_coord.y);
    float2 weights[3] = {0.5f * pown(1.5f - fx, 2), 0.75f - pown(fx - 1.f, 2), 0.5f * pown(fx - 0.5f, 2)};
    float stress = -time_delta * 4 * E * P_VOL * (J[pid] - 1.f) * GRID_SPAN_INVSQ;
    float4 affine = (float4)(stress, 0.f, 0.f, stress) + P_MASS * Cmat[pid];
    // printf("stress: %f, affine: %v4f\n", stress, affine);
    // scatter to grid
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int2 offset = (int2)(i, j);
            float2 pos_diff = (convert_float2(offset) - fx) * GRID_SPAN;
            float weight = weights[i].x * weights[j].y;
            int linear_coord = coord2index(coord + offset);
            float2 APIC_term = mat2x2_mul_float2(affine, pos_diff);
            float2 v_incr = P_MASS * velocity[pid] + APIC_term;
            atomic_add_g_f2(&grid_v[linear_coord], weight * v_incr);
            atomic_add_g_f(&grid_m[linear_coord], weight * P_MASS);
            // printf("gcoord: %v2d, APIC: %v2f\n", coord + offset, APIC_term);
            // grid_v[linear_coord] += weight * (P_MASS * velocity[pid] + mat2x2_mul_float2(affine, pos_diff));
            // grid_m[linear_coord] += weight * P_MASS;
        }
    }
}
