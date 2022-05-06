#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

const int PARTICLE_N = 8192;
const int GRID_SIZE = 32;
const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
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

void atomic_add_g_f3(volatile global float3 *addr, float3 val) {
    union float_cvt {
        float3 f3;
        float f1[3];
    };
    volatile global float *cvt_addr = (volatile global float *) addr;
    atomic_add_g_f(cvt_addr, val.x);
    atomic_add_g_f(cvt_addr + 1, val.y);
    atomic_add_g_f(cvt_addr + 2, val.z);
}

int coord2index(int3 coord) {
    int result = coord.z * GRID_SIZE_SQ + coord.y * GRID_SIZE + coord.x;
    return result;
}

typedef struct {
    float3 r1;
    float3 r2;
    float3 r3;
} Mat3;

Mat3 mat3_add(Mat3 a, Mat3 b) {
    return (Mat3){.r1 = a.r1 + b.r1, .r2 = a.r2 + b.r2, .r3 = a.r3 + b.r3};
}

Mat3 mat3_mul(float scalar, Mat3 mat) {
    return (Mat3){.r1 = mat.r1 * scalar, .r2 = mat.r2 * scalar, .r3 = mat.r3 * scalar};
}

Mat3 make_id_mat(float diagonal) {
    return (Mat3){.r1 = (float3)(diagonal, 0., 0.), .r2 = (float3)(0., diagonal, 0.), .r3 = (float3)(0., 0., diagonal)};
}

float3 mat3x3_mul_float3(Mat3 mat, float3 vec) {
    return (float3)(dot(mat.r1, vec), dot(mat.r2, vec), dot(mat.r3, vec));
}

kernel void particle2grid(
    const float time_delta, // time step length
    global const float3 *position, // particle property
    global float3 *velocity, // particle property
    global Mat3 *Cmat, // particle property; 3x3 mat in row major
    global float *J, // particle property
    global float3 *grid_v, // grid velocity
    global float *grid_m // grid mass
    ) {
    size_t pid = get_global_linear_id();
    float3 grid_coord = position[pid] / GRID_SPAN;
    float3 fx, bx; // fraction and integer part of the position
    bx = grid_coord - 0.5f;
    int3 coord = convert_int3(bx);
    fx = grid_coord - convert_float3(coord);
    float3 weights[3] = {0.5f * pown(1.5f - fx, 2), 0.75f - pown(fx - 1.f, 2), 0.5f * pown(fx - 0.5f, 2)};
    float stress = -time_delta * 4 * E * P_VOL * (J[pid] - 1.f) * GRID_SPAN_INVSQ;
    Mat3 affine = mat3_add(make_id_mat(stress), mat3_mul(P_MASS, Cmat[pid]));
    // scatter to grid
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                int3 offset = (int3)(i, j, k);
                float3 pos_diff = (convert_float3(offset) - fx) * GRID_SPAN;
                float weight = weights[i].x * weights[j].y * weights[k].z;
                int linear_coord = coord2index(coord + offset);
                float3 APIC_term = mat3x3_mul_float3(affine, pos_diff);
                float3 v_incr = P_MASS * velocity[pid] + APIC_term;
                atomic_add_g_f3(&grid_v[linear_coord], weight * v_incr);
                atomic_add_g_f(&grid_m[linear_coord], weight * P_MASS);
            }
        }
    }
}
