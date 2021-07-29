__kernel void matrix_mac(__global const float* d_a, __global const float* d_b, __global float* d_c, int N, int P) {

    // Get the index of the current element to be processed
    int z = get_global_id(0);
    float sum=0;
        int i = z / P;
        int j = z % P;
    for (int k = 0; k < N; k++) {

        sum += d_a[i * N + k] * d_b[k * P + j];
    }
    // Do the operation
   d_c[z]+=sum;
}