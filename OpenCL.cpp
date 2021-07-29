
#include <iostream>
#include <fstream>
#include <string>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <time.h>
#define MAX_SOURCE_SIZE (0xA00000)


void oclPrintDevName(cl_device_id device)
{
	char device_string[1024];
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
	printf("%s \n", device_string);
}


int main()
{
	//Data preperation and Matrix Filling
#pragma region
	// Creating and Filling the Array

	int M = 16;
	int N = 10;
	int P = 14;
	float* h_a;    // We uses this reference variable to access
	float* h_b;   // dynamically created array elements
	float* h_c;
	float* h_d_s;
	float* h_d_p;

	h_a = (float*)calloc(M * N, sizeof(float));  // Make double array of size elements
	h_b = (float*)calloc(N * P, sizeof(float));  // Make double array of size elements
	h_c = (float*)calloc(M * P, sizeof(float));  // Make double array of size elements
	h_d_s = (float*)calloc(M * P, sizeof(float));  // Make double array of size elements
	h_d_p = (float*)calloc(M * P, sizeof(float));  // Make double array of size elements

	// Fill the array with 1 
	for (int i = 0; i < M * N; i++) {
		h_a[i] = 1.0f;
	}
	for (int i = 0; i < N * P; i++) {
		h_b[i] = 1.0f;
	}
	for (int i = 0; i < M * P; i++) {
		h_c[i] = 0.0f;
	}
	for (int i = 0; i < M * P; i++) {
		h_d_s[i] = 0.0f;
		h_d_p[i] = 0.0f;
	}


#pragma endregion

	//serial excution

#pragma region
	clock_t s_start = clock();
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < P; j++) {
			h_d_s[i * P + j] = h_c[i * P + j];
			for (int k = 0; k < N; k++) {
				h_d_s[i * P + j] += h_a[i * N + k] * h_b[k * P + j];
			}
		}
	}

	clock_t s_end = clock();
#pragma endregion

	//parallel execution
#pragma region
	clock_t p_start = clock();
	// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;
	// Read the program source
	std::ifstream sourceFile("kernel.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	source_size = sourceCode.length();
	char *new_char = &sourceCode[0];

	// Get platform and device information
	cl_platform_id platform_id = NULL;

	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(3, &platform_id, &ret_num_platforms);

	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	
	oclPrintDevName(device_id);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each matrix 
	cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, M * N * sizeof(float), NULL, &ret);
	cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, N * P * sizeof(float), NULL, &ret);
	cl_mem d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, M * P * sizeof(float), NULL, &ret);

	// Copy data to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, M * N * sizeof(float), h_a, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, N * P * sizeof(float), h_b, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, d_c, CL_TRUE, 0, M * P * sizeof(float), h_c, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&new_char, (const size_t*)&source_size, &ret);

	// Build the program
	size_t len = 0;
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	char* buffer = (char*)malloc(source_size);
	ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
	//

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "matrix_mac", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_a);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_b);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_c);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&N);
	ret = clSetKernelArg(kernel, 4, sizeof(int), (void*)&P);

	// Execute the OpenCL kernel on the list
	int temp_size = M * P ;
	size_t global_item_size = temp_size ; // Process the entire lists
	size_t local_item_size = 2; // Divide work items into groups

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable D
	ret = clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, M * P * sizeof(float), h_d_p, 0, NULL, NULL);

	clock_t p_end = clock();

#pragma endregion

	//Data Validation
	for (int i = 0; i < M * P; i++) {
		if (h_d_p[i] != h_d_s[i]) {
			std::cout << "Invalid Result " << h_d_p[i] << "   " << h_d_s[i] << "  coordinates = "<< i/P<<"   "<<i%P<<"\n";
		}
	}

	double serial_t = 0.0;
	double parallel_t = 0.0;

	//serial_t += (double)(s_end - s_start) / CLOCKS_PER_SEC;
	//parallel_t += (double)(p_end - p_start) / CLOCKS_PER_SEC;

	printf("Serial Time elpased is %f seconds\n", serial_t);
	printf("Parallel Time elpased is %f seconds", parallel_t);

}

