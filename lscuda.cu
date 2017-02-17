#include <iostream>
#include <iomanip>

const int width = 25;

static void cudaSafeCall(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
}

int main(void)
{
	int gpuCount = 0;

	cudaSafeCall(cudaGetDeviceCount(&gpuCount));

	std::cout << std::setw(width) << std::left << "GPU(s):" << gpuCount << std::endl;
	return 0;
}


