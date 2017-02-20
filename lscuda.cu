#include <iostream>
#include <sstream>
#include <iomanip>

const int width = 34;

static void cudaSafeCall(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
}

static int getCoreNumPerSP(int major, int minor)
{
	switch (major)
	{
		case 2:
		{
			if (minor == 0)
			{
				return 32;
			}
			else if (minor == 1)
			{
				return 48;
			}
			else
			{
				return 0;
			}
		}
		case 3:
		{
			return 192;
		}
		case 5:
		{
			return 128;
		}
		case 6:
		{
			if (minor == 0)
			{
				return 64;
			}
			else if ((minor == 1) || (minor == 2))
			{
				return 128;
			}
			else
			{
				return 0;
			}
		}
		default:
		{
			return 0;
		}
	}
}

static std::string formatSize(size_t size)
{
	std::ostringstream stringStream;

	stringStream.precision(1);
	stringStream << std::fixed;

	if (size < 1024 * 10)
	{
		stringStream << size;
	}
	else if (size < 1024 * 1024 * 10)
	{
		stringStream << (float)size / 1024 << " KiB";
	}
	else if (size < size_t(1024) * size_t(1024) * size_t(1024) * size_t(10))
	{
		stringStream << (float)size / (1024 * 1024) << " MiB";
	}
	else
	{
		stringStream << (float)size / (1024 * 1024 * 1024) << " GiB";
	}
	return stringStream.str();
}

static void displayDeviceProperties(cudaDeviceProp& prop, int device)
{
	std::cout << std::setw(width) << std::left << "GPU Device ID:" << device << std::endl;
	std::cout << std::setw(width) << std::left << "  Name:" << prop.name << std::endl;
	std::cout << std::setw(width) << std::left << "  Compute Capability:" << prop.major << "." << prop.minor << std::endl;
	std::cout << std::setw(width) << std::left << "  MultiProcessor(s):" << prop.multiProcessorCount << std::endl;
	std::cout << std::setw(width) << std::left << "  Cores Per MultiProcessor:" << getCoreNumPerSP(prop.major, prop.minor) << std::endl;
	std::cout << std::setw(width) << std::left << "  Max Threads Per MultiProcessor:" << prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << std::setw(width) << std::left << "  Clock Rate:" << prop.clockRate / (1000) << " MHz" << std::endl;
	std::cout << std::setw(width) << std::left << "  Warp Size:" << prop.warpSize << std::endl;
	std::cout << std::setw(width) << std::left << "  L2 Cache Size:" << formatSize(prop.l2CacheSize) << std::endl;
	std::cout << std::setw(width) << std::left << "  Global Memory Size:" << formatSize(prop.totalGlobalMem) << std::endl;
	std::cout << std::setw(width) << std::left << "  Constant Memory Size:" << formatSize(prop.totalConstMem) << std::endl;
	std::cout << std::setw(width) << std::left << "  One-Dimension Texture Size:" << prop.maxTexture1D << std::endl;
	std::cout << std::setw(width) << std::left << "  Two-Dimension Texture Size:" << prop.maxTexture2D[0] << " x "
			<< prop.maxTexture2D[1] << std::endl;
	std::cout << std::setw(width) << std::left << "  Three-Dimension Texture Size:" << prop.maxTexture3D[0] << " x "
			<< prop.maxTexture3D[1] << " x " << prop.maxTexture3D[2] << std::endl;
	std::cout << std::setw(width) << std::left << "  Shared Memory Size Per Block:" << formatSize(prop.sharedMemPerBlock) << std::endl;
	std::cout << std::setw(width) << std::left << "  Max Threads Per Block:" << prop.maxThreadsPerBlock << std::endl;
	std::cout << std::setw(width) << std::left << "  Registers Per Block:" << prop.regsPerBlock << std::endl;
	std::cout << std::setw(width) << std::left << "  Block Dimension:" << prop.maxThreadsDim[0] << " x "
			<< prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
	std::cout << std::setw(width) << std::left << "  Grid Dimension:" << prop.maxGridSize[0] << " x "
			<< prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;

	return;
}

int main(void)
{
	int gpuCount = 0;
	cudaSafeCall(cudaGetDeviceCount(&gpuCount));

	int runtimeVersion = 0;
	cudaSafeCall(cudaRuntimeGetVersion(&runtimeVersion));

	int driverVersion = 0;
	cudaSafeCall(cudaDriverGetVersion(&driverVersion));

	cudaDeviceProp *prop = new cudaDeviceProp[gpuCount];
	if (prop == NULL)
	{
		std::cout << "The memory is too small, and please enlarge it, thanks!" << std::endl;
		exit(1);
	}

	for (int i = 0; i < gpuCount; i++)
	{
		cudaSafeCall(cudaGetDeviceProperties(prop + i, i));
	}

	std::cout << std::setw(width) << std::left << "CUDA Runtime Version:" << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
	std::cout << std::setw(width) << std::left << "CUDA Driver Version:" << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
	std::cout << std::setw(width) << std::left << "GPU(s):" << gpuCount << std::endl;

	for (int i = 0; i < gpuCount; i++)
	{
		displayDeviceProperties(prop[i], i);
	}

	delete[] prop;
	return 0;
}


