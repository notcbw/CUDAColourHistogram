#include "ColourHistogramGen.hpp"
#include <stdexcept>
#include <device_atomic_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <iostream>

namespace chgen
{
	// Counts the unique values of each colour channel in the supplied image buffer
	// The format of the image should be RGB888
	__global__ void kImageAnalysis(int size, uint8_t* in,
		uint32_t* r, uint32_t* g, uint32_t* b)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < size)
		{
			uint8_t val = in[idx];
			int c = idx % 3;

			/* increment the corresponding colour stats slot */
			switch (c)
			{
			case 0:
				atomicAdd(&r[val], 1);
			case 1:
				atomicAdd(&g[val], 1);
			case 2:
				atomicAdd(&b[val], 1);
			}
		}
	}

	// Plots the statistics of the colours to a image of 768x320
	// Output format is RGB888
	__global__ void kStatsToImage(uint32_t* r, uint32_t* g, uint32_t* b, uint32_t max, uint8_t* out)
	{
		int px = blockIdx.x * blockDim.x + threadIdx.x;
		int py = blockIdx.y * blockDim.y + threadIdx.y;

		if (px < 768 && py < 320)
		{
			uint32_t* in;
			int c = px % 3;
			switch (c)
			{
			case 0:
				in = r;
				break;
			case 1:
				in = g;
				break;
			case 2:
				in = b;
				break;
			}

			/* calculate the y value of the top pixel of the current bar */
			float freq = static_cast<float>(in[px / 3]) / static_cast<float>(max);
			int top_y = static_cast<int>(320.0f - 320.0f * freq);

			/* draw */
			int index = (py * 768 + px) * 3;
			if (py >= top_y)
			{

				switch (c)
				{
				case 0:
					out[index] = 0xff;
					out[index + 1] = 0x00;
					out[index + 2] = 0x00;
					break;
				case 1:
					out[index] = 0x00;
					out[index + 1] = 0xff;
					out[index + 2] = 0x00;
					break;
				case 2:
					out[index] = 0x00;
					out[index + 1] = 0x00;
					out[index + 2] = 0xff;
					break;
				}
			}
			else
			{
				out[index] = 0x00;
				out[index + 1] = 0x00;
				out[index + 2] = 0x00;
			}
		}
	}
}

int chgen::CudaCount()
{
	int c = 0;
	cudaGetDeviceCount(&c);
	return c;
}

chgen::ColourHistogramGen::ColourHistogramGen()
{
	/* allocate gpu memory */
	cudaError_t err;
	err = cudaMalloc(&gpu_stats_r, 256 * sizeof(uint32_t));
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to allocate device memory");

	err = cudaMalloc(&gpu_stats_g, 256 * sizeof(uint32_t));
	if (err != cudaSuccess)
	{
		cudaFree(gpu_stats_r);
		throw std::runtime_error("Failed to allocate device memory");
	}

	err = cudaMalloc(&gpu_stats_b, 256 * sizeof(uint32_t));
	if (err != cudaSuccess)
	{
		cudaFree(gpu_stats_r);
		cudaFree(gpu_stats_g);
		throw std::runtime_error("Failed to allocate device memory");
	}

	/* zero the gpu memory */
	cudaMemset(gpu_stats_r, 0, 256 * sizeof(uint32_t));
	cudaMemset(gpu_stats_g, 0, 256 * sizeof(uint32_t));
	cudaMemset(gpu_stats_b, 0, 256 * sizeof(uint32_t));
}

chgen::ColourHistogramGen::~ColourHistogramGen()
{
	/* deallocate gpu memory */
	cudaFree(gpu_stats_r);
	cudaFree(gpu_stats_g);
	cudaFree(gpu_stats_b);
	if (hist_buf != nullptr)
		free(hist_buf);
}

void chgen::ColourHistogramGen::Analyse(chgen::Image& im)
{
	int size = im.width * im.height * 3;
	
	/* allocate gpu image buffer */
	uint8_t* gpu_imgbuf;
	cudaError_t err = cudaMalloc(&gpu_imgbuf, size);
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to allocate device image buffer");

	/* copy image data to device */
	err = cudaMemcpy(gpu_imgbuf, im.data, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to copy image to CUDA device buffer");

	/* call cuda kernel to do the statistics*/
	int block_size = 256;
	int grid_size = (size + block_size - 1) / block_size;

	kImageAnalysis<<<grid_size, block_size>>>(size, gpu_imgbuf, gpu_stats_r, gpu_stats_g, gpu_stats_b);
	cudaDeviceSynchronize();

	/* free gpu image buffer */
	cudaFree(gpu_imgbuf);
}

std::unique_ptr<struct chgen::ColourStats> chgen::ColourHistogramGen::GetColourStats()
{
	std::unique_ptr<struct ColourStats> cs = std::make_unique<struct ColourStats>();
	
	/* copy stats from gpu to cs heap */
	cudaError_t err;
	err = cudaMemcpy(cs->r, gpu_stats_r, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to copy data from CUDA device");

	err = cudaMemcpy(cs->g, gpu_stats_g, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to copy data from CUDA device");

	err = cudaMemcpy(cs->b, gpu_stats_b, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to copy data from CUDA device");

	return cs;
}

std::unique_ptr<chgen::Image> chgen::ColourHistogramGen::GetHistogramImage()
{
	uint8_t* im_gpu;
	cudaError_t err = cudaMalloc(&im_gpu, 768 * 320 * 3 * sizeof(uint8_t));
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to allocate device histogram buffer");

	/* calculate the max value of the statistics arrays */
	thrust::device_ptr<uint32_t> th_r(gpu_stats_r);
	uint32_t max_r = *thrust::max_element(th_r, th_r + 255);

	thrust::device_ptr<uint32_t> th_g(gpu_stats_g);
	uint32_t max_g = *thrust::max_element(th_g, th_g + 255);

	thrust::device_ptr<uint32_t> th_b(gpu_stats_b);
	uint32_t max_b = *thrust::max_element(th_b, th_b + 255);

	uint32_t max = (max_g > max_r) ? max_g : max_r;
	max = (max > max_b) ? max : max_b;

	/* call kernel to generate histogram */
	dim3 block_dim(32, 32);
	dim3 grid_dim(24, 10);
	kStatsToImage<<<grid_dim, block_dim>>>(gpu_stats_r, gpu_stats_g, gpu_stats_b, max, im_gpu);

	/* allocate memory */
	hist_buf = (uint8_t*)malloc(768 * 320 * 3 * sizeof(uint8_t));

	/* copy image data */
	err = cudaMemcpy(hist_buf, im_gpu, 768 * 320 * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		throw std::runtime_error("Failed to copy histogram image data");

	cudaFree(im_gpu);

	std::unique_ptr<Image> im_ptr = std::make_unique<Image>();
	im_ptr->width = 768;
	im_ptr->height = 320;
	im_ptr->data = hist_buf;

	return im_ptr;
}
