#pragma once

#include <memory>

namespace chgen
{
	struct ColourStats
	{
		uint32_t r[256];
		uint32_t g[256];
		uint32_t b[256];
	};

	typedef struct Image
	{
		unsigned int width, height;
		uint8_t* data;
	} Image;

	class ColourHistogramGen
	{
	public:
		ColourHistogramGen();
		~ColourHistogramGen();

		// Analyse image, store the statistics of the colour channels in GPU buffer
		void Analyse(chgen::Image& im);

		// Returns the statistics of the colour channels of the last analysis
		std::unique_ptr<struct chgen::ColourStats> GetColourStats();

		// Returns the colour histogram image of the last analysis
		std::unique_ptr<chgen::Image> GetHistogramImage();

	private:
		uint32_t* gpu_stats_r;
		uint32_t* gpu_stats_g;
		uint32_t* gpu_stats_b;
		uint8_t* hist_buf = nullptr;
	};
}