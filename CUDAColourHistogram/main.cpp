#include "ColourHistogramGen.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <list>
#include <stdexcept>
#include <sail-c++/sail-c++.h>

namespace fs = std::filesystem;

// check if the extension of a path is a supported image file
// return true if file is supported
bool CheckSupportedExt(const fs::path& fpath)
{
	std::string ext = fpath.extension().string();

	if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
		ext == ".tif" || ext == ".tiff" || ext == ".bmp" ||
		ext == ".webp" || ext == ".avif")
	{
		return true;
	}
	else
	{
		return false;
	}
}

// function to generate the histogram image under the same directory of the file
// format of the histogram is hardcoded as png
void GenerateHistogram(chgen::ColourHistogramGen gen, const fs::path& fpath)
{
	/* open image */
	sail::image im(fpath.string());

	/* convert image to RGB888 */
	im.convert(SAIL_PIXEL_FORMAT_BPP24_RGB);

	/* copy the array of pixels of the image to an Image instance */
	unsigned int len = im.pixels_size();
	chgen::Image im_to_analyse =
	{
		im.width(),
		im.height(),
		(uint8_t*)im.pixels()
	};

	/* generate the histogram */
	gen.Analyse(im_to_analyse);
	std::unique_ptr<chgen::Image> hist = gen.GetHistogramImage();

	/* write the histogram */
	std::string output_path(fpath.string() + "_hist.png");
	sail::image_output out(output_path);
	sail::image outimg(hist->data, SAIL_PIXEL_FORMAT_BPP24_RGB, hist->width, hist->height, hist->width * 3);
	sail_status_t s = out.next_frame(outimg);
	if (s != SAIL_OK)
		throw std::runtime_error("Failed to add frame to png file");

	s = out.finish();
	if (s != SAIL_OK)
		throw std::runtime_error("Failed to write png file");

	std::cout << "Processed: " << fpath.string() << std::endl;
}

void GenerateStatsCsv(chgen::ColourHistogramGen gen, const fs::path& fpath)
{
	/* open image */
	sail::image im(fpath.string());

	/* convert image to RGB888 */
	im.convert(SAIL_PIXEL_FORMAT_BPP24_RGB);

	/* copy the array of pixels of the image to an Image instance */
	unsigned int len = im.pixels_size();
	chgen::Image im_to_analyse =
	{
		im.width(),
		im.height(),
		(uint8_t*)im.pixels()
	};

	/* generate the stats */
	gen.Analyse(im_to_analyse);
	std::unique_ptr<struct chgen::ColourStats> stats = gen.GetColourStats();

	std::ofstream csv_file(fpath.string() + "_stats.csv");

	for (uint32_t r : stats->r)
		csv_file << r << ", ";
	csv_file << std::endl;
	for (uint32_t g : stats->g)
		csv_file << g << ", ";
	csv_file << std::endl;
	for (uint32_t b : stats->b)
		csv_file << b << ", ";

	csv_file.close();

}

int main(int argc, char* argv[])
{
	if (chgen::CudaCount() < 1)
	{
		std::cout << "CUDA-capable device not found! Exiting" << std::endl;
		return 1;
	}

	fs::path path;
	bool histogram;

	switch (argc)
	{
	case 3:
		if (std::string(argv[1]) == "-g" || std::string(argv[1]) == "-s")
		{
			path = argv[2];
			histogram = (std::string(argv[1]) == "-g");

			break;
		}
	default:
		std::cout << "Usage: " << std::endl <<
			"CUDAColourHistogram -g <path> : Generate histogram" << std::endl <<
			"CUDAColourHistogram -s <path> : Generate colour statistics" << std::endl;
		return 0;
	}

	std::list<fs::path> imgfiles;

	/* search the specified path recursively, add path of supported file to list */
	for (const fs::directory_entry& dir_entry :
		fs::recursive_directory_iterator(path))
	{
		if (CheckSupportedExt(dir_entry.path()))
		{
			try 
			{
				fs::path canonical = fs::canonical(dir_entry.path());
				imgfiles.push_back(canonical);
			}
			catch (...)
			{
				continue;
			}
		}
	}

	/* process every image file in the list */
	try
	{
		chgen::ColourHistogramGen gen;
		if (histogram)
		{

			for (const fs::path& img : imgfiles)
				GenerateHistogram(gen, img);
		}
		else
		{
			for (const fs::path& img : imgfiles)
				GenerateStatsCsv(gen, img);
		}
	}
	catch (const std::exception& exc)
	{
		std::cerr << exc.what();
		return 1;
	}
	
	return 0;
}