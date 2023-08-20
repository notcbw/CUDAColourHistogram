# CUDAColourHistogram

This tool takes advantage of the Nvidia CUDA platform to generate colour histograms quickly for a specified batch of images. 

Alternatively, the statistics of the colour channels of the images can be exported to a CSV file.

Sail is used as the image decoding library so that it supports multiple image formats.

(Written as an exercise for CUDA)

## Requirements for Compiling

- Visual Studio 2022
- CUDA Toolkit 12.1
- Sail 0.9.0-rc3 [\(https://github.com/HappySeaFox/sail/tree/master\)](https://github.com/HappySeaFox/sail/tree/master)

## Running the Program

### Generating colour histograms for a folder

```
CUDAColourHistogram -g <path to folder>
```

### Generating colour statistics csv for a folder

```
CUDAColourHistogram -s <path to folder>
```

## License

MIT License

Copyright (c) 2023 Bowen Cui

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
