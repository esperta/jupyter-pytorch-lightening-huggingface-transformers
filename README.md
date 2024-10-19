## Setup for training Hugging Face Transformers on a Linux box with an NVIDIA GPU.

### Current setup
```
jupyter
Python 3.12
PyTorch 2.4
CUDA 12.4 and CUDA Toolkit 12.4
[GPU NVIDIA Tesla V100 SXM2 16GB](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)  
Linux x86-64 openSUSE Leap 15
```
A comment on NVIDIA model names - Tesla is the name for data center accelerators and the letter before the model number is the architecture.
For instance, my setup uses data center accelerator (a Tesla accelerator) from the Volta architecture (v) with model number 100 and using a SXM2 socket.


## GPU drivers and their installation
GPUs have evolving architectures and capabilities.  To help characterize the computing power of each GPU model, it has a compute capability(cc). For AI training, the GPU needs to have compute capability at least 3.0. So, even if your GPU is CUDA-enabled, you need to double-check if the architecture supports AI training.

To fully understand GPU installations, see [NVIDIA guide to datacenter drivers](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Datacenter_Drivers.pdf). Though it is datacenter guide, it has a wealth of information, much of which applies to installations on a single node.  Actual installation on Linux is covered int the [guide for NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#) This guide also provides step-by-step instructions and faqs for driver installation on Linux.  

Tor those using Tesla, the NVIDIA white-paper on Tesla is found [here](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).


### CUDA and CUDA Toolkits
CUDA Toolkit 12.4 works with NVIDIA GPUs from the G8x series onwards, including GeForce, Quadro, and the Tesla line. 

For other GPUs, the compute capability of the NVIDIA GPUs determine the CUDA Toolkit to use.  Each version of CUDA toolkit has a minimum GPU compute capability that it supports. 
To determine the CUDA Toolkit version:
1. Find the compute capability of your GPU using this [list from NVIDIA](https://developer.nvidia.com/cuda-gpus)
2. Use the compute capacity to find the Toolkit version here.


#### GPUs for CUDA Toolkit 12.4


#### CUDA Toolkit 12.4
The CUDA Toolkit contains
- NVIDIA display drivers (CUDA drivers) for Linux and Windows
- CUDA C++ Core Libraries
- [NSight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) profiling and analysis for CUDA kernels
- [NSight Systems](https://developer.nvidia.com/nsight-systems/get-started) performance tuning; profiles hardware metrics and CUDA apps, APIs, and libraries 
- support for GCC 13 as a host-side compilers
NVIDIA overview [here](https://developer.nvidia.com/blog/cuda-toolkit-12-4-enhances-support-for-nvidia-grace-hopper-and-confidential-computing/).

#### CUDA 12.4
See [NVIDIA CUDA technical doc](https://docs.nvidia.com/cuda/doc/index.html).
#### Installing CUDA Toolkit 12.4 on Linux x86_64 SLES 15
Installer for all OS's supported is found [here](https://developer.nvidia.com/cuda-12-4-0-download-archive).  The same site provides installation instructions.

### Compatability matrix for PyTorch, Python and CUDA
| PyTorch  |               Python	              |                   Stable CUDA                   |
|:--------:|:----------------------------------:|:-----------------------------------------------:|
|   2.5	   | >=3.9, <=3.12, (3.13 experimental) |	CUDA 11.8, CUDA 12.1, CUDA 12.4, CUDNN 9.1.0.70 |
|   2.4	   |         >=3.8, <=3.12	            |     CUDA 11.8, CUDA 12.1, CUDNN 9.1.0.70        |



### Test system

#### Operating System: openSUSE Leap 15.5

#### Software
- KDE Plasma Version: 5.27.9
- KDE Frameworks Version: 5.103.0
- Qt Version: 5.15.8
- Kernel Version: 5.14.21-150500.55.83-default (64-bit)

#### Hardware
- Graphics Platform: X11
- Processors: 8 × Intel® Core™ i7-4790 CPU @ 3.60GHz
- Memory: 15.3 GiB of RAM
- Graphics Processor: Mesa Intel® HD Graphics 4600
- Manufacturer: Hewlett-Packard
- Product Name: HP Z230 Tower Workstation



### Base Installer for Linux x86_64 SLES 15	
Installation Instructions:
```
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-sles15-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
sudo rpm -i cuda-repo-sles15-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
sudo zypper refresh
sudo zypper install -y cuda-toolkit-12-4
```

Additional installation options are detailed [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages))

### Driver Installer for Linux x86_64 SLES 15		
NVIDIA Driver Instructions (choose one option)

To install the legacy kernel module flavor:
```
sudo zypper install -y cuda-drivers
```

To install the open kernel module flavor:
```
sudo zypper install -y nvidia-open-driver-G06-kmp-default
sudo zypper install -y cuda-drivers
```

To switch between NVIDIA Driver kernel module flavors see [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#switching-between-driver-module-flavors).

## Using a Tesla v100 GPU on a traditional workstation or server 
A single server with Tesla V100 GPUs can replace hundreds of commodity CPU-only servers for both traditional HPC and AI workloads. 
Every researcher and engineer can now afford an AI supercomputer to tackle their most challenging work.


### Final notes on the toolkit
- The CUDA Toolkit contains Open-Source Software. The source code can be found [here](https://developer.download.nvidia.com/compute/cuda/opensource/12.4.0).
- The checksums for the installer and patches can be found in [Installer Checksums](https://developer.download.nvidia.com/compute/cuda/12.4.0/docs/sidebar/md5sum.txt).
- For further information, see the [Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and the [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

```

