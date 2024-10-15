## Setup for training Hugging Face Transformers

### Stack
```Full stack
jupyter
python 3.12
pyTorch 1.5
torch 
CUDA 12.4 and CUDA Toolkit 12.4
Linux x86-64 openSUSE Leap 15
```

### Compatability matrix for PyTorch, Python and CUDA
|PyTorch version | Python	Stable CUDA
2.5	>=3.9, <=3.12, (3.13 experimental)	CUDA 11.8, CUDA 12.1, CUDA 12.4, CUDNN 9.1.0.70
2.4	>=3.8, <=3.12	CUDA 11.8, CUDA 12.1, CUDNN 9.1.0.70
### CUDA 12.4
CUDA 12.4 works with NVIDIA GPUs from the G8x series onwards, including GeForce, Quadro, and the Tesla line. For more info, 
- see CUDA-capable GPUs at developer.nvidia.com/cuda-gpus
- check release notes for the CUDA Toolkit for a list of supported products

#### CUDA Toolkit 12.4
To run a CUDA application, your system needs:
- A CUDA enabled GPU
- An NVIDIA display driver that's compatible with the CUDA Toolkit used to build the application
- The right version of any libraries the application relies on for dynamic linking

### Tested on:  OS, Software and Hardware

#### Operating System: openSUSE Leap 15.5
#### Software
- KDE Plasma Version: 5.27.9
* KDE Frameworks Version: 5.103.0
+ Qt Version: 5.15.8
- Kernel Version: 5.14.21-150500.55.83-default (64-bit)

#### Hardware=
Graphics Platform: X11
Processors: 8 × Intel® Core™ i7-4790 CPU @ 3.60GHz
Memory: 15.3 GiB of RAM
Graphics Processor: Mesa Intel® HD Graphics 4600
Manufacturer: Hewlett-Packard
Product Name: HP Z230 Tower Workstation

## Installing CUDA Toolkit 12.4
Download Installer for your OS (e.g. Linux x86_64 SLES 15)
The base installer is available for download at https://developer.nvidia.com/cuda-12-4-0-download-archive

### Base Installer	
Installation Instructions:
```
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-sles15-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
sudo rpm -i cuda-repo-sles15-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
sudo zypper refresh
sudo zypper install -y cuda-toolkit-12-4
```

Additional installation options are detailed [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#switching-between-driver-module-flavors)

Driver Installer	
NVIDIA Driver Instructions (choose one option)
To install the legacy kernel module flavor:
sudo zypper install -y cuda-drivers
To install the open kernel module flavor:
sudo zypper install -y nvidia-open-driver-G06-kmp-default
sudo zypper install -y cuda-drivers
To switch between NVIDIA Driver kernel module flavors see here.

The CUDA Toolkit contains Open-Source Software. The source code can be found here.
The checksums for the installer and patches can be found in Installer Checksums.
For further information, see the Installation Guide for Linux and the CUDA Quick Start Guide.

```

