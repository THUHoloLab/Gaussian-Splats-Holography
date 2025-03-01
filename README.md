# Gaussian Splatting Holography
This is the official implementation of Gaussian splatting holography (GSH), a groundbreaking paradigm of lensless hologracphic reconstruction methods. The GSH uses the 2D Gaussian splatting, a novel graphic primitive, for optical field representation, parameter compression, and twin image-free holographic reconstruction. With compact 2D Gaussian representation of a lightning fast CUDA-fused differentiable rendering, the GSH achieves high imaging performance for inline holography, enabling lensless imaging for both amplitude and quantitative phase patterns. 

## Quick start
### Requirements
* An NVIDIA GPU; All shown results come from an RTX 3090.
* MATLAB 2024a or above

### Test the code
* Clone this repository, and run "main_gsh.m" in MATLAB, the code begins reconstruction of a simulated samples.
* Different simulation results is available by tuning the parameters including wavelength, diffraction distance, and pixel size.
* run "rendering_single.m" if you want to test the Gaussian primitive for rendering of a single image.

### To build the codes
> ".mexw64" files are compiled cuda codes use "mexcuda" in MATLAB. 
> ".mexw64" works as a normal function that can be called by MATLAB. 

The CUDA source codes for GSH are released in "mxsplat/cuda/". If you want to change the CUDA codes and compile the to generate ".mexw64" files, you will need:
* CUDA v12.8
* Visual Studio 2022 Community
* Windows Kits 10.0.26100.0 (higher version may be available but not tested)

## Lightning fast Gaussian primitives
We use MATLAB + CUDA programming for the acceleration of 2D Gauassian's rendering speed, and makes the splatting fully differentiable. Rasterization of a total of 200000 Gaussians onto an image of 2k resolution generally takes about 20~30 ms when tested on a GPU of NVIDIA RTX 3090. 
