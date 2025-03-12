
# Bachelor's Final Project: Scanpaths in 360º videos

Juan Lorente Guarnieri

## Resume
Bachelor's Final Project of a Computer Engineering student majoring in computing at the University of Zaragoza.
This project consists of generate realistic scanpaths in 360º videos using saliency maps with SST-Sal by The Graphics and Imaging Lab.

Visit their [website](https://graphics.unizar.es/projects/SST-Sal_2022/)

## Requirements
The code has been tested with:

```
matplotlib==3.3.4 
numba==0.53.1 
numpy==1.20.1
opencv_python==4.5.4.58 
Pillow==9.1.1 
fastdtw==0.3.4
editdistance==0.8.1
requests==2.31.0
scipy==1.6.2 
torch==1.5.1+cu92 
torchvision==0.6.1+cu92 tqdm
```
These torch and torchvision version are not provided by the pip package so you can download [here](https://pytorch.org/get-started/previous-versions/)
## Usage

Import the video for which you want to generate the scanpath to the following path:

```
├── data
    ├── (directory name)
        ├── original
        ├── saliency
```
In the 'original' folder, the frames of the original video will be saved as '.png' files. Similarly, in the 'saliency' folder, the saliency maps for each image from '/original' will be placed such that the image is in grayscale, with brighter areas indicating higher saliency values, I recommend using [SST-Sal](https://github.com/edurnebernal/SST-Sal/) for creating the saliency maps.


To execute the program, launch the following command:

```
python main.py 
```
Additionally, after running the program, an '/output' folder will be created containing the results as either frames or as a video, depending on the settings in config.py.