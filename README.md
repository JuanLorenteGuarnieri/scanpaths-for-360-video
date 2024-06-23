
# Bachelor's Final Project: Study and evaluation of scanpath prediction methods for 360º video

**Author**: Juan Lorente Guarnieri

**Directors**: Daniel Martín Serrano, Edurne Bernal Berdún

**Presenter**: Ana Belén Serrano Pacheu
## Resume

In recent years, virtual reality has gained significant importance in areas such as gaming, education, and medicine. To create immersive and realistic experiences, understanding human attention is crucial: What captures observers' attention? Which regions are most interesting? Predicting visual attention patterns can help content creators optimize their designs by identifying areas of high interest.

This project focuses on analyzing and comparing different methods for predicting scanpaths in 360º videos, which represent the trajectory of an observer's gaze over time. These scanpaths provide insights into which visual elements capture user attention and how they are processed over time.

Initially, seven heuristic methods based on saliency map sampling were implemented. Then, a deep learning prediction model was designed and implemented, based on two state-of-the-art models, to study its potential in predicting visual trajectories. Various established metrics were used to evaluate the methods.

Results suggest that advanced saliency sampling methods, such as probabilistic sampling and inhibition of return, more accurately replicate human visual exploration patterns compared to basic methods. The deep learning model achieved results comparable to heuristic methods, indicating multiple avenues for further research.

## Requirements
The code has been tested with Python 3.6.9:

```
matplotlib==3.3.4
numba==0.50.1
numpy==1.19.5
opencv_python==4.5.4.58
Pillow==8.4.0
fastdtw==0.3.4
editdistance==0.6.1
requests==2.31.0
scipy==1.5.4
torch==1.5.1+cu92
torchvision==0.6.1+cu92
tqdm==4.64.1
tensorboard==2.10.1
termcolor==1.1.0
```
These torch and torchvision version are not provided by the pip package so you can download [here](https://pytorch.org/get-started/previous-versions/)
## Usage

Import the video for which you want to generate the scanpath to the following path:

```
├── data
    ├── frames
        ├── (video name)
    ├── saliency_maps
        ├── (video name)
    ├── test_split_VRET.txt
    ├── train_split_VRET.txt
├── runs
├── results
├── models
├── output_scanpaths
```
- **data**: Contains the videos and related data.
  - **frames**: Place the frames of the original video here as '.png' files in the corresponding directories.
    - **(video name)**: ID of the D-SAV360 dataset video.
  - **saliency_maps**: Place the saliency maps for each image from the 'frames' folder here. The saliency maps should be grayscale images where brighter areas indicate higher saliency values. If there isn't ground truth data, it is recommended to use [SST-Sal](https://github.com/edurnebernal/SST-Sal/) for creating the saliency maps.
    - **(video name)**: ID of the D-SAV360 dataset video.
  - **test_split_VRET.txt**: Specifies which videos to use for testing.
  - **train_split_VRET.txt**: Specifies which videos to use for training.

- **runs**: Contains TensorBoard logs for visualizing training and evaluation processes.

- **results**: Contains the results of the model inference, which can be either frames or a video, depending on the settings specified in `config.py`.

- **models**: Contains the weights of the models used. Note that these weights are from models overfitted on the D-SAV360 dataset.
- **output_scanpaths**: Contains the scanpaths generated with heuristic methods and the videos generated with the visualizer.
- **videos**: Place D-SAV360 videos here and execute `dataset_extration.py`  to get the frames of the videos.

To use the D-SAV360 dataset download it at the following [link](https://graphics.unizar.es/projects/D-SAV360/).

### Execution

To execute the program, follow these steps:

#### Generating and Analyzing Scanpaths

To generate, visualize, or analyze scanpaths using heuristic methods, run the following command:
```
python main.py
```
You can modify the settings in `config.py` to specify whether the output should be frames or a video, and to select the specific analysis or visualization parameters.

After running the program, an 'output' folder will be created containing the results, which can be either frames or a video.

#### Training

To train the model, run the following command:

```
python train.py
```
Ensure that the training data is correctly placed in the 'data/train_split_VRET.txt' file.

#### Inference

To run inference and generate scanpaths for the videos, run the following command:
```
python inference.py
```
