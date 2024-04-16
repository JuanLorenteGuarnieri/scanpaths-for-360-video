#####################################################################################
# General parameters
#####################################################################################

# Indicates if the program should create a new scanpath/s
generate = False
# Indicates if the scanpath sould be used to create a video with the scanpath overlapping it
visualize = False
# Indicates if a scanpath sould be analyzed with some metrics
analyze = True

#####################################################################################
# Scanpath generator parameters
#####################################################################################

# Indicates the number of scanpaths to be generated, if visualize option is set use n_scanpaths=1 
# to avoid generating all videos and overwritting them
n_scanpaths = 10
# Path to the folder containing the original frames and saliency maps
folder_path = './data/5035'
# Indicates if the program used for generate scanpaths shows the saliency maps in the output video
overlay_saliency = False
# Indicates the number of the last prediction frames that should be shown in the output video
g_history_length = 10 

### Type of scanpath generator to use ###
# ('random', 'max_saliency', 'percentile_saliency', 'probabilistic_saliency' or 'inhibition_saliency')
scanpath_generator_type = 'random'

# Percentile for random scanpath generation type 'percentile_saliency' (e.g., 50 for median)
percentile = 70

# Importance factor for probabilistic scanpath generation type 'probabilistic_saliency' and 'inhibition_saliency'
# (e.g., 1.0 for no adjustment)
probabilistic_importance = 8.0

# Parameters for scanpath generation with inhibition type 'inhibition_saliency'
inhibition_radius = 20  # Radius around selected points where saliency will be reduced
inhibition_decay = 0.8  # Factor by which saliency is reduced within the inhibition radius
inhibition_history_length = 5  # Number of recent points to consider for inhibition

equator_bias = True # if True decrease the saliency of Y far from the equator
bias_strength = 1.0 # Strength of the equatorial bias to increase the saliency of points near the horizontal center

fixation_distance = True   # if True decrease the saliency of points based on their distance from a current fixation point
fixation_angle = 5


#####################################################################################
# Scanpath visualizer parameters
#####################################################################################
# type of visualizer: "preview" or "multi" or "thumbnail"
v_type = "thumbnail"

# Indicates the file where the visualizer will load the scanpaths from
v_name = "elephant"
v_parameters = '_N100_inhibition_saliency_R20_D0.9_L5'


# Indicates the scanpath numer i who will be loaded to be visualize [1 ... N]
i_scanpath = 6
# Indicates the number of the last prediction frames that should be shown in the output video
v_history_length = 10 


#####################################################################################
# Scanpath analyzer parameters
#####################################################################################

# Indicates the file where the visualizer will load the scanpaths from
a_name = ""
a_parameters = 'inhibition_saliency_R20_D0.8_L5'


#####################################################################################
# Data Loader parameters
#####################################################################################
# Sequence length
sequence_length = 20
# Image resolution (height, width)
resolution = (240, 320)
# Path to the folder containing the RGB frames
frames_dir = 'data/frames'
# Path to the folder containing the optical flow
optical_flow_dir = 'data/optical_flow'
# Path to the folder containing the ground truth saliency maps
gt_dir = 'data/saliency_maps'
# Txt file containing the list of video names to be used for training
videos_train_file = 'data/train_split_VRET.txt'
# Txt file containing the list of video names to be used for testing
videos_test_file = 'data/test_split_VRET.txt'

#####################################################################################
# Training parameters
#####################################################################################
# Batch size
batch_size = 1
# Nº of epochs
epochs = 1
# Learning rate
lr = 0.8
# Hidden dimension of the model (SST-Sal uses hidden_dim=36)
hidden_dim = 36
# Percentage of training data intended to validation
validation_split = 0.2
# Name of the model ( for saving pruposes)
model_name = 'SST-Sal-2-epochs'
# Path to the folder where the checkpoints will be saved
ckp_dir = 'checkpoints'
# Path to the folder where the model will be saved
models_dir ='models'
# Path to the folder where the training logs will be saved
runs_data_dir = 'runs'

#####################################################################################
# Inference parameters
#####################################################################################
# Path to the folder containing the model to be used for inference
# inference_model = 'models/SST_Sal.pth' # with optical flow
inference_model = 'models/SST_Sal_wo_OF.pth' # without optical flow
# Path to the folder where the inference results will be saved
results_dir = 'results'
# Path to the folder containing the videos to be used for inference
videos_folder = 'data/videos'
# Indicates if the model used for inference is trained with or without optical flow
of_available = False
