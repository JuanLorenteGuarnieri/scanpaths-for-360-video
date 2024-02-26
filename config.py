#####################################################################################
# Scanpath generator parameters
#####################################################################################
# Path to the folder containing the original frames and saliency maps
folder_path = './data/input_2'
# Indicates if the program used for generate scanpaths shows the saliency maps in the output video
overlay_saliency = True
# Indicates the number of the last prediction frames that should be shown in the output video
history_length = 10
# Type of scanpath generator to use ('random' or 'max_saliency' or 'percentile_saliency' or 'probabilistic_saliency')
scanpath_generator_type = 'percentile_saliency'
# Percentile for random scanpath generation type 'percentile_saliency' (e.g., 50 for median)
percentile = 10
# Importance factor for probabilistic scanpath generation type 'probabilistic_saliency' (e.g., 1.0 for no adjustment) [probabilistic_importance > 0]
probabilistic_importance = 2.0
