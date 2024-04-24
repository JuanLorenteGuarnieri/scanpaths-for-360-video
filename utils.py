import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pylab import *
from tqdm import tqdm
import os
import cv2
import config

def read_txt_file(path_to_file):
    """
    The names of the videos to be used for training, they must be in a single line separated
    with ','.
    :param path_to_file: where the file is saved (ex. 'data/file.txt')
    :return: list of strings with the names
    """

    with open(path_to_file) as f:
        for line in f:
            names = line.rsplit('\n')[0].split(',')
    return names

def frames_extraction(path_to_videos):
    """
    Extracts the frames from the videos and save them in the same folder as the original videos.
    :param path_to_videos: path to the videos
    :return:
    """
    samples_per_second = 8
    destination_folder = os.path.join(config.videos_folder, 'frames')
    videos_folder = config.videos_folder

    video_names = os.listdir(videos_folder)
    with tqdm(range(len(video_names)), ascii=True) as pbar:
        for v_n, video_name in enumerate(video_names):
            
            video = cv2.VideoCapture(os.path.join(videos_folder, video_name))
            fps = video.get(cv2.CAP_PROP_FPS)
            step = round(fps / samples_per_second)

            new_video_folder = os.path.join(destination_folder, str(v_n).zfill(3))

            if not os.path.exists(new_video_folder):
                os.makedirs(new_video_folder)

            success, frame = video.read()
            frame_id = 0
            frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
            cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
            frame_id += 1

            while success:
                success, frame = video.read()
                if frame_id % step == 0 and success:
                    frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
                    cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
                frame_id += 1
            pbar.update(1)

def blend(sample_sal_map, sample_frame):

    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(sample_sal_map)
               * 2 ** 8).astype(np.uint8)[:, :, :3]

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    dst = cv2.addWeighted(np.array(sample_frame).astype(
        np.uint8), 0.5, heatmap, 0.5, 0.0)
    return dst

def save_video(frames_dir, pred_dir, gt_dir, output_vid_name = 'SST-Sal_pred.avi'):
    """
    Saves the video with the predicted and ground truth saliency maps.
    :param frames: list of frames 
    :param pred: list of predicted saliency maps
    :param gt: list of ground truth saliency maps
    """

    out_pred = cv2.VideoWriter(os.path.join( config.results_dir, output_vid_name), cv2.VideoWriter_fourcc(*'DIVX'), 8, (1024, 540))
    if not gt_dir is None:
        out_gt = cv2.VideoWriter(os.path.join( config.results_dir, 'gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 8, (1024, 540))

    video_frames_names = os.listdir(pred_dir)
    video_frames_names = sorted(video_frames_names, key=lambda x: int((x.split(".")[0]).split("_")[1]))

    for iFrame in tqdm(video_frames_names):
        
        img_frame = cv2.imread(os.path.join(frames_dir, iFrame))
        img_frame = cv2.resize(img_frame, (1024, 540), interpolation=cv2.INTER_AREA)

        if not gt_dir is None:
            gt_salmap = cv2.imread(os.path.join(gt_dir, iFrame.split('_')[1]), cv2.IMREAD_GRAYSCALE)
            gt_salmap = cv2.resize(gt_salmap, (1024, 540), interpolation=cv2.INTER_AREA)


        pred_salmap = cv2.imread(os.path.join(pred_dir, iFrame), cv2.IMREAD_GRAYSCALE)
        pred_salmap = cv2.resize(pred_salmap, (1024, 540), interpolation=cv2.INTER_AREA)


        pred_blend = blend(pred_salmap, img_frame)
        out_pred.write(pred_blend)
        if not gt_dir is None:
            gt_blend = blend(gt_salmap, img_frame)
            out_gt.write(gt_blend)
    out_pred.release()
    if not gt_dir is None:
        out_gt.release()


"""
  code from:
https://github.com/DaniMS-ZGZ/saliency/blob/master/metrics/utils.py
"""
def scanpath_to_string(scanpath, height, width, Xbins, Ybins, Tbins):
	"""
			a b c d ...
		A
		B
		C
		D

		returns Aa
	"""
	if Tbins !=0:
		try:
			assert scanpath.shape[1] == 3
		except Exception as x:
			print("Temporal information doesn't exist.")

	height_step, width_step = height//Ybins, width//Xbins
	string = ''
	num = list()
	for i in range(scanpath.shape[0]):
		fixation = scanpath[i].astype(np.int32)
		xbin = fixation[0]//width_step
		ybin = ((height - fixation[1])//height_step)
		corrs_x = chr(65 + xbin)
		corrs_y = chr(97 + ybin)
		T = 1
		if Tbins:
			T = fixation[2]//Tbins
		for t in range(T):
			string += (corrs_y + corrs_x)
			num += [(ybin * Xbins) + xbin]
	return string, num


def global_align(P, Q, SubMatrix=None, gap=0, match=1, mismatch=-1):
	"""
		https://bitbucket.org/brentp/biostuff/src/
	"""
	UP, LEFT, DIAG, NONE = range(4)
	max_p = len(P)
	max_q = len(Q)
	score   = np.zeros((max_p + 1, max_q + 1), dtype='f')
	pointer = np.zeros((max_p + 1, max_q + 1), dtype='i')

	pointer[0, 0] = NONE
	score[0, 0] = 0.0
	pointer[0, 1:] = LEFT
	pointer[1:, 0] = UP

	score[0, 1:] = gap * np.arange(max_q)
	score[1:, 0] = gap * np.arange(max_p).T

	for i in range(1, max_p + 1):
		ci = P[i - 1]
		for j in range(1, max_q + 1):
			cj = Q[j - 1]
			if SubMatrix is None:
				diag_score = score[i - 1, j - 1] + (cj == ci and match or mismatch)
			else:
				diag_score = score[i - 1, j - 1] + SubMatrix[cj][ci]
			up_score   = score[i - 1, j] + gap
			left_score = score[i, j - 1] + gap

			if diag_score >= up_score:
				if diag_score >= left_score:
					score[i, j] = diag_score
					pointer[i, j] = DIAG
				else:
					score[i, j] = left_score
					pointer[i, j] = LEFT
			else:
				if up_score > left_score:
					score[i, j ]  = up_score
					pointer[i, j] = UP
				else:
					score[i, j]   = left_score
					pointer[i, j] = LEFT

	align_j = ""
	align_i = ""
	while True:
		p = pointer[i, j]
		if p == NONE: break
		s = score[i, j]
		if p == DIAG:
			# align_j += Q[j - 1]
			# align_i += P[i - 1]
			i -= 1
			j -= 1
		elif p == LEFT:
			# align_j += Q[j - 1]
			# align_i += "-"
			j -= 1
		elif p == UP:
			# align_j += "-"
			# align_i += P[i - 1]
			i -= 1
		else:
			raise ValueError
	# return align_j[::-1], align_i[::-1]
	return score.max()




# def MM_simplify_scanpath(P, th_glob, th_dur, th_amp):

# 	class Scanpath(object):
# 		"""
# 			Modeling scanpaths simialar to codes published by authors.

# 		"""
# 		def __init__(self, fixation==list()):
# 			# class Saccade(object):
# 				# def __init__(self, fixations):
# 			self.x = list()
# 			self.y = list()
# 			self.lenx = list()
# 			self.leny = list()
# 			self.len = list()
# 			self.theta = list()
# 			self.dur = list()
# 			if fixations:
# 				self.prep(fixations)

# 		def prep(self, fixations):
# 			for fix_idx, fix in enumerate(fixations):
# 				self.x.append(fix[0])
# 				self.y.append(fix[1])
# 				self.dur.append(fix[2])
# 				if fix_idx >= 1:
# 					self.lenx.append(fix[0] - self.x[fix_idx -1])
# 					self.leny.append(fix[1] - self.y[fix_idx -1])
# 					tmp = self.cart2pol(self.lenx[fix_idx-1], self.leny[fix_idx-1])
# 					self.theta.append(tmp[1])
# 					self.len.append(tmp[0])

# 		def cart2pol(self, x, y):
# 			rho = np.sqrt(x**2 + y**2)
# 			phi = np.arctan2(y, x)
# 			return(rho, phi)


# 		def add_saccade(self, x, y, lenx, leny, Len, theta, dur):
# 			self.x.append(x)
# 			self.y.append(y)
# 			self.lenx.append(lenx)
# 			self.leny.append(leny)
# 			self.len.append(Len)
# 			self.theta.append(theta)
# 			self.dur.append(dur)


# 	def simplify_duration(P, th_glob=th_glob, th_dur=th_dur):
# 		i = 0

# 		p_sim = Scanpath()

# 		while i < len(P.x):
# 			if i < length(sp.saccade.x):
# 				v1=[P.lenx[i],P.leny[i]];
# 				v2=[P.lenx[i+1],P.leny[i+1]];
# 				angle = np.arccos(np.dot(v1,v2))
# 				angle = angle / (np.linalg.norm(v1,2)*np.linalg.norm(v2,2));
# 			else:
# 				angle = np.inf;

# 			if (angle < th_glob) and (i < len(P.x)):

# 				#Do not merge saccades if the intermediate fixation druations are
# 				# long
# 				if P.dur[i+1] >= th_dur:
# 					p_sim.add_saccade()
# 					[sp,spGlobal,i,durMem] = keepSaccade(sp,spGlobal,i,j,durMem);
# 					j = j+1;
# 					continue,
# 				end

# 				% calculate sum of local vectors.
# 				v_x = sp.saccade.lenx(i) + sp.saccade.lenx(i+1);
# 				v_y = sp.saccade.leny(i) + sp.saccade.leny(i+1);
# 				[theta,len] = cart2pol(v_x,v_y);

# 				% ... save them a new global vectors
# 				spGlobal.saccade.x(j) = sp.saccade.x(i);
# 				spGlobal.saccade.y(j) = sp.saccade.y(i);
# 				spGlobal.saccade.lenx(j) = v_x;
# 				spGlobal.saccade.leny(j) = v_y;
# 				spGlobal.saccade.len(j) = len;
# 				spGlobal.saccade.theta(j) = theta;

# 				%... and sum up all the fixation durations
# 				spGlobal.fixation.dur(j) = sp.fixation.dur(i);%+sp.fixation.dur(i+1)/2+durMem;
# 				durMem = 0;%sp.fixation.dur(i+1)/2;
# 				i = i+2;


# 	def simplyfy_length(th_glob=th_glob, th_amp=th_amp):
# 		pass


# 	P = scanpath(P)

# 	l = 10000
# 	while True
# 		P = simplify_duration(P, Tdur)
# 		P = simplify_length(P, Tamp)
# 		if l == P.fixation[0]:
# 			break
# 		l = len(P.fixation[0])

# 	return P






"""
  code by:
		Daniel Martín
"""

import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import re
import cv2
import matplotlib.cm as cm

def get_gnomonic_hom(center_lat_lon, origin_image, height_width, fov_vert_hor=(60.0, 60.0) ):
    '''Extracts a gnomonic viewport with height_width from origin_image 
    at center_lat_lon with field of view fov_vert_hor.
    '''
    org_height_width, _ = origin_image.shape[:2], origin_image.shape[-1]
    height, width = height_width
    
    if len(origin_image.shape) == 3:
        result_image = np.zeros((height, width, 3))
    else:
        result_image = np.zeros((height, width))        

    sphere_radius_lon = width / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = height / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    y, x = np.mgrid[0:height, 0:width]
    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0/sphere_radius_lon
    K_inv[1, 1] = 1.0/sphere_radius_lat
    K_inv[0, 2] = -width/(2.0*sphere_radius_lon)
    K_inv[1, 2] = -height/(2.0*sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3,3))
    R_lat[0,0] = 1.0
    R_lat[1,1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2,2] = R_lat[1,1]
    R_lat[1,2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2,1] = -1.0 * R_lat[1,2]

    R_lon = np.zeros((3,3))
    R_lon[2,2] = 1.0
    R_lon[0,0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1,1] = R_lon[0,0]
    R_lon[0,1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1,0] = - R_lon[0,1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1,3,3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod/np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    org_img_y_x = lat_lon / np.array([180.0, 360.0]) * np.array(org_height_width)
    org_img_y_x = np.clip(org_img_y_x, 0.0, np.array(org_height_width).reshape(1, 2) - 1.0).astype(int)
    org_img_y_x = org_img_y_x.astype(int)
    
    if len(origin_image.shape) == 3:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int), :] = origin_image[org_img_y_x[:, 0],
                                                                     org_img_y_x[:, 1], :]  
    else:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int)] = origin_image[org_img_y_x[:, 0],
                                                                     org_img_y_x[:, 1]] 
    return result_image.astype(float), org_img_y_x

def gnomonic2lat_lon(x_y_coords, fov_vert_hor, center_lat_lon):
	'''
	Converts gnomonoic (x, y) coordinates to (latitude, longitude) coordinates.
	
	x_y_coords: numpy array of floats of shape (num_coords, 2) 
	fov_vert_hor: tuple of vertical, horizontal field of view in degree
	center_lat_lon: The (lat, lon) coordinates of the center of the viewport that the x_y_coords are referencing.
	'''
	sphere_radius_lon = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
	sphere_radius_lat = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

	x, y = x_y_coords[:,0], x_y_coords[:,1]

	x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

	K_inv = np.zeros((3, 3))
	K_inv[0, 0] = 1.0/sphere_radius_lon
	K_inv[1, 1] = 1.0/sphere_radius_lat
	K_inv[0, 2] = -1./(2.0*sphere_radius_lon)
	K_inv[1, 2] = -1./(2.0*sphere_radius_lat)
	K_inv[2, 2] = 1.0

	R_lat = np.zeros((3,3))
	R_lat[0,0] = 1.0
	R_lat[1,1] = np.cos(np.radians(-center_lat_lon[0]))
	R_lat[2,2] = R_lat[1,1]
	R_lat[1,2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
	R_lat[2,1] = -1.0 * R_lat[1,2]

	R_lon = np.zeros((3,3))
	R_lon[2,2] = 1.0
	R_lon[0,0] = np.cos(np.radians(-center_lat_lon[1]))
	R_lon[1,1] = R_lon[0,0]
	R_lon[0,1] = - np.sin(np.radians(-center_lat_lon[1]))
	R_lon[1,0] = - R_lon[0,1]

	R_full = np.matmul(R_lon, R_lat)

	dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1,3,3) * x_y_hom.reshape(-1, 1, 3), axis=2)

	sphere_points = dot_prod/np.linalg.norm(dot_prod, axis=1, keepdims=True)

	lat = np.degrees(np.arccos(sphere_points[:, 2]))
	lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

	lat_lon = np.column_stack([lat, lon])
	lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

	return lat_lon

def angle2img(lat_lon_array, img_height_width):
	'''
	Convertes an array of latitude, longitude coordinates to image coordinates with range (0, height) x (0, width)
	'''
	return lat_lon_array / np.array([180., 360.]).reshape(1,2) * np.array(img_height_width).reshape(1,2)

def img2angle(x_y_array, img_height_width):
	'''
	Convertes an array of latitude, longitude coordinates to image coordinates with range (0, height) x (0, width)
	'''
	x = x_y_array[0]
	y = x_y_array[1]
	x = x / img_height_width[0] * 360
	y = y / img_height_width[1] * 180
	return [x,y]


def plot_fov(center_lat_lon, ax, color, fov_vert_hor, height_width):
	'''
	Plots the correctly warped FOV at a given center_lat_lon.
	center_lat_lon: Float tuple of latitude, longitude. Position where FOV is centered
	ax: The matplotlib axis object that should used for plotting.
	color: Color of the FOV box.
	height_width: Height and width of the image.
	'''
	# Coordinates for a rectangle.
	coords = []
	coords.append([np.linspace(0.0, 1.0, 100), [1.]*100])
	coords.append([[1.]*100, np.linspace(0.0, 1.0, 100)])
	coords.append([np.linspace(0.0, 1.0, 100), [0.]*100])
	coords.append([[0.]*100, np.linspace(0.0, 1.0, 100)])	

	lines = []
	for coord in coords:
		lat_lon_array = gnomonic2lat_lon(np.column_stack(coord), fov_vert_hor=fov_vert_hor, 
										 center_lat_lon=center_lat_lon)
		img_coord_array = angle2img(lat_lon_array, height_width)
		lines.append(img_coord_array)
		
	split_lines = []
	for line in lines:
		diff = np.diff(line, axis=0)
		wrap_idcs = np.where(np.abs(diff)>np.amin(height_width))[0]
		
		if not len(wrap_idcs):
			split_lines.append(line)
		else:
			split_lines.append(line[:wrap_idcs[0]+1])
			split_lines.append(line[wrap_idcs[0]+1:])

	for line in split_lines:
		ax.plot(line[:,1], line[:,0], color=color, linewidth=5.0, alpha=0.5)


def plot_viewport(scanpath, color, fov_vert_hor, path_to_save, image):
    # Ensure the output directory exists
	os.makedirs(path_to_save, exist_ok=True)
	points_x = [point[0]*image.shape[0] for point in scanpath]
	points_y = [point[1]*image.shape[1] for point in scanpath]
	lat_lon = None
	frame_no = 0
	for x_, y_ in zip(points_x, points_y):
			plt.close('all')

			if lat_lon is None:
				lat_lon = [0,0]
				lat_lon[1] = ((x_ / image.shape[1]) * 360)
				lat_lon[0] = ((y_ / image.shape[0]) * 180)
			else:
				diff_x = x_ - last_point[0]
				diff_y = y_ - last_point[1]
				lat_lon[1] = lat_lon[1] + ((diff_x / image.shape[1]) * 360)
				lat_lon[0] = lat_lon[0] + ((diff_y / image.shape[0]) * 180)
			last_point = [x_,y_] 

			fig, ax = plt.subplots(frameon=False, figsize=(16,9))
			
			ax.grid(b=False)
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)

			ax.imshow(image)
			ax.axis('tight')
			ax.set_xlim([0,image.shape[1]])
			ax.set_ylim([image.shape[0], 0])

			fov_vert = 106.188
			aspect_ratio = 0.82034051
			fov_hor = fov_vert * aspect_ratio
			fov_vert_hor = np.array([fov_vert, fov_hor])

			ax.plot(x_, y_, marker='o', markersize=12., color=color, alpha=.8)
			plot_fov(lat_lon, ax, color, fov_vert_hor, height_width=np.array([image.shape[0], image.shape[1]]))


			fig.savefig(os.path.join(path_to_save, "%06d.png"%frame_no), bbox_inches='tight', pad_inches=0, dpi=160)
			frame_no += 1
			fig.clf()
   

def plot_all_viewports(scanpaths, fov_vert_hor, path_to_save, name):
    # Ensure the output directory exists
    os.makedirs(path_to_save, exist_ok=True)
    
    cmap = cm.get_cmap('rainbow', len(scanpaths))  # Obtiene el mapa de colores
    colors = [cmap(i) for i in range(len(scanpaths))]
    
    # Determine the number of frames based on the longest scanpath
    max_length = max(len(scanpath) for scanpath in scanpaths)
    # Initialize a list to hold the last point of each scanpath for diff calculation
    last_points = [None] * len(scanpaths)
    lat_lon = [None] * len(scanpaths)
    
    if os.path.exists("./data/"+ name + "/original/"):
        original_video_path = "./data/"+ name + "/original/"
    else:
        original_video_path = None
    
    if original_video_path:
          video_frames = sorted([f for f in os.listdir(original_video_path) if f.endswith('.png') or f.endswith('.jpg')])
    else:
      image = np.ones((720, 1280, 3), dtype=np.uint8) * 255
      
    fov_vert = 106.188
    aspect_ratio = 0.82034051
    fov_hor = fov_vert * aspect_ratio
    fov_vert_hor = np.array([fov_vert, fov_hor])
    
    for frame_no in range(max_length):
        plt.close('all')  # Close all existing plots to avoid memory issues
        if original_video_path:
          image = mpimg.imread(os.path.join(original_video_path, video_frames[frame_no]))
        fig, ax = plt.subplots(frameon=False, figsize=(16, 9))
        ax.grid(False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.imshow(image)
        ax.axis('tight')
        ax.set_xlim([0, image.shape[1]])
        ax.set_ylim([image.shape[0], 0])
        
        # Iterate through each scanpath and plot the current point if it exists
        for i, (scanpath, color) in enumerate(zip(scanpaths, colors)):
            if frame_no < len(scanpath):
                point = scanpath[frame_no]
                x_ = point[1] * image.shape[1]
                y_ = point[0] * image.shape[0]

                if lat_lon[i] is None:
                    # Initialize lat_lon for the first point in each scanpath
                    lat_lon[i] = [0, 0]
                    lat_lon[i][1] = ((x_ / image.shape[1]) * 360)
                    lat_lon[i][0] = ((y_ / image.shape[0]) * 180)
                else:
                    # Calculate diff_x and diff_y with respect to the last point
                    diff_x = x_ - last_points[i][0]
                    diff_y = y_ - last_points[i][1]
                    # Update lat_lon based on the differences
                    lat_lon[i][1] += ((diff_x / image.shape[1]) * 360)
                    lat_lon[i][0] += ((diff_y / image.shape[0]) * 180) 

                # Update the last_points list with the current point
                last_points[i] = [x_, y_]

                ax.plot(x_, y_, marker='o', markersize=12., color=color, alpha=.8)
                
                plot_fov(lat_lon[i], ax, color, fov_vert_hor, height_width=np.array([image.shape[0], image.shape[1]]))

        # Save the figure for the current frame
        fig.savefig(f"{path_to_save}/frame_{str(frame_no).zfill(5)}.png", bbox_inches='tight', pad_inches=0, dpi=160)
        fig.clf()
        

def plot_thumbnail(scanpath, path_to_save, name):
    # Ensure the output directory exists
    os.makedirs(path_to_save, exist_ok=True)
  
    if os.path.exists("./data/"+ name + "/original/"):
        original_video_path = "./data/"+ name + "/original/"
    else:
      print("Could not find the original video directory")
      return
    
    video_frames = sorted([f for f in os.listdir(original_video_path) if f.endswith('.png') or f.endswith('.jpg')])

    lat_lon = None
    last_point = None
    height_width = [800,800]
    for frame_no in range(len(scanpath)):
        plt.close('all')  # Close all existing plots to avoid memory issues
        image = mpimg.imread(os.path.join(original_video_path, video_frames[frame_no]))
        point = scanpath[frame_no]
        x_ = point[1] * image.shape[1]
        y_ = point[0] * image.shape[0]

        if lat_lon is None:
          # Initialize lat_lon for the first point in each scanpath
          lat_lon = [0, 0]
          lat_lon[1] = ((x_ / image.shape[1]) * 360)
          lat_lon[0] = ((y_ / image.shape[0]) * 180)
        else:
          # Calculate diff_x and diff_y with respect to the last point
          diff_x = x_ - last_point[0]
          diff_y = y_ - last_point[1]
          # Update lat_lon based on the differences
          lat_lon[1] += ((diff_x / image.shape[1]) * 360)
          lat_lon[0] += ((diff_y / image.shape[0]) * 180)
        # Update the last_points list with the current point
        last_point = [x_, y_]
        thumbnail_image, org_img_y_x = get_gnomonic_hom(lat_lon, image, height_width, fov_vert_hor=(60.0, 60.0))
        fig, ax = plt.subplots(frameon=False, figsize=(16, 9))
        ax.grid(False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.imshow(thumbnail_image)
        ax.axis('tight')
        ax.set_xlim([0, height_width[1]])
        ax.set_ylim([height_width[0], 0])

        # Save the figure for the current frame
        fig.savefig(f"{path_to_save}/frame_{str(frame_no).zfill(5)}.png", bbox_inches='tight', pad_inches=0, dpi=160)
        fig.clf()