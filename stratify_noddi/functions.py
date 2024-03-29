#!/usr/bin/env python

import os
import time
import datetime
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import filters


assert "ANTSPATH" in os.environ, "The environment variable ANTSPATH must be declared."
ANTSPATH = os.environ['ANTSPATH']
assert "FSLDIR" in os.environ, "The environment variable FSLDIR must be declared."
FSLDIR = os.environ['FSLDIR']

def check_dwi_files(dwi_opts, extension):
	"""
	Checks the dwi file list, selects the file with the desired extension, and checks if the file exists.
	
	Parameters
	----------
	dwi_opts : list
	extension : str
	
	Returns
	-------
	selected_file : str
	"""
	for dwi_file in dwi_opts:
		if dwi_file.endswith(extension):
			selected_file = dwi_file
			assert os.path.exists(dwi_file), "Error: dwi file [%s] does not exist. Please check your path."
	try:
		return(selected_file)
	except NameError:
		return(None)

def generate_qc_tsv(images, outname, sep = "\t", generate_histogram = True, nonimages = ['V1', 'V2', 'V3', 'mo', 'restore-V1', 'restore-V2','restore-V3', 'tensor', 'mask']):
	"""
	Generate QC log files Summary Tensor Fitting with weighted least square
	
	Parameters
	----------
	images : list
	outname : str
	
	Returns
	-------
	None
	"""
	Image = []
	Mean = []
	Std = []
	Min = []
	Max = []
	Noutlier = []

	for img in images:
		data = nib.load(img).get_fdata()
		data[np.isnan(data)] = 0
		index = data != 0
		metric = os.path.basename(img).replace(".nii.gz","").split("_")[-1]
		if not metric in nonimages:
			data_subset = np.copy(data[index])
			mu = np.mean(data_subset)
			stdev = np.std(data_subset)
			outlier = ((np.abs(data_subset) - mu) > 5*stdev)
			if np.sum(outlier) != 0:
				print("%s : n_outliers = %d" % (metric, np.sum(outlier)))
			Image.append(metric)
			Mean.append(np.mean(data_subset))
			Std.append(np.std(data_subset))
			Min.append(np.min(data_subset))
			Max.append(np.max(data_subset))
			Noutlier.append(np.sum(outlier))
			if generate_histogram:
				plt.hist(data_subset[~outlier], bins=200)
				if np.sum(outlier) > 0:
					textstr = 'Warning %d >5 s.d. outlier voxels removed.' % np.sum(outlier)
					props = dict(boxstyle='round', facecolor='red', alpha=0.5)
					plt.text(0.1, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
					verticalalignment='top', bbox=props)

				plt.savefig(fname = img.replace(".nii.gz","_histogram.png"),
								bbox_inches = 'tight')
				plt.close()
	outpd = pd.DataFrame()
	outpd["Img"] = Image
	outpd["Mean"] = Mean
	outpd["SD"] = Std
	outpd["Min"] = Min
	outpd["Max"] = Max
	outpd["Noutliers"] = Noutlier
	outpd.to_csv(outname, sep = sep, index = False)

def get_wildcard(searchstring, printarray = False): # super dirty
	"""
	Essentially glob but using bash. It outputs search arrays if more than one file is found.
	
	Parameters
	----------
	searchstring : str
	printarray : bool
	
	Returns
	-------
	outstring : str or array
	"""
	tmp_name = 'tmp_wildcard_%d' % np.random.randint(100000)
	os.system('echo %s > %s' % (searchstring, tmp_name))
	outstring = np.genfromtxt(tmp_name, dtype=str)
	os.system('rm %s' % tmp_name)
	if outstring.ndim == 0:
		return str(outstring)
	else:
		print("Multiple wildcards found ['%s' length = %d]. Outputting an array." % (searchstring, len(outstring)))
		if printarray:
			print (outstring)
		return outstring

def runCmd_log(cmd, logname = 'cmd_log'):
	"""
	Run a system command and logs it
	
	Parameters
	----------
	cmd : str
		Text string of the system command.
	logname : str
		The log file output file.
	Returns
	-------
	outstring : str or array
	"""
	ct = time.time()
	with open("cmd_log", "a") as logfile:
		logfile.write("[%s]\nCMD: %s\n" % (datetime.datetime.now(),cmd))
	os.system(cmd)
	print("Timestamp\t[%s]\tElapsed\t[%1.2fs]\n[CMD] %s" % (datetime.datetime.now(), (time.time() - ct), cmd))

def antsLinearRegCmd(numthreads, reference, mov, out_basename, outdir = None, use_float = False):
	"""
	Wrapper for ANTs linear registration with some recommended parameters.
	Rigid transfomration: gradient step = 0.1
	Mutual information metric: weight = 1; bins = 32.
	Convergence: [1000x500x250x100,1e-6,10]
	shrink-factors: 8x4x2x1
	smoothing-sigmas: 3x2x1x0vox
	
	Parameters
	----------
	numthreads : int
		The number of threads for parallel processing
	reference : str
		The reference image.
	mov : str
		The moving image.
	out_basename : str
		Output basename.
	outdir : str
		Output directory (options).

	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('''export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; \
						%s/antsRegistration -d 3 \
							-r [ %s , %s, 1] \
							-t Rigid[0.1] \
							-m MI[ %s , %s , 1, 32] \
							--convergence [1000x500x250x100,1e-6,10] \
							--shrink-factors 8x4x2x1 \
							--smoothing-sigmas 3x2x1x0vox \
							-o [%s%s_, %s%s.nii.gz]''' % (numthreads,
											ANTSPATH,
											reference,
											mov,
											reference,
											mov,
											outdir,
											out_basename,
											outdir,
											out_basename))
	if use_float:
		ants_cmd += ' --float'
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd

def antsNonLinearRegCmd(numthreads, reference, mov, out_basename, outdir = None, use_float = False):
	"""
	Wrapper for ANTs non-linear registration with some recommended parameters. I recommmend first using antsLinearRegCmd.
	SyN transformation: [0.1,3,0] 
	Mutual information metric: weight = 1; bins = 32.
	Convergence: [100x70x50x20,1e-6,10]
	shrink-factors: 8x4x2x1
	smoothing-sigmas: 3x2x1x0vox
	
	Parameters
	----------
	numthreads : int
		The number of threads for parallel processing
	reference : str
		The reference image.
	mov : str
		The moving image.
	out_basename : str
		Output basename.
	outdir : str
		Output directory (options).

	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('''export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; \
					%s/antsRegistration -d 3 \
						--transform SyN[0.1,3,0] \
						-m MI[ %s , %s , 1, 32] \
						--convergence [100x70x50x20,1e-6,10] \
						--shrink-factors 8x4x2x1 \
						--smoothing-sigmas 3x2x1x0vox \
						-o [%s%s_, %s%s.nii.gz]''' % (numthreads,
															ANTSPATH,
															reference,
															mov,
															outdir,
															out_basename,
															outdir,
															out_basename))
	if use_float:
		ants_cmd += ' --float'
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd

def antsApplyTransformCmd(reference, mov, warps, outname, outdir = None, inverse = False, multipleimages = False):
	"""
	Wrapper for applying ANTs transformations (warps).
	
	Parameters
	----------
	reference : str
		The reference image.
	mov : str
		The moving image.
	warps : arr
		An array of warps to appy. It must always be an array even for a single warp!
	outname : str
		Output basename.
	outdir : str
		Output directory (options).

	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	if multipleimages:
		e_ = '3'
	else:
		e_ = '0'

	warps = np.array(warps)
	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('%s/antsApplyTransforms -d 3 -r %s -i %s -e %s -o %s%s --float' % (ANTSPATH,
																										reference,
																										mov,
																										e_,
																										outdir,
																										outname))
	for i in range(len(warps)):
		if inverse:
			ants_cmd = ants_cmd + (' -t [%s, 1]' % warps[i])
		else:
			ants_cmd = ants_cmd + (' -t %s' % warps[i])
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd


def antsBetCmd(numthreads, input_image, output_image_brain):
	"""
	Wrapper for applying ANTs transformations (warps).
	
	Parameters
	----------
	numthreads : int
		The number of threads for parallel processing
	input_image : str
		Anatomical image.
	output_image_brain : str
		Brain extracted output image.
	
	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	be_template = "%s/ants_tbss/ants_oasis_template_ras/T_template0.nii.gz" % scriptwd
	be_probability_mask = "%s/ants_tbss/ants_oasis_template_ras/T_template0_BrainCerebellumProbabilityMask.nii.gz" % scriptwd
	be_registration_mask = "%s/ants_tbss/ants_oasis_template_ras/T_template0_BrainCerebellumRegistrationMask.nii.gz" % scriptwd
	ants_cmd = ("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; \
					%s/antsBrainExtraction.sh -d 3 \
						-a %s -e %s -m %s -f %s \
						-o %s" % (numthreads,
									ANTSPATH, 
									input_image,
									be_template,
									be_probability_mask,
									be_registration_mask,
									output_image_brain))
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd


def round_mask_transform(mask_image):
	"""
	Binarize a mask using numpy round and overwrites it.
	
	Parameters
	----------
	mask_image : str
		/path/to/mask_image
	
	Returns
	-------
	None
	"""
	img = nib.load(mask_image)
	img_data = img.get_data()
	img_data = np.round(img_data)
	nib.save(nib.Nifti1Image(img_data,img.affine), mask_image)


def nifti_to_float_precision(img_name):
	"""
	Converts nifti image to float32 precision.
	
	Parameters
	----------
	img_name : str
		/path/to/image
	
	Returns
	-------
	None
	"""
	img = nib.load(img_name)
	if img.get_data_dtype() != '<f4':
		img_data = img.get_data()
		nib.save(nib.Nifti1Image(img_data.astype(np.float32), img.affine), img_name)


def autothreshold(data, threshold_type = 'yen', z = 2.3264):
	"""
	Autothresholds data.
	
	Parameters
	----------
	data : array
		data array for autothresholding
	threshold_type : str
		autothresholding algorithms {'otsu' | 'li' | 'yen' | 'otsu_p' | 'li_p' | 'yen_p' | 'zscore_p'}. '*_p' calculates thresholds on only positive data.
		Citations:
			Otsu N (1979) A threshold selection method from gray-level histograms. IEEE Trans. Sys., Man., Cyber. 9: 62-66.
			Li C.H. and Lee C.K. (1993) Minimum Cross Entropy Thresholding Pattern Recognition, 26(4): 617-625
			Yen J.C., Chang F.J., and Chang S. (1995) A New Criterion for Automatic Multilevel Thresholding IEEE Trans. on Image Processing, 4(3): 370-378.
	z : float
		z-score threshold for using zscore_p
	
	Returns
	-------
	lthres : float
		The lower threshold.
	uthres : float
		The highier threshold.

	"""
	if threshold_type.endswith('_p'):
		data = data[data>0]
	else:
		data = data[data!=0]
	if data.size == 0:
		print("Warning: the data array is empty. Auto-thesholding will not be performed")
		return 0, 0
	else:
		if (threshold_type == 'otsu') or (threshold_type == 'otsu_p'):
			lthres = filters.threshold_otsu(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		elif (threshold_type == 'li')  or (threshold_type == 'li_p'):
			lthres = filters.threshold_li(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		elif (threshold_type == 'yen') or (threshold_type == 'yen_p'):
			lthres = filters.threshold_yen(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		elif threshold_type == 'zscore_p':
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
			if lthres < 0:
				lthres = 0.001
		else:
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
		if uthres > data.max(): # for the rare case when uthres is larger than the max value
			uthres = data.max()
		return lthres, uthres


