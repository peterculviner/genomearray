# code for local testing of genomearray code on laublab server
import sys, os
import numpy as np
sys.path.append(os.path.relpath("/home/laublab/notebooks/dropbox_link/culviner/repositories/genomearray/"))
import genomearray as ga

# test region slicing functionality

def test_genomewrt_noaddl():
	genome_data = np.asarray([[0,1,2,3,4,5,6,7,8,9],
			   			  [0,1,2,3,4,5,6,7,8,9]])
	input_regions = np.asarray([[0,1,8],
						  	    [1,1,8]])
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (0,0), wrt = 'genome')
	assert output[0][0] == 1 and output[0][1] == 8 and output[1][0] == 1 and output[1][1] == 8

def test_5to3wrt_noaddl():
	genome_data = np.asarray([[0,1,2,3,4,5,6,7,8,9],
			   			  [0,1,2,3,4,5,6,7,8,9]])
	input_regions = np.asarray([[0,1,8],
						  	    [1,1,8]])
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (0,0), wrt = '5_to_3')
	assert output[0][0] == 1 and output[0][1] == 8 and output[1][0] == 8 and output[1][1] == 1

def test_genomewrt_addl():
	# check both sides
	genome_data = np.asarray([[0,1,2,3,4,5,6,7,8,9],
			   			  [0,1,2,3,4,5,6,7,8,9]])
	input_regions = np.asarray([[0,1,8],
						  	    [1,1,8]])
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (1,1), wrt = 'genome')
	assert output[0][0] == 0 and output[0][1] == 9 and output[1][0] == 0 and output[1][1] == 9
	# check to ensure edges are properly handled
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (1,0), wrt = 'genome')
	assert output[0][0] == 0 and output[0][1] == 8 and output[1][0] == 0 and output[1][1] == 8

def test_5to3wrt_addl():
	# check both sides
	genome_data = np.asarray([[0,1,2,3,4,5,6,7,8,9],
			   			  [0,1,2,3,4,5,6,7,8,9]])
	input_regions = np.asarray([[0,1,8],
						  	    [1,1,8]])
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (1,1), wrt = '5_to_3')
	assert output[0][0] == 0 and output[0][1] == 9 and output[1][0] == 9 and output[1][1] == 0
	# check to ensure edges are properly handled
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (1,0), wrt = '5_to_3')
	assert output[0][0] == 0 and output[0][1] == 8 and output[1][0] == 9 and output[1][1] == 1

# test position slicing functionality

def test_pos_5to3wrt_addl():
	# check both sides
	genome_data = np.asarray([[0,1,2,3,4,5,6,7,8,9],
			   			  [0,1,2,3,4,5,6,7,8,9]])
	input_regions = np.asarray([[0,1],
					  	    [1,1],
					  	    [1,1]])
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (1,1), wrt = '5_to_3')
	assert output[0][0] == 0 and output[0][1] == 2 and output[1][0] == 2 and output[1][1] == 0 and output[1][0] == 2 and output[1][1] == 0
	# check to ensure edges are properly handled
	output = ga.regionfunc(lambda x: (x[0],x[-1]), input_regions, genome_data, addl_nt = (1,0), wrt = '5_to_3')
	assert output[0][0] == 0 and output[0][1] == 1 and output[1][0] == 2 and output[1][1] == 1 and output[1][0] == 2 and output[1][1] == 1