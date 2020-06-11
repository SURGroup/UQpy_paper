from UQpy.Reliability import SubsetSimulation
import matplotlib.pyplot as plt
import numpy as np
from UQpy.SampleMethods import MMH
from UQpy.Distributions import Normal, MVNormal
from UQpy.RunModel import RunModel
import glob
import pickle
import os
import math
import time

calling_directory = os.getcwd()
t = time.time()

var_names = ['qtd', 'fy']

abaqus_sfe_model = RunModel(model_script='abaqus_subset_sfe_model_script.py',
                            input_template='abaqus_input_subset_sfe.py',
                            output_script='extract_abaqus_output_subset_sfe.py',
                            var_names=['qtd', 'fy'], model_dir='Subset_SFE',  ntasks=24)
print('Example: Created the model object.')

# Specify the target distribution. This is standard normal for use with subset simulation in UQpy.
dist = MVNormal(mean=np.zeros(2), cov=np.eye(2))

# Define the initial samples from the distribution
x = dist.rvs(nsamples=1000, random_state=834765)

# Run Subset Simulation
x_ss = SubsetSimulation(mcmc_class=MMH, runmodel_object=abaqus_sfe_model, samples_init=x, p_cond=0.1,
                        nsamples_per_ss=1000, verbose=True, random_state=923457, log_pdf_target=dist.log_pdf,
                        dimension=2, nchains=100)

# Save the results in a dictionary
results = {'pf': x_ss.pf, 'cov': x_ss.cov1, 'samples': x_ss.samples, 'g': x_ss.g}

# Pickle the results dictionary in the current directory. The basename and extension of the desired pickle file:
res_basename = 'Subset_sfe_results'
res_extension = '.pkl'

# Create a new results file with a larger index than any existing results files with the same name in the current
# directory.
res_file_list = glob.glob(res_basename + '_???' + res_extension)
if len(res_file_list) == 0:
    res_file_name = res_basename + '_000' + res_extension
else:
    max_number = max(res_file_list).split('.')[0].split('_')[-1]
    res_file_name = res_basename + '_%03d' % (int(max_number) + 1) + res_extension

res_file_name = os.path.join(calling_directory, res_file_name)
# Save the results to this new file.
with open(res_file_name, 'wb') as f:
    pickle.dump(results, f)
print('Saved the results to ' + res_file_name)

print('Example: Done!')
print('Time elapsed: %.2f minutes' % float((time.time() - t) / 60.0))

