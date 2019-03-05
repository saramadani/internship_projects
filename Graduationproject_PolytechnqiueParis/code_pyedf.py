# Notebook initialization

%autosave 0
%matplotlib notebook

#We need to import these modules with the following commands:
from __future__ import division, print_function
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import pyedflib
fig_w, fig_h = (4.5, 3.5)
plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = False
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

# Configuration settings

path_folder = '/share/datasets/sleep-edfx/'
edf_ids_excluded =  ['SC4061', 'ST7071', 'ST7092', 'ST7132', 'ST7141', 'SC4131', 'ST7201',
                     'ST7011',  'ST7041', 'ST7101', 'ST7121', 'ST7151', 'ST7171', 'ST7241' ]
channels_ref = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'] 
channels_ref_short = ['EEG1', 'EEG2', 'EOG', 'EMG']
metric_short = {'loss': 'loss', 'categorical_accuracy': 'acc'}

freq_target = 60 #target frequency
awake_pre_post = 10
n_val = 300
epochs = 30
batch_size = 128

channel_masks = [[True, True, True, True], [True, False, False, False], [False, True, False, False],
                 [False, False, True, False], [False, False, False, True], [True, True, False, False],
                 [True, False, True, False], [True, False, False, True], [False, True, True, False],
                 [False, True, False, True], [False, False, True, True], [True, True, True, False], 
                 [True, True, False, True], [True, False, True, True], [False, True, True, True]]


#Get list of filenames
edf_files = [item for item in os.listdir(path_folder)
                  if (os.path.splitext(item)[1]).lower()=='.edf'] #get edf files with .edf suffix
#select first night subjects
edf_first_night = [item for item in os.listdir(path_folder)
                  if (os.path.splitext(item)[1]).lower()=='.edf' 
                   and item[5]=='1' ]                           #select files related to first night
edf_ids = set([item[:6] for item in edf_first_night
                        if item[:6]not in edf_ids_excluded])    #exclude inappropriate files

psg_files, hyp_files = ([], []) #initialize psg_fles and hyp_files lists to fill them later

for edf_id in sorted(edf_ids): #loop through edf_ids
    psg_files += [[os.path.join(path_folder, item) for item in edf_first_night
                     if (item[:6]==edf_id and item.endswith('-PSG.edf'))][0]]  #append psg files
    hyp_files += [[os.path.join(path_folder, item) for item in edf_first_night
                     if (item[:6]==edf_id and item.endswith('-Hypnogram.edf'))][0]] #append hyp files

#select second night subjects    
edf_second_night = [item for item in os.listdir(path_folder)
                  if (os.path.splitext(item)[1]).lower()=='.edf' 
                    and item[5]=='2' ]                          #select files related to first night
edf_ids = set([item[:6] for item in edf_second_night
                        if item[:6]not in edf_ids_excluded]) #exclude inappropriate files
psg_files, hyp_files = ([], [])   #initialize psg_fles and hyp_files lists to fill them later

for edf_id in sorted(edf_ids):   #loop through edf_ids
    psg_files += [[os.path.join(path_folder, item) for item in edf_second_night
                     if (item[:6]==edf_id and item.endswith('-PSG.edf'))][0]]  #append psg files
    hyp_files += [[os.path.join(path_folder, item) for item in edf_second_night
                     if (item[:6]==edf_id and item.endswith('-Hypnogram.edf'))][0]]  #append hyp files

#Get informations about files durations, start time for each file using pyedflib library

for psg_file, hyp_file in zip(psg_files, hyp_files): #loop through both of psg and hyp files
        
    with pyedflib.EdfReader(os.path.join(path_folder, psg_file)) as psg, \ #open psg file
         pyedflib.EdfReader(os.path.join(path_folder, hyp_file)) as hyp:   #open hyp file
        
        psg_start_datetime = psg.getStartdatetime()         #get psg start time
        hyp_start_datetime = hyp.getStartdatetime()         #get hyp start time
        
        psg_file_duration = psg.getFileDuration()           #get psg duration
        hyp_file_duration = sum(hyp.readAnnotations()[1])   #get hyp duration
        
        print("psg_start_datetime:"+str(psg_start_datetime)) #print psg start time
        print("Duration_psg:"+str(psg_file_duration))        #print psg start duration
        
        print("hyp_start_datetime:"+str(hyp_start_datetime)) #print hyp start time
        print("Duration_hyp:"+str(hyp_file_duration))        #print hyp start duration
        
# check time alignment 
# The start time and duration of PSG is compared to that of hypnogram, and differences are reported.

print('Time delta between PSG and hypnogram\n')

for psg_file, hyp_file in zip(psg_files, hyp_files): #loop through both of psg and hyp files
        
    with pyedflib.EdfReader(os.path.join(path_folder, psg_file)) as psg, \ #open psg file
         pyedflib.EdfReader(os.path.join(path_folder, hyp_file)) as hyp:   #open hyp file
        
        psg_start_datetime = psg.getStartdatetime()                 #get psg start time
        hyp_start_datetime = hyp.getStartdatetime()                 #get hyp start time
        psg_file_duration = psg.getFileDuration()                   #get psg duration
        hyp_start, hyp_duration, hyp_score = hyp.readAnnotations()  #get hyp duration
        
        if hyp_score[-1][-1]=='?':
            hyp_start = hyp_start[:-1]
            hyp_duration = hyp_duration[:-1]
            hyp_score = hyp_score[:-1]
            
        hyp_file_duration = sum(hyp_duration)
        time_delta = int(psg_file_duration-hyp_file_duration)
        
        assert (psg_start_datetime - hyp_start_datetime).total_seconds()==0, \
               'ERROR: different starting time.'
        
        if time_delta==0:
            print('ok\t', psg_file, hyp_file)
        else:
            print(time_delta, 's\t', psg_file, hyp_file, hyp_score[-1])

