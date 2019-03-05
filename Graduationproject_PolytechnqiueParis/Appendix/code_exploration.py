#Import Datasets

#Polysomnography (PSG) records are imported from 
#edf files into the dictionary `dataset`
#with record filenames as keys.

def load_sleep_edf(psg_files, hyp_files, channels_ref=None,
                   signals_as_array = False, 
                   hypno_aasm = hypno_aasm,
                   epoch_time=30, verbose=True):
     
    """Import recordings from edf databases in provided psg and hyp files.
    psg_files: list containing psg subjects.
    hyp_files: list containing hyp subjects.
     
    returns dataset containing signals: array containing the PSG channels
                                        of shape (epochs, samples, channels)
                               hypnogram: array containing hypnogram of shape (epochs, 1)
                               channels: list of PSG channels names.
                               freqs: list of PSG channels frequency.
    """
    dataset = {}
 
    if verbose:
        print('opening', len(psg_files), 'records', '\n')
             
    for psg_file, hyp_file in zip(psg_files, hyp_files):
         
        x, y = ([], [])
        freqs, channels = ([], [])
        psg_file_key = (os.path.split(psg_file)[1])[:7]
         
        with pyedflib.EdfReader(psg_file) as psg, \
             pyedflib.EdfReader(hyp_file) as hyp:
             
            hyp_start, hyp_duration, hyp_score = hyp.readAnnotations()
            hyp_valid = (hyp_score != 'Sleep stage ?')
            hyp_score = hyp_score[hyp_valid]
            hyp_duration = hyp_duration[hyp_valid]
 
            # expand sleep stages to epochs
            for duration, score in zip(hyp_duration, hyp_score):
                y += [hypno_aasm[score]]*int(duration//epoch_time)
            y = np.array(y).reshape(-1, 1)
             
            edf_channels = psg.getSignalLabels() 
            edf_freqs = psg.getSampleFrequencies()
             
            if channels_ref is None:
                channels_ref = edf_channels
             
            for idx_channel, channel in enumerate(edf_channels):
                if channel in channels_ref:
                    channels += [channel]
                    freqs += [edf_freqs[idx_channel]]
                    x += [psg.readSignal(idx_channel).reshape(-1, 1)]

            # time alignment psg vs hyp
            epochs_x = [x[channel_idx].shape[0]//int(epoch_time*freqs[channel_idx])
                        for channel_idx in range(len(freqs))]
             
            epochs_y = y.shape[0]
            epochs_valid = np.min(epochs_x+[epochs_y])
            y = y[:epochs_valid]
             
            for channel_idx, channel in enumerate(channels):
                x[channel_idx] = x[channel_idx][:epochs_valid*int(epoch_time*freqs[channel_idx]),:]
                x[channel_idx] = x[channel_idx].reshape(epochs_valid, -1)
                 
            if signals_as_array:
                x = np.stack(x, axis=2)
                 
            # add record to dataset
            dataset[psg_file_key] = {'hypnogram': y, 
                                     'signals': x, 
                                     'channels': channels,
                                     'freqs': freqs}
         
            if verbose:
                print(' '+psg_file_key, channels, freqs)
             
    if verbose:
        print('\n', 'done (', len(dataset.keys()), 'records )')
 
    return dataset

dataset = load_sleep_edf(psg_files, hyp_files, channels_ref=channels_ref) #apply load_sleep_edf function


dataset_records = sorted(dataset.keys()) #set dataset keys

record_ref = dataset_records[-1] 
dataset_record_ref = {item: dataset[record_ref][item] for item in dataset[record_ref]}
dataset_channels = dataset_record_ref['channels']
dataset_channels_short = [channels_ref_short[idx_channel]
                          for idx_channel, channel in enumerate(dataset_channels)]

# Data Pre-Processing :Several pre-processing steps are applied to the dataset prior to model training.
#Awake state event reduction:

for record in dataset: #loop through dataset records
    hyp = dataset[record]['hypnogram'] #get hypnogram
    idx_start = max(0, np.where(hyp!=0)[0][0] - awake_pre_post) 
    idx_end = min(np.where(hyp!=0)[0][-1] + awake_pre_post, hyp.shape[0])
    dataset[record]['hypnogram'] = dataset[record]['hypnogram'][idx_start:idx_end]  #keep 5 minutes\
                                                   #before/after the first/last non awake sleep state
   
    for idx_channel in range(len(dataset[record]['channels'])): #loop through psg signals\
                                                      #to adjust them accorgingly to the new hypnogram
        dataset[record]['signals'][idx_channel] = dataset[record]['signals']\
        [idx_channel][idx_start:idx_end]
                                                             #adjust psg signals to hypnograms
        
 #visualize hypnogram

fig, axs = plt.subplots(1, 2, figsize=(2*fig_w,fig_h)) ##create a figure with two subplots

plt.sca(axs[0]) #set the first subplot
hypno = dataset_record_ref['hypnogram'] #set hypnogram
plt.plot(dataset_record_ref['hypnogram'], label='pre') #plot the hypnogram 
plt.xlabel('Epochs') #add a label to the x-axis
plt.ylabel('Sleep stages') #add a label to the y-axis
plt.title('Hypnogram pre') #add a title to the subplot

plt.sca(axs[1])  #set the second subplot
hypno = dataset[record_ref]['hypnogram'] 
plt.plot(hypno, label='pre') 
plt.title('Hypnogram post') #add a title to the subplot

plt.tight_layout() #adjust the plot
fig.canvas.draw() #update the figure

# Re-sample dataset to target frequency

from scipy import signal

def gcd(a, b):
    return gcd(b, a % b) if b else a

def updown(n1, n2):
    div = gcd(n1,n2)
    return (n2//div, n1//div)

# print unique frequencies in dataset

verbose = False #set verbose to false
freqs = [] #initialize freqs list

for record in dataset: #loop through dataset
    freqs += dataset[record]['freqs'] #append frequencies to freqs list
print('(pre) unique frequencies in dataset', sorted(set(freqs)))

#Resample dataset to target frequency

for record in dataset: #loop through dataset
    for idx_channel, channel in enumerate(dataset[record]['channels']): #loop through channels
        freq = dataset[record]['freqs'][idx_channel] #set freq as a current frequency\
                                                      #corresponding to the current index
        if freq!=freq_target:
            up, down = updown(freq, freq_target) #perform down/upsampling applying updown function
            data = dataset[record]['signals'][idx_channel] #set PSG signals
            data_d = sp.signal.resample_poly(np.concatenate(data, axis=0),
                                             up, down) #build new data afterup/downsampling 
            
            dataset[record]['signals'][idx_channel] = np.array(np.split(data_d , data.shape[0]))
            dataset[record]['freqs'][idx_channel] = freq_target
            
            if verbose:
                print(record, channel, freq, '->', freq_target)

freqs = []
for record in dataset: #loop through dataset
    freqs += dataset[record]['freqs'] #fill freqs list with sampled frequencies
print('(post) unique frequencies in dataset', sorted(set(freqs))) # print unique frequencies in dataset

#Since all channels have the same sampling frequency, we need to stack psg signals into one array.

for record in sorted(dataset): #loop through dataset
    if type(dataset[record]['signals'])==type(list()):
        dataset[record]['signals']= np.stack(dataset[record]['signals'],
                                             axis=2) #join PSG arrays and save them into array.

    print('sample record\t', record)
    print('PSG channels shape:\t', dataset[record]['signals'].shape) #print psg signals shapes
    print('hypnogram shape:\t', dataset[record]['hypnogram'].shape)  #print hypnograms shapes
    
#Plot PSG channels

fig = plt.figure() #create a figure
for item_x, item_channel in zip(range(dataset[record]['signals'].shape[2]),
                                channels_ref): #loop through signals array
    data=dataset[record]['signals'][700,:200,item_x] #build data with the first 200 samples\
                                                     #of epoch 700 of signals array
    plt.plot(data,label=str(item_channel)) #plot PSG channels

plt.title('PSG channels')  #add a title to the plot
plt.xlabel('Samples')      #add a label to the x-axis
plt.ylabel('Voltage (V)')  #add a label to the y-axis
plt.legend()               #add a legend to the plot
plt.show()                 #show the figure

# Statistical metrics Exploration

def stats(dataset[record]['signals'], channels_ref = channels_ref):
    """Print minimum, maximum, mean and standard deviation along dimension 2 of array.
    signals: array with shape (epoch, sample, channel)""" 
    print('PSG'+'\t min\t\t max\t\t mean\t\t std\n')
    for item, item_channel in zip(range(dataset[record]['signals'].shape[2]), 
                                  dataset[record]['channels']): #iterate over signals array
        print(item_channel,'\t','{0:.4f}'.format(np.min(dataset[record]['signals'][:,:,item])),
              '\t','{0:.4f}'.format(np.max(dataset[record]['signals'][:,:,item])),'\t',
              '{0:.4f}'.format(np.mean(dataset[record]['signals'][:,:,item])),'\t',
              '{0:.4f}'.format(np.std(dataset[record]['signals'][:,:,item])))
        #print statistics of array x (min,max,mean,std)

for record in dataset: #loop through dataset
    print(record) #print record
    print(stats(dataset[record]['signals'])) #apply stats function to print signals metrics
       
#Show channels variability

plt.figure() #create figure
data_to_plot = [] #initialize data_to_plot as an empty list

for item, item_channel in zip(range(dataset[record]['signals'].shape[2]),
                              dataset[record]['channels']): #loop through signals and channels
    data_to_plot += [dataset[record]['signals']\
                     [150,:150,item].flatten()]
    #append to data_to_plot listdata with the first 150 samples of epoch 150 of signals array
plt.boxplot(data_to_plot, notch=True, patch_artist=True) #plot a boxplot describing channels variability
plt.xticks([1, 2, 3, 4], channels_ref_short, rotation=45) 
plt.title('PSG Channels Variability') #add a title to the plot
plt.xlabel('Channels')    #add a label to the x-axis
plt.ylabel('Voltage (V)') #add a label to the y-axis
plt.show() #show the figure

#Sroportion of sleep stages

y_name = {0: 'AWA', 1:'N1', 2:'N2', 3:'N3', 4:'REM'} #build y_name dictionary

y_keys = y_name.keys() #show keys of y_name dictionary
print('unique elements of hypnogram: {}'.format(y_keys))

proportions = [] #initialize proportions list
for value in y_name.keys(): #loop through y_name dictionary
    p = len(y[y==value])/len(y) #set sleep stage proportion
    print(y_name[value],'\t{0:.4f}'.format(p)) #print sleep stage proportions
    proportions += [float('{0:.4f}'.format(p))] #fill proportions list with sleep stages proportions
print(proportions) #print proportions list

#Visualize proportion of sleep stages

plt.figure(figsize=(5.5, 5.5), dpi=100) #create figure
pie_colors = colors = ['salmon', 'olive',  'teal', 'brown', 'sienna'] #set pie chart colors
explode = (0.045, 0, 0, 0, 0)  
plt.pie(proportions, explode=explode, labels=y_name.values(),
        colors=colors, autopct='%1.1f%%',
        startangle=80) #plot pie chart describing sleep stages distribuion
plt.title('Sleep Stages Distribution') #add title to the plot
plt.tight_layout() #adjust the plot
plt.show() #show the figure

#Visualize PSG samples per sleeping stage

fig = plt.figure(figsize=(20,3.5)) #create figure 
fontP = FontProperties()
fontP.set_size(5.5)
for idx,item in enumerate(y_name.keys()): #loop through y_name dictionary 
    A=np.where(y==item)[0]
    plt.subplot(1,len(y_name.keys()),idx+1) #create a subplot for each 
    for index, channel, color in zip(range(dataset[record]['signals'].shape[2]), 
                                     channels_ref_short,
                                     colors): #loop through signals array
        data=dataset[record]['signals'][A[2],:1000,index] #create a data containing 1000 samples of PSG
        plt.plot(data+index/3,
                 label='Channel '+str(channel),
                 color=color) #plot PSG channels for each stage
    plt.title(y_name[item]) #add a title to each subplot
    plt.xlabel('Samples') #add a label to x_axis of each subplot
    
    if idx==0:
        plt.ylabel('Voltage(V)') #add a label to y_axis

plt.legend(loc=1, prop=fontP) #add a legend to the plot
plt.tight_layout() #adjust the plot



