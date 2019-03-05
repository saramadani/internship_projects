#    Prepare data for training
#create to_input function to convert PSG channels into vector representation\
#with optional selection of channels by boolean array.
def to_input(u, tensor=True, channel_mask=None):
    """Convert data array to shape (batch, data).
    u: array with shape (batch, samples, channels)
    channel_mask: boolean mask of channels to include
    
    returns x_out: array x with shape (batch, samples*selected_channels) if tensor is False
                   array x with shape (batch, samples, selected_channels, 1) if tensor is True"""
    if channel_mask:
        u_out = u[:,:, channel_mask]
    else:
        u_out = u
    
    if tensor:
        u_out = u_out.reshape((u_out.shape[0], u_out.shape[1], u_out.shape[2], 1))
    else:
        u_out = u_out.reshape((u_out.shape[0], u_out.shape[1]*u_out.shape[2]))
    return u_out

#create to_input function to convert sleep scores into `one-hot-encoding`\
#i.e. score `0` to `[1 0 0 0 0]`, score `1` to  `[0 1 0 0 0]`, etc.

def to_output(u, num_classes=5):
    """Convert label array to one-hot-encoding with shape (batch, binary_label).
    u: label array with shape (batch, 1)
    
    returns: u_out (array with shape (batch, binary_label))"""
    
    #convert a float class array to binary class array
    u_out = keras.utils.to_categorical(u,num_classes=num_classes)
    return u_out 

#Perform Data Generator

class PSGSequence(Sequence):
    def __init__(self, dataset, records=None, epochs_selected=None, batch_size=512,
                 tensor=False, channel_mask=None, process_input=to_input, process_output=to_output):
    
        self.dataset = dataset
        self.records = records
        self.epochs_selected = epochs_selected
        self.batch_size = batch_size
        self.tensor = tensor
        self.channel_mask = channel_mask
        self.process_input = process_input
        self.process_output = process_output

        if self.records is None:
            self.records = list(self.dataset.keys())
            self.epochs_selected = None
        if self.epochs_selected is None:
            self.epochs_selected = [np.array([True]*self.dataset[record]['hypnogram'].shape[0])
                                    for record in self.records]

        self.record_epoch = []
        for idx_record, record in enumerate(self.records):
            selected = self.epochs_selected[idx_record]
            for epoch in np.arange(len(selected))[selected]:
                self.record_epoch += [[record, epoch]]
                                
        self.on_epoch_end()
        self.epochs_n = len(self.record_epoch)
        
        x_batch, y_batch = self.__getitem__(0)
        
        self.x_batch_shape = x_batch.shape
        self.y_batch_shape = y_batch.shape
        
    def on_epoch_end(self):
        random.shuffle(self.record_epoch)
        
    def __len__(self):
        return int(np.ceil(self.epochs_n / float(self.batch_size)))
    
    def __getitem__(self, idx):
        record_epoch_batch =  self.record_epoch[idx * self.batch_size:(idx + 1) * self.batch_size]
                                
        x_batch, y_batch = ([],[])
        
        for record, epoch in record_epoch_batch:
            
            x_batch += [self.dataset[record]['signals'][epoch]]
            y_batch += [self.dataset[record]['hypnogram'][epoch]]
                   
        return self.process_input(np.array(x_batch), tensor=self.tensor, 
                                  channel_mask=self.channel_mask), 
                                  self.process_output(np.array(y_batch))
    
#create epochs_train_val to split dataset into train and validation sets 

def epochs_train_val(dataset, records, size): 
    
    epochs_train, epochs_val = ([], [])

    for record in records:

        epochs_selected = np.array([False]*dataset[record]['hypnogram'].shape[0])
        idx = np.random.choice(dataset[record]['hypnogram'].shape[0], size=size, replace=False) 
        epochs_selected[idx] = True

        epochs_val += [epochs_selected[epochs_selected]]
        epochs_train += [epochs_selected[~epochs_selected]]
        
    return epochs_train, epochs_val

records_test = [dataset_records[-1]]
records_train = dataset_records[:-1]

#Split dataset into train and val sets, the validation set contains 300 epochs from each record.
epochs_train, epochs_val = epochs_train_val(dataset, records_train, n_val) 

#Generate train, test and val sets for softmax, ANN and DNN models
generator_test = PSGSequence(dataset, 
                             records=records_test,
                             batch_size=batch_size) #generate test set
generator_val = PSGSequence(dataset, 
                            records=records_train, 
                            epochs_selected=epochs_val,
                            batch_size=batch_size) #generate val set
generator_train = PSGSequence(dataset, 
                              records=records_train, 
                              epochs_selected=epochs_train, 
                              batch_size=batch_size) #generate train set

input_dim = generator_train.x_batch_shape[1]
output_dim = generator_train.y_batch_shape[1]

print('generator test', generator_test.x_batch_shape)
print('generator train', generator_train.x_batch_shape)
print('generator val', generator_val.x_batch_shape)

#Generate train, test and val sets for CNN model

generator_train_cnn = PSGSequence(dataset, 
                                  records=records_train, 
                                  epochs_selected=epochs_train, 
                                  batch_size=batch_size,
                                  tensor=True)  #generate test set
generator_val_cnn = PSGSequence(dataset, 
                                records=records_train, 
                                epochs_selected=epochs_val,
                                batch_size=batch_size,
                                tensor=True)  #generate val set
generator_test_cnn = PSGSequence(dataset, 
                                 records=records_test,
                                 batch_size=batch_size,
                                 tensor=True)   #generate train set

input_shape = generator_train_cnn.x_batch_shape[1:]
output_dim = generator_train_cnn.y_batch_shape[1]

print('generator test', generator_test_cnn.x_batch_shape)
print('generator train', generator_train_cnn.x_batch_shape)
print('generator val', generator_val_cnn.x_batch_shape)
