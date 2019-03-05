#Softmax model : Simple network with softmax output layer.
#create model
from keras import Input, Model
from keras.layers import Dense, Layer
from keras.models import Sequential

def model_softmax(input_dim=1800, output_dim=5, optimizer='adadelta'):
    """Define softmax network
    returns m: Keras model with softmax output
    """
    
    m = Sequential()
    m.add(Layer(input_shape=(input_dim,), name='input'))
    m.add(Dense(output_dim, activation='softmax', name='output'))
    
    m.compile(loss='binary_crossentropy',
              metrics=['categorical_accuracy'],
              optimizer=optimizer)
    
    m.name = 'softmax'
    
    return m

#sumarize model

model = model_softmax(input_dim, output_dim)
if model is not None:
    model.summary()

#train model    
print('model', model.name,'\n')

time_start = time()
softmax_history = model.fit_generator(generator_train, epochs=epochs)
                  
print('\ntraining for', '{0:.2f}'.format(time()-time_start), 's')

metrics_val = model.evaluate_generator(generator_val)
metrics_test = model.evaluate_generator(generator_test)

print()
for idx_metric, metric in enumerate(model.metrics_names):
    print(metric_short[metric], '\t\tval', '{0:.2f}'.format(metrics_val[idx_metric]),
          '\ttest', '{0:.2f}'.format(metrics_test[idx_metric]))
#plot categorical accuracy and loss

fig, axs = plt.subplots(1, 2, figsize=(2*fig_w,fig_h))

plt.sca(axs[0])
plt.plot(softmax_history.history['categorical_accuracy'])
plt.ylabel('Accuracy')
plt.title('Categorical Accuracy')

plt.sca(axs[1])
plt.plot(ann_history.history['loss'])
plt.ylabel('Loss')
plt.title('Loss')
plt.tight_layout()
fig.text(0.5, 0.01,'Epochs', ha='center')
fig.canvas.draw()

#ANN model: ANN model with single hidden layer with 256 ReLU units and softmax output.
#create model
def model_ann(input_dim=1800, output_dim=5, h_dim=256, optimizer='adadelta'):
    """Define shallow ANN model
    returns m: shallow ANN Keras model
    """

    m = Sequential()
    m.add(Layer(input_shape=(input_dim,), name='input'))
    m.add(Dense(h_dim, activation='relu', name='h_1'))
    m.add(Dense(output_dim, activation='softmax', name='output'))
    
    m.compile(loss='binary_crossentropy',
              metrics=['categorical_accuracy'],
              optimizer=optimizer)
    
    m.name = 'ann'
        
    return m

#sumarize model

model = model_ann(input_dim, output_dim)
if model is not None:
    model.summary()

#train model    
print('model', model.name,'\n')

time_start = time()
ann_history = model.fit_generator(generator_train, epochs=epochs)
                  
print('\ntraining for', '{0:.2f}'.format(time()-time_start), 's')

metrics_val = model.evaluate_generator(generator_val)
metrics_test = model.evaluate_generator(generator_test)

print()
for idx_metric, metric in enumerate(model.metrics_names):
    
    print(metric_short[metric], '\t\tval', '{0:.2f}'.format(metrics_val[idx_metric]),
          '\ttest', '{0:.2f}'.format(metrics_test[idx_metric]))
    
#plot categorical accuracy and loss

fig, axs = plt.subplots(1, 2, figsize=(2*fig_w,fig_h))

plt.sca(axs[0])
plt.plot(ann_history.history['categorical_accuracy'])
plt.ylabel('Accuracy')
plt.title('Categorical Accuracy')

plt.sca(axs[1])
plt.plot(ann_history.history['loss'])
plt.ylabel('Loss')
plt.title('Loss')
plt.tight_layout()
fig.text(0.5, 0.01,'Epochs', ha='center')
fig.canvas.draw()

#DNN model: DNN model with multiple hidden layers with ReLU units and softmax output.
def model_dnn(input_dim=1800, output_dim=5, h_dim=[256, 32], optimizer='adadelta'):
    """Define shallow DNN model
    returns m: shallow DNN Keras model
    """

    m = Sequential()
    m.add(Layer(input_shape=(input_dim,), name='input'))
    
    for idx_n, n in enumerate(h_dim):
        m.add(Dense(n, activation='relu', name='h_'+str(idx_n+1)))
    m.add(Dense(output_dim, activation='softmax', name='output'))
    
    m.compile(loss='binary_crossentropy',
              metrics=['categorical_accuracy'],
              optimizer=optimizer)
    
    m.name = 'dnn'
        
    return m

#sumarize model

model = model_dnn(input_dim, output_dim)
if model is not None:
    model.summary()
    
#train model    
print('model', model.name,'\n')

time_start = time()
dnn_history = model.fit_generator(generator_train, epochs=epochs)
                  
print('\ntraining for', '{0:.2f}'.format(time()-time_start), 's')

metrics_val = model.evaluate_generator(generator_val)
metrics_test = model.evaluate_generator(generator_test)

print()
for idx_metric, metric in enumerate(model.metrics_names):
    print(metric_short[metric], '\t\tval', '{0:.2f}'.format(metrics_val[idx_metric]),
          '\ttest', '{0:.2f}'.format(metrics_test[idx_metric]))
    
#plot categorical accuracy and loss

fig, axs = plt.subplots(1, 2, figsize=(2*fig_w,fig_h))
plt.sca(axs[0])
plt.plot(dnn_history.history['categorical_accuracy'])
plt.ylabel('Accuracy')
plt.title('Categorical Accuracy')

plt.sca(axs[1])
plt.plot(ann_history.history['loss'])
plt.ylabel('Loss')
plt.title('Loss')
plt.tight_layout()
fig.text(0.5, 0.01,'Epochs', ha='center')
fig.canvas.draw()

#CNN model: CNN model with multiple convolutional / max pool layers, ReLU units and softmax output.

from keras.layers import MaxPooling2D, Conv2D, Flatten, Dropout
from keras import Input, Model
from keras.layers import Dense, Layer
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

def model_cnn(input_shape=(1800,3,1), output_dim=5, 
              c_size=[32, 16, 8], k_size=[3, 3, 3], h_dim=[256, 32], d_p=[0.25, 0.25]):
    """Define CNN model
    returns model_cnn : CNN Keras model
    """

    m = Sequential()
    m.add(Layer(input_shape=input_shape, name='input'))
    
    for idx_n, n in enumerate(c_size):
        
        m.add(Conv2D(n, (k_size[idx_n], 1), padding='same', activation='relu', name='c_'+str(idx_n+1)))
        m.add(MaxPooling2D((2, 1), padding='same', name='p_'+str(idx_n+1)))
    
    m.add(Flatten(name='flatten'))
              
    for idx_n, n in enumerate(h_dim):
        m.add(Dense(n, activation='relu', name='h_'+str(idx_n+len(c_size)+1)))
        m.add(Dropout(d_p[idx_n], name='d_'+str(idx_n+len(c_size)+1)))
    
    m.add(Dense(output_dim, activation='relu', name='output'))
    sgd_lr, sgd_momentum , sgd_decay = (0.1, 0.8, 0.003)
    sgd = keras.optimizers.SGD(lr=sgd_lr,
                       momentum=sgd_momentum, 
                       decay=sgd_decay,
                       nesterov=False)
    m.compile(loss='binary_crossentropy',
              metrics=['categorical_accuracy'],
              optimizer=sgd)
              
    m.name = 'cnn'
    
    return m

#summarize model
model = model_cnn(input_shape, output_dim)
if model is not None:
    model.summary()

#train model

print('model', model.name,'\n')

time_start = time()
cnn_history = model.fit_generator(generator_train_cnn, epochs=epochs)
                  
print('\ntraining for', '{0:.2f}'.format(time()-time_start), 's')

metrics_val = model.evaluate_generator(generator_val_cnn)
metrics_test = model.evaluate_generator(generator_test_cnn)

print()

for idx_metric, metric in enumerate(model.metrics_names):
    print(metric_short[metric], '\t\tval', '{0:.2f}'.format(metrics_val[idx_metric]),
          '\ttest', '{0:.2f}'.format(metrics_test[idx_metric]))
#plot categorical accuracy and loss   
fig, axs = plt.subplots(1, 2, figsize=(2*fig_w,fig_h))
plt.sca(axs[0])
plt.plot(cnn_history.history['categorical_accuracy'])
plt.ylabel('Accuracy')
plt.title('Categorical Accuracy')

plt.sca(axs[1])
plt.plot(cnn_history.history['loss'])
plt.ylabel('Loss')
plt.title('Loss')
plt.tight_layout()
fig.text(0.5, 0.01,'Epochs', ha='center')
fig.canvas.draw()

#Cross validation Performance

def cross_validation(dataset,
                     model_ref,
                     tensor=False,
                     channel_mask=None,
                     epochs=epochs, 
                     batch_size=batch_size,
                     n_val=300,
                     verbose=True):
    """Leave-one-out cross validation scheme at database level
    dataset: dataset containing records with PSG data and hypnograms
    hyp_files: list containing hypnograms of edf databases
    model_ref: Keras model
    epochs: number of training epochs
    batch_size : number of mini-batch samples
    verbose: print intermediate results otherwise no output
    
    returns models: list with trained Keras models
            metrics: list with validation and test accuracy"""
        
    if verbose:
        print('model', model_ref().name, '-', len(dataset), 'fold cross-validation\n')
    
    models, metrics = ([], [])
    
    for idx_record, record in enumerate(dataset_records):
        records_test = [record]
        records_train = [key for key in dataset_records if key not in records_test]

        epochs_train, epochs_val = epochs_train_val(dataset, records_train, n_val)

        generator_train = PSGSequence(dataset, 
                                      records=records_train, 
                                      epochs_selected=epochs_train, 
                                      batch_size=batch_size,
                                      tensor=tensor,
                                      channel_mask=channel_mask)
        generator_val = PSGSequence(dataset, 
                                    records=records_train, 
                                    epochs_selected=epochs_val,
                                    batch_size=batch_size,
                                    tensor=tensor,
                                    channel_mask=channel_mask)
        generator_test = PSGSequence(dataset, 
                                     records=records_test,
                                     batch_size=batch_size,
                                     tensor=tensor,
                                     channel_mask=channel_mask)
        
        if tensor:
            input_shape_dim = generator_train.x_batch_shape[1:]
        else:
            input_shape_dim = generator_train.x_batch_shape[1]
            
        output_dim = generator_train.y_batch_shape[1]

        model = model_ref(input_shape_dim, output_dim)
        
        history = model.fit_generator(generator_train, epochs=epochs, verbose=False)

        metrics_train = [history.history[metric] for metric in model.metrics_names]
        metrics_val = model.evaluate_generator(generator_val)
        metrics_test = model.evaluate_generator(generator_test)

        models += [model]
        
        metrics += [[metrics_train, metrics_val, metrics_test]]      
        if verbose:
            if idx_record==0:
                print('fold', '\t', 'acc train', '\t', 'acc val', '\t', 'acc test\n')
            print(idx_record+1, 
                  '\t', '{0:.2f}'.format(metrics_train[1][-1]),
                  '\t\t', '{0:.2f}'.format(metrics_val[1]),
                  '\t\t', '{0:.2f}'.format(metrics_test[1]))
            
    # shape (fold, set, metric)
    metrics = np.array(metrics)
    
    if verbose:
        print('\n\t', '{0:.2f} ({1:.2f})'.format(np.mean(metrics[:,0,1][-1]), 
                                                 np.std(metrics[:,0,1][-1])),
              '\t', '{0:.2f} ({1:.2f})'.format(np.mean(metrics[:,1,1]), 
                                               np.std(metrics[:,1,1])),
              '\t', '{0:.2f} ({1:.2f})'.format(np.mean(metrics[:,2,1]), 
                                               np.std(metrics[:,2,1]))) 
    return models, metrics

#validate all models:

for model, idx in zip(all_models.keys(),range(4)):
    if model='CNN':
        models, metrics = cross_validation(dataset,
                                       all_models[model],
                                       tensor=True,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=True)
    else:
        models, metrics = cross_validation(dataset,
                                       all_models[model],
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=True)
        
        
#plot loss and accuracy metrics of all models
fig = plt.figure(figsize=(20,3.5))

x_range = np.arange(epochs)+1

for model, idx in zip(all_models.keys(),range(4)):
    if model=='CNN':
        models, metrics = cross_validation(dataset,
                                       all_models[model],
                                       tensor=True,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=False)
    else:
        models, metrics = cross_validation(dataset,
                                       all_models[model],
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=False)
        
    plt.subplot(1,len(all_models),idx+1)
    
    accuracy = []
    
    for item in metrics:
        accuracy += [item[0][1]]
    
    for item in accuracy:
        plt.plot(x_range, item, color='khaki')

    plt.plot(x_range, np.mean(accuracy,axis=0), color='olive', label = 'mean')
    plt.title('Train accuracy '+model)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    