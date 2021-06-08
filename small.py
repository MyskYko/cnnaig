import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda


###### MODIFY FROM HERE #####

maxshamt = 10

def getmodel():
    # Define the model
    inputs = Input((32, 32, 3))
    x = inputs
    x = Conv2D(3, (3, 3), strides=(3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Conv2D(3, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

ntrainepoch = 1

########## TO HERE ##########

defshamt = 3

# Set GPU memory allocation to dynamic
try:
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('no gpu loaded')
    
# Set seed to replicate results
tf.random.set_seed(42)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert 8-bit images to 3-bit and bring them to [0, 1] range
x_train = np.floor(x_train / 32) / 8
x_test = np.floor(x_test / 32) / 8

# Convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Do not use test set for now
x_test = x_train[40000:]
y_test = y_train[40000:]
x_train = x_train[:40000]
y_train = y_train[:40000]

# Split training data into training and validation sets
train_size = int(x_train.shape[0] * 0.8)
trainX = x_train[:train_size]
trainY = y_train[:train_size]
valX = x_train[train_size:]
valY = y_train[train_size:]

# Define a data generator on the training images to prevent overfitting
batch_size = 64
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = datagen.flow(trainX, trainY, batch_size=batch_size)
steps_per_epoch = int(trainX.shape[0] / batch_size)



def trainmodel(model, filepath):
    print(model.summary())

    # Use an exponentially decaying learning rate for training
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-3,
        decay_steps=5000,
        decay_rate=0.9,
        staircase=False)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'])

    # Save the best validation set model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        verbose=1,
        monitor='val_acc',
    mode='max',
    save_best_only=True)
    
    # Train the network
    history = model.fit(it_train, validation_data=(valX, valY), epochs=ntrainepoch, callbacks=[model_checkpoint_callback])

    # Compute accuracy on the test set
    model.load_weights(filepath)
    [loss, acc] = model.evaluate(x_test, y_test)
    print('Loss:', loss)
    print('Accuracy:', acc * 100)
    
    # Get the current learning rate
    print(model.optimizer._decayed_lr(tf.float32).numpy())
    
    # Plot the training vs validation accuracy
    plt.clf()
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    plt.legend()
    plt.savefig(f'{filepath}.curve.png')
    
    # Plot the training vs validation loss
    #plt.plot(history.history['loss'], color='blue', label='train')
    #plt.plot(history.history['val_loss'], color='orange', label='test')
    #plt.legend()

def evalmodel(model):
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    [loss, acc] = model.evaluate(x_test, y_test)
    print('Loss:', loss)
    print('Accuracy:', acc * 100)

def quantizeconv(weights):
    w0 = weights[0]
    w0 = np.log2(abs(w0))
    w0 = np.floor(w0).astype('int32')
    w0 = np.maximum(w0, np.max(w0) - maxshamt)
    w0 = 2.0 ** w0
    w0 = np.where(weights[0] > 0, w0, -w0)
    '''
    print(w0[0])
    print(np.max(w0))
    print(np.min(w0))
    '''
    w1 = weights[1]
    w1 = np.clip(w1, -0.5, 0.5)
    w1 = w1 * (1 << (defshamt + maxshamt))
    w1 = np.floor(w1)
    w1 = w1 / (1 << (defshamt + maxshamt))
    '''
    print(w1)
    print(np.max(w1))
    print(np.min(w1))
    '''
    return [w0, w1]

def ClipByVal(a):
    return tf.where(a == 0, a, tf.divide(tf.clip_by_value(tf.floor(tf.multiply(a, 1 << defshamt)), -32, 31), 1 << defshamt))
def ClipLayer():
    return Lambda(ClipByVal)

def insertlayer(model, layer_id, new_layer):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)
    model = Model(inputs=layers[0].input, outputs=x)
    model.save("_tmp_model")
    model = tf.keras.models.load_model("_tmp_model")    
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help="do training")
parser.add_argument('--quant', action='store_true', help="do quantization")
parser.add_argument('--verilog', action='store_true', help="gen verilog")
args = parser.parse_args()

checkpoint_filepath = 'small_float.h5'
if args.train:
    model = getmodel()
    trainmodel(model, checkpoint_filepath)

quantized_filepath = 'small_quantized.h5'
if args.quant:
    model = getmodel()
    layerid = 0
    while layerid < len(model.layers):
        layername = model.layers[layerid].__class__.__name__
        if layername == 'Conv2D' or layername == 'Dense':
            if layerid < len(model.layers) - 1: # not last layer
                model = insertlayer(model, layerid+1, ClipLayer())
            model.load_weights(checkpoint_filepath)
            weights = model.layers[layerid].get_weights()
            weights = quantizeconv(weights)
            model.layers[layerid].set_weights(weights)
            for i in range(layerid+1):
                model.layers[i].trainable = False
            evalmodel(model)
            if layerid == len(model.layers) - 1:
                print(model.summary())
                break
            checkpoint_filepath = f'small_quant{layerid}.h5'
            trainmodel(model, checkpoint_filepath)
        layerid += 1
    model.save_weights(quantized_filepath)
    print(model.get_weights())



def fptest(model):
    np.random.seed(41)
    testin = np.random.randint(8, size=(32, 32, 3))
    
    image = testin / 8
    image = np.expand_dims(image, axis=0)
    inp = model.input
    partial_model = Model(model.inputs, model.layers[1].output)
    outputs = [layer.output for layer in model.layers]
    functors = [Model(inputs=inp, outputs=out) for out in outputs]
    eximages = [func([image], training=False)[0] for func in functors]
    
    from fpsimulate import fpsimulate
    image = testin / 8
    images = fpsimulate(model.layers, image)
    
    for i, image in enumerate(images):
        print(f'layer {i}')
        print(f'diff num: {np.count_nonzero(image != eximages[i])} / {image.size}')
        a = np.where(eximages[i] != 0, image / eximages[i], 1)
        print(f'ratio: {a.min()} ~ {a.max()}')
    print(np.argmax(images[-1]) == np.argmax(eximages[-1]))

def fpinttest(model):
    np.random.seed(41)
    testin = np.random.randint(8, size=(32, 32, 3))
    
    from fpsimulate import fpsimulate
    image = testin / 8
    fpimages = fpsimulate(model.layers, image, False, False)
    
    from intsimulate import intsimulate
    image = (testin, 3)
    images = intsimulate(model.layers, image)
    
    for i, image in enumerate(images):
        print(f'layer {i}')
        print(f'diff num: {np.count_nonzero(image[0] / (1 << image[1]) != fpimages[i])} / {image[0].size}')
        a = np.where(fpimages[i] != 0, image[0] / (1 << image[1]) / fpimages[i], 1)
        print(f'ratio: {a.min()} ~ {a.max()}')


def intverilogtest(model):
    np.random.seed(41)
    testin = np.random.randint(8, size=(32, 32, 3))
    
    from intsimulate import intsimulate
    image = (testin, 3)
    intimages = intsimulate(model.layers, image)
        
    from genverilog import genverilog
    import os
    import datetime
    dirname = 'test_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir(dirname)
    os.chdir(dirname)
    image = (np.shape(testin), 3, 3)
    images = genverilog(model.layers, image, 'test')

    infile = open('inp.txt', mode='w')
    infile.write(f'{testin.size * 3} 1\n')
    for i in range(testin.size * 3):
        infile.write(f'in[{i}] ')
    infile.write('\n')
    for i in range(np.shape(testin)[0]):
        for j in range(np.shape(testin)[1]):
            for k in range(np.shape(testin)[2]):
                l = testin[i][j][k]
                infile.write(f'{l%2} {l//2%2} {l//4%2} ')
    infile.seek(infile.tell() - 1, os.SEEK_SET)
    infile.write('\n')
    infile.close()

    outfile = open('out.txt', mode='w')
    outnum = intimages[-1][0].size
    bitwidth = images[-1][2]
    outfile.write(f'{testin.size * 3} {outnum * bitwidth} 1\n')
    for i in range(testin.size * 3):
        outfile.write(f'in[{i}] ')
    for i in range(outnum * bitwidth):
        outfile.write(f'out[{i}] ')
    outfile.seek(outfile.tell() - 1, os.SEEK_SET)
    outfile.write('\n')
    for i in range(np.shape(testin)[0]):
        for j in range(np.shape(testin)[1]):
            for k in range(np.shape(testin)[2]):
                l = testin[i][j][k]
                outfile.write(f'{l%2} {l//2%2} {l//4%2} ')
    oimage = intimages[-1][0].flatten()
    for i in range(oimage.size):
        tmp = oimage[i]
        for l in range(bitwidth):
            outfile.write(f'{tmp%2} ')
            tmp //= 2
    outfile.seek(outfile.tell() - 1, os.SEEK_SET)
    outfile.write('\n')
    outfile.close()

    cmd = '~/yosys/yosys -p \"read_verilog test*.v; synth -auto-top; write_verilog a.v; flatten; aigmap; write_blif a.blif\"'
    if os.system(cmd):
        print('yosys failed')
        exit(1)
    cmd = '~/abc/abc -c \"read a.blif; strash; print_stats; dc2; print_stats; dc2; print_stats; dc2; print_stats; dc2; print_stats; dc2; print_stats; write_blif b.blif\"'
    if os.system(cmd):
        print('abc failed')
        exit(1)
    cmd = '../cadex patsim b.blif inp.txt out2.txt'
    if os.system(cmd):
        print('cadex failed')
        exit(1)
    cmd = 'diff out.txt out2.txt'
    if os.system(cmd):
        print('diff failed')
        exit(1)

    os.chdir('..')
    
    
if args.verilog:    
    model = getmodel()
    for layerid in range(len(model.layers)-1):
        layername = model.layers[layerid].__class__.__name__
        if layername == 'Conv2D' or layername == 'Dense':
            model = insertlayer(model, layerid+1, ClipLayer())
    model.load_weights(quantized_filepath)

    print(model.summary())
    #fptest(model)
    #fpinttest(model)
    intverilogtest(model)



