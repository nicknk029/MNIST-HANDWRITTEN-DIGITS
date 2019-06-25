# IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

# LOADING THE DATASET

( x_train, y_train ), ( x_test, y_test ) = mnist.load_data()

print( 'Train size:', x_train.shape )
print( 'Test size:', x_test.shape )

# IMAGE PLOT WITH LABEL 

plt.figure( figsize= ( 15, 15 ) )

for i in range( 15 ) :   
    plt.subplot( 3, 5, i+1 )
    plt.imshow( x_train[i], cmap='gray' )
    plt.title( y_train[i], fontsize = 20 )
    
# RESHAPING

X_train = x_train.reshape( x_train.shape[0], 1, 28, 28 )
X_test = x_test.reshape( x_test.shape[0], 1, 28, 28 )

# CONVERT TO FLOAT SO THAT WE WILL HAVE DECIMAL POINTS AFTER DIVISION

X_train = X_train.astype( 'float32' )
X_test = X_test.astype( 'float32' )
print( 'X_train after reshape:', X_train.shape )

# NORMALIZING- 255 IS THE HIGHEST VALUE

X_train = X_train / 255
X_test = X_test / 255
print( np.max( X_train[0] ) )
print( np.min( X_train[0] ) )


# CONVERT Y VALUES TO CATEGORICAL

y_train = np_utils.to_categorical( y_train )
y_test = np_utils.to_categorical( y_test )
num_classes = y_test.shape[1]

# CNN WITH 3 LAYERS

model = Sequential()
model.add( Conv2D( 32, ( 3, 3 ), input_shape = ( 1, 28, 28 ), activation = 'relu', data_format = 'channels_first') )
model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) ) 
model.add( Conv2D( 64, ( 2, 2 ), activation = 'relu' ) )
model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) ) 
model.add( Conv2D( 10, ( 2, 2 ), activation = 'relu' ) )
model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) ) 
model.add( Dropout( 0.3 ) )
model.add( Flatten() )
model.add( Dense( 128, activation = 'relu' ) )
model.add( Dense( 20, activation = 'relu' ) )
model.add( Dense( 10, activation = 'softmax' ) )
model.compile( optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')
model.fit( X_train, y_train, validation_data = ( X_test, y_test ), epochs = 5, batch_size = 200, verbose = 2)
