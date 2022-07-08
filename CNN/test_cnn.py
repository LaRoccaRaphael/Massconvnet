import os
import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# dataloader
def bin_ms(spec,mass_range,tol):
    
    vec_bin = np.arange(mass_range[0],mass_range[1],tol)
    bin_spec = np.zeros((len(vec_bin)))
    
    for i in range(0,len(vec_bin)):
        bin_i = vec_bin[i]
        index = (spec[:,0]  >= bin_i) & (spec[:,0]  < (bin_i+tol))
        bin_spec[i] = np.sum(spec[index ,2])
    
    return bin_spec

def load_spectra(file,mass_range,tol,df):

    mass_vec = np.arange(mass_range[0],mass_range[1],tol)

    mat_bin_vec = np.zeros((np.shape(df)[0],len(mass_vec)))
    labes_vec = []

    ct = 0
    for index, row in df.iterrows():
        #print(index, row['MSI name'], row['MSI pixel id'])
        #labes_vec.append(row['Annotations 2'])
        path = file + row['MSI name'] +"/spec_" + str(row['MSI pixel id']) + ".npy"

        spec = np.load(path)
        bin_spec = bin_ms(spec,mass_range,tol)
        mat_bin_vec[ct,:] = bin_spec
        print(ct)
        ct = ct +1
    mat_bin_vec = np.transpose(mat_bin_vec)
    return(mat_bin_vec)


## Model variant_Lecun (model 1 : 4 layers)
def build_model_lecun():
    model = Sequential([
        Conv1D(filters=6, kernel_size=21, strides=1, padding='same', activation='relu', input_shape= ms_input_shape,
               kernel_initializer=keras.initializers.he_normal()),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=16, kernel_size=5, strides=1, padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84),
        Dense(nb_classes, activation='sigmoid') # or Activation('softmax')
    ])
    return model


## Model variant_LeNet (model 2: 5 layers)
def build_model_lenet():
    model = Sequential([
        Conv1D(filters=16, kernel_size=21, strides=1, padding='same', input_shape= ms_input_shape,
               kernel_initializer=keras.initializers.he_normal()),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=32, kernel_size=11, strides=1, padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(filters=64, kernel_size=5, strides=1, padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(2050, activation='relu'),
        Dropout(0.5),
        Dense(nb_classes, activation='sigmoid') # or Activation('softmax')
    ])
    return model


df = pd.read_csv("/media/USB/DL_MASS/Dataset/METASPACE_IM/Annot_table.csv",sep=",", header=0)
sub_df = df.loc[df['train random']==True]

file = "/media/USB/DL_MASS/Dataset/METASPACE_IM/MSI/centroid_data/param_1/"

# to change according to the sample
mass_range = [200,1400]
tol = 0.1
nb_classes = 2

mat_bin_vec = load_spectra(file,mass_range,tol,df)
np.save("/media/USB/DL_MASS/Dataset/METASPACE_IM/MSI/centroid_data/mat_bin_vec", mat_bin_vec)


mat_bin_vec = np.load("/media/USB/DL_MASS/Dataset/tof_mat_bin_vec_train.npy")


mat_data = minmax_scale(np.log(mat_bin_vec+1), axis=0, feature_range=(0, 1))
mat_data = mat_data.astype("float32")
data = mat_data
y_train = sub_df.loc[sub_df['train'] ==True,"Annotations"]
y_train_conv = to_categorical(y_train,num_classes=nb_classes)
x_train = np.transpose(data[:,sub_df['train'] ==True])
x_train_conv = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))


# model training
history = model.fit(x=x_train_conv, y=y_train_conv, batch_size=254, verbose=1, epochs=20, validation_split=0.2, callbacks=[earlyStopping, reduce_lr])


# model building
ms_input_shape = (len(np.arange(mass_range[0],mass_range[1],tol)),    1)


model = build_model_lecun()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),metrics=['accuracy']) # or categorical_crossentropy
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.0000001)
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')


# evaluation
sub_df = df.loc[df['train']==False]
mat_bin_vec = np.load("/media/USB/DL_MASS/Dataset/tof_mat_bin_vec_test.npy")


mat_data = minmax_scale(np.log(mat_bin_vec+1), axis=0, feature_range=(0, 1))
mat_data = mat_data.astype("float32")

data = mat_data

y_test = sub_df.loc[sub_df['train'] ==False,"Annotations"]
y_test_conv = to_categorical(y_test,num_classes=nb_classes)
x_test = np.transpose(data[:,sub_df['train'] ==False])
x_test_conv = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

loss, acc = model.evaluate(x_test_conv, y_test_conv, verbose=1)
print('final accuracy: tol ' +str(tol) +" modellecun " + "sub"+str(0)+" accuracy",acc)
y_classes = model.predict_classes(x_test_conv, verbose=1)
matrix = confusion_matrix(y_test, y_classes)
print(matrix)
