# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
import plotly.io as pio
pio.renderers.default='browser'

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint



def create_generators_without_augmentation(train_images_path,valid_images_path,test_images_path):
    train_datagen = ImageDataGenerator(rescale=1./255.)

    valid_datagen = ImageDataGenerator(rescale=1./255.)
    
    test_datagen = ImageDataGenerator(rescale=1./255.)


    train_generator = train_datagen.flow_from_directory(directory=train_images_path,
                                                        target_size=(50, 50),
                                                        batch_size=32,
                                                        class_mode='categorical')

    valid_generator = valid_datagen.flow_from_directory(directory=valid_images_path,
                                                        target_size=(50, 50),
                                                        batch_size=32,
                                                        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(directory=test_images_path,
                                                      target_size=(50, 50),
                                                      batch_size=32,
                                                      class_mode='categorical',
                                                      shuffle=False)
    
    return train_generator,valid_generator,test_generator


def build_model():
    
    model = Sequential()

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=3, activation='softmax'))
    
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
    

def create_model_checkpoint(path_to_file_with_best_weights):
    checkpoint = ModelCheckpoint(filepath=path_to_file_with_best_weights, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    return checkpoint

def train_model(train_generator,steps_per_epoch,epochs,valid_generator,validation_steps,checkpoint,model):
    
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=valid_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[checkpoint])
    
    return history
    
def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Accuracy vs. Val Accuracy', xaxis_title='Epoki', yaxis_title='Accuracy', yaxis_type='log')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki', yaxis_title='Loss', yaxis_type='log')
    fig.show()
    

def load_best_weights(path_to_file_with_best_weights,model):
    
    model.load_weights(path_to_file_with_best_weights)
    
    return model

def plot_confusion_matrix(cm,classes):
    # Mulitclass classification, 3 classes
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=classes, index=classes[::-1])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=800, height=800, title='Confusion Matrix', font_size=16)
    fig.show()

def check_model(model,test_generator):
    y_prob = model.predict_generator(test_generator, test_generator.samples)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes
    print(test_generator.class_indices)
    classes = list(test_generator.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm,classes)
    print('Accuracy = ',accuracy_score(y_true,y_pred))
    
    
    
    

def main():
    train_generator,valid_generator,test_generator = create_generators_without_augmentation('./Data/done_images/train',
                                                                                          './Data/done_images/valid',
                                                                                            './Data/done_images/test')
    batch_size = 32
    steps_per_epoch = 3259 // batch_size
    validation_steps = 406 // batch_size
    
    model = build_model()
    checkpoint = create_model_checkpoint('best_model_weights.hdf5')
    
    history = train_model(train_generator,steps_per_epoch,30,valid_generator,validation_steps,checkpoint,model)
    
    plot_hist(history)
    
    model = load_best_weights('best_model_weights.hdf5',model)
    
    check_model(model,test_generator)
    
    return 0

main()
    
    
    
    
    
    