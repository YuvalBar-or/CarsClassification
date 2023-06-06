import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.metrics import Precision, Recall
from keras.optimizers import RMSprop
from keras.optimizers.legacy import RMSprop as LegacyRMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.applications import DenseNet121

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Define the train and validation directories
train_dir = 'training'
validation_dir = 'validation'

def prepare_data(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=1,
            class_mode='binary',
            shuffle=False)

    return train_generator, validation_generator

def create_model():
    base_model = DenseNet121(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, x)

    optimizer = RMSprop(learning_rate=1e-4)

    if isinstance(optimizer, RMSprop) and 'M1' in os.uname().machine:
        optimizer = LegacyRMSprop(learning_rate=1e-4)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', Precision(), Recall()]
    )

    return model

def train_model(model, train_generator, validation_generator, epochs=2):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=2
    )
    return history

def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history['precision']
    val_precision = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, precision, 'bo', label='Training precision')
    plt.plot(epochs, val_precision, 'b', label='Validation precision')
    plt.title('Training and validation precision')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, recall, 'bo', label='Training recall')
    plt.plot(epochs, val_recall, 'b', label='Validation recall')
    plt.title('Training and validation recall')
    plt.legend()

    plt.show()

def classify_images(model, validation_generator):
    class_labels = {0: 'damage', 1: 'whole'}  # Mapping of class indices to labels

    predictions = model.predict(validation_generator)
    predicted_labels = np.round(predictions).flatten()
    predicted_classes = [class_labels[int(label)] for label in predicted_labels]

    filenames = validation_generator.filenames
    for filename, predicted_class in zip(filenames, predicted_classes):
        print(f"Image: {filename} | Predicted label: {predicted_class}")

def main():
    train_generator, validation_generator = prepare_data(train_dir, validation_dir)
    model = create_model()
    history = train_model(model, train_generator, validation_generator)
    model.save("classifier.h5")
    plot_metrics(history)
    classify_images(model, validation_generator)


if __name__ == '__main__':
    main()
