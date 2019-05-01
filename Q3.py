import os
import cv2
import imgaug as aug
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from pathlib import Path
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHASHSEED'] = '0'
seed=1234
# Set the numpy seed
np.random.seed(seed)
aug.seed(seed)

#load data
training_data = Path('/Users/haozhang/PycharmProjects/lab3/training')
validation_data = Path('/Users/haozhang/PycharmProjects/lab3/validation')
labels_path = Path('/Users/haozhang/PycharmProjects/lab3/monkey_labels.txt')

labels_info = []

# Read the file
lines = labels_path.read_text().strip().splitlines()[1:]
for line in lines:
    line = line.split(',')
    line = [x.strip(' \n\t\r') for x in line]
    line[3], line[4] = int(line[3]), int(line[4])
    line = tuple(line)
    labels_info.append(line)

# Convert the data into a pandas dataframe
labels_info = pd.DataFrame(labels_info, columns=['Label', 'Latin Name', 'Common Name',
                                                 'Train Images', 'Validation Images'], index=None)
labels_info.head(10)
labels_dict= {'n0':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'n5':5, 'n6':6, 'n7':7, 'n8':8, 'n9':9}
names_dict = dict(zip(labels_dict.values(), labels_info["Common Name"]))
print(names_dict)

# Creating a dataframe for the training dataset
train_df = []
for folder in os.listdir(training_data):
    imgs_path = training_data / folder
    imgs = sorted(imgs_path.glob('*.jpg'))
    for img_name in imgs:
        train_df.append((str(img_name), labels_dict[folder]))

train_df = pd.DataFrame(train_df, columns=['image', 'label'], index=None)
# shuffle the dataset
train_df = train_df.sample(frac=1.).reset_index(drop=True)

# Creating dataframe for validation data
valid_df = []
for folder in os.listdir(validation_data):
    imgs_path = validation_data / folder
    imgs = sorted(imgs_path.glob('*.jpg'))
    for img_name in imgs:
        valid_df.append((str(img_name), labels_dict[folder]))

valid_df = pd.DataFrame(valid_df, columns=['image', 'label'], index=None)
# shuffle the dataset
valid_df = valid_df.sample(frac=1.).reset_index(drop=True)

# How many samples do we have in our training and validation data?
print("Number of traininng samples: ", len(train_df))
print("Number of validation samples: ", len(valid_df))

# sneak peek of the training and validation dataframes
print("\n", train_df.head(), "\n")
print("\n", valid_df.head())
img_rows, img_cols, img_channels = 224,224,3
batch_size=8
# total number of classes in the dataset
nb_classes=10
# Augmentation sequence
seq = iaa.OneOf([
    iaa.Fliplr(),
    iaa.Affine(rotate=20),
    iaa.Multiply((1.2, 1.5))])
def data_generator(data, batch_size, is_validation_data=False):
    n = len(data)
    nb_batches = int(np.ceil(n / batch_size))
    indices = np.arange(n)
    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)
    batch_labels = np.zeros((batch_size, nb_classes), dtype=np.float32)
    while True:
        if not is_validation_data:
            np.random.shuffle(indices)

        for i in range(nb_batches):
            # get the next batch
            next_batch_indices = indices[i * batch_size:(i + 1) * batch_size]

            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(data.iloc[idx]["image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = data.iloc[idx]["label"]

                if not is_validation_data:
                    img = seq.augment_image(img)

                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                batch_data[j] = img
                batch_labels[j] = to_categorical(label, num_classes=nb_classes)

            batch_data = preprocess_input(batch_data)
            yield batch_data, batch_labels

#training data
train_data_gen = data_generator(train_df, batch_size)
# validation data
valid_data_gen = data_generator(valid_df, batch_size, is_validation_data=True)
# simple function that returns the base model
def get_base_model():
    base_model = VGG16(input_shape=(img_rows, img_cols, img_channels), weights='imagenet', include_top=True)
    return base_model

base_model = get_base_model()
base_model_output = base_model.layers[-2].output
x = Dropout(0.7,name='drop2')(base_model_output)
output = Dense(10, activation='softmax', name='fc3')(x)

model = Model(base_model.input, output)
for layer in base_model.layers[:-1]:
    layer.trainable=False

optimizer = RMSprop(0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

es = EarlyStopping(patience=10, restore_best_weights=True)

#  save model
chkpt = ModelCheckpoint(filepath="model1", save_best_only=True)

# number of training and validation steps for training and validation
nb_train_steps = int(np.ceil(len(train_df)/batch_size))
nb_valid_steps = int(np.ceil(len(valid_df)/batch_size))
nb_epochs=1
# train the model
history1 = model.fit_generator(train_data_gen,
                              epochs=nb_epochs,
                              steps_per_epoch=nb_train_steps,
                              validation_data=valid_data_gen,
                              validation_steps=nb_valid_steps,
                              callbacks=[es,chkpt])

train_acc = history1.history['acc']
valid_acc = history1.history['val_acc']
# get the loss
train_loss = history1.history['loss']
valid_loss = history1.history['val_loss']
# get the number of entries
xvalues = np.arange(len(train_acc))

# visualize
f,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(xvalues, train_loss)
ax[0].plot(xvalues, valid_loss)
ax[0].set_title("Loss curve")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'validation'])

ax[1].plot(xvalues, train_acc)
ax[1].plot(xvalues, valid_acc)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend(['train', 'validation'])

plt.show()

valid_loss, valid_acc = model.evaluate_generator(valid_data_gen, steps=nb_valid_steps)
print(f"Final validation accuracy: {valid_acc*100:.2f}%")