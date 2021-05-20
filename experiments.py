import random

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import dataset_split as ds
import os
import cv2 as cv
import pandas as pd

BATCH_SIZE = 16

labels = []
data = []

PATH_TO_REAL_DATASET='Datasets/NaturalFaces/Images/'
PATH_TO_ARTIFICIAL_DATASET='Datasets/ArtificialFaces'
files = np.array(os.listdir(PATH_TO_ARTIFICIAL_DATASET))
random.shuffle(files)

NUMBER_OF_FILES = len(files[:40])

print("Extracting each data into respective label folders....")
for idx, path in enumerate(files[:40]):
    label =' '
    if(path.endswith('Mask.jpg')):
        label = 'withmask'
    else:
        label = "withoutmask"
    #pre_str = path.split('.')[0].split('_')[-2:]
    #label = pre_str[0]+pre_str[1]
    face = cv.imread(os.path.join(PATH_TO_ARTIFICIAL_DATASET, path))
    face = cv.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    labels.append(label)
    data.append(face)

data = np.array(data, dtype="float32")
labels = np.array(labels)
lb = LabelEncoder()
labels = lb.fit_transform(labels)

aug = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

#docelowo na wejsciu dwa zbiory, jeden z rzeczywistymi maseczkami drugi z sztucznymi

rs = 10
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state =rs)
true_labels = []
NUM_EPOCHS = 1
NUM_UNFREEZED_LAYERS = 1

#macierz z wynikami |WARSTWA|FOLD|ETYKIETA|PREDYKCJA|
predictions = np.zeros((NUM_UNFREEZED_LAYERS, n_splits * n_repeats, NUMBER_OF_FILES // 4), dtype=int)
true_labels = np.zeros((NUM_UNFREEZED_LAYERS, n_splits * n_repeats, NUMBER_OF_FILES // 4), dtype=int)


for index in range(0,NUM_UNFREEZED_LAYERS):
    print(f"{index}")
    for fold_id, (train_index, test_index) in enumerate(rskf.split(data, labels)):

        print(f"Fold Number: {fold_id}")
        #podział na zbio walidacjny i testowy w proporcjach 50|50
        X_val, X_test, y_val, y_test = train_test_split(data[test_index], labels[test_index],
                                                               test_size=0.5, random_state=rs)

        #labels = to_categorical(labels, dtype="int32")
        X_train = data[train_index]
        Y_train, Y_test, Y_val = to_categorical(labels[train_index], dtype="int32"),\
                          to_categorical(labels[y_test], dtype="int32"),\
                          to_categorical(labels[y_val], dtype="int32")
        #Y_train, Y_test = labels[train_index], labels[test_index]
        baseModel = MobileNetV2(weights="imagenet", include_top=False,
                                input_tensor=Input(shape=(224, 224, 3)))
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="sigmoid")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        #Tu odmrożonie warstw w zależności od parametru
        for layer in baseModel.layers[:index]:
            layer.trainable = False

        print("[INFO] compiling model...")
        opt = Adam(lr=1e-4, decay=1e-4 / 10)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

        csv_logger = CSVLogger('training_log_1.csv',append=True)
        print("[INFO] training head...")
        H = model.fit(
            aug.flow(X_train, Y_train, batch_size=16),
            steps_per_epoch=len(X_train) // 16,
            validation_data=(X_val, Y_val),
            validation_steps=len(X_val) // 16,
            callbacks=[csv_logger],
            epochs=NUM_EPOCHS)

        predicts = model.predict(X_test)
        predicts_labels = np.argmax(predicts, axis=-1)

        for i in range(len(predicts_labels)):
            true_labels[index, fold_id, i] = labels[y_test][i]
            predictions[index, fold_id, i] = predicts_labels[i]

        print(accuracy_score(y_test, predicts_labels))
        model.save(f"model_{index}.model", save_format="h5")






# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 10), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, 10), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig('second_traning_plot.png')
#
# print(f"Mean of  {np.mean(acc_per_fold)}")

print("END EXPERIMENTS")
