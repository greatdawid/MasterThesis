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
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import dataset_split as ds
import os
import cv2 as cv
import pandas as pd

BATCH_SIZE = 16

classes = ["without_mask", "with_mask"]
labels = []
data = []
#img = cv.imread('00088.png')
#img = cv.resize(img, (1024, 1024))
#img = img[1762:2554, 870:1662]
#cv.imshow('testwin',img)# scale to dimension
#cv.waitKey(0)
#exit(0)
# def prepare_data_and_labels_():
pan = pd.read_json('kaggle_dataset.json')
df = pd.DataFrame(pan)
# print(ds.PATH_TO_IMAGES)
# for idx, image in enumerate(os.listdir(ds.PATH_TO_IMAGES)):
#    X, Y = df["pic_dims"][idx]
#    img = cv.imread(image)
#    cv.imshow('w',img)
#    #cv.resize(img,(200,200))
#    #cv.resize(img,(int(X),int(Y)))
#    for face in df.columns[3:]:
#        info = df[face][idx]
#        print(info)
# print("Wczytuje obrazy")

print("Extracting each data into respective label folders....")
for idx, image in enumerate(os.listdir(ds.PATH_TO_IMAGES)):
    img = cv.imread(os.path.join(ds.PATH_TO_IMAGES, image))
    # scale to dimension
    X, Y = df["pic_dims"][idx]
    cv.resize(img, (int(X), int(Y)))
    # find the face in each object
    for face in df.columns[3:]:
        info = df[face][idx]
        if info != 0:
            label = info[0]
            if (label == "mask_weared_incorrect"):
                label = 'without_mask'
                info[0] = 'without_mask'
            info[0] = info[0].replace(str(label), str(classes.index(label)))
            info = [int(each) for each in info]
            face = img[info[2]:info[4], info[1]:info[3]]
            if ((info[3] - info[1]) > 40 and (info[4] - info[2]) > 40):
                try:
                    face = cv.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    data.append(face)
                    labels.append(label)
                except:
                    pass



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
predictions = []
true_labels = []
NUM_EPOCHS = 10
for fold_id, (train_index, test_index) in enumerate(rskf.split(data, labels)):

    print(f"Fold Number: {fold_id}")
    #labels = to_categorical(labels, dtype="int32")
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = to_categorical(labels[train_index], dtype="int32"),\
                      to_categorical(labels[test_index], dtype="int32")
    #Y_train, Y_test = labels[train_index], labels[test_index]
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

#  !!Jak ma wyglądać strategia dla odmrazania? Czy np dla kazdego folda mam robic
    #  kolejnego fora z ilościa odmrozonych warstw? Ilość, 1-10?
    #!! Jak duża ma być ilość odmrożonych warstw?
    #zakladam, że po prostu odmrażam kolejne warstwy bez wzgledu jakie są
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = Adam(lr=1e-4, decay=1e-4 / 10)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    csv_logger = CSVLogger('training_log.csv',append=True)
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(X_train, Y_train, batch_size=16),
        steps_per_epoch=len(X_train) // 16,
        validation_data=(X_test, Y_test),
        validation_steps=len(X_test) // 16,
        callbacks=[csv_logger],
        epochs=NUM_EPOCHS)
    #predictions.append(model.predict(X_test)) # czy zrzutowac to na 0 lub 1
    predicts = model.predict(X_test)
    predicts_labels = np.argmax(predicts, axis=-1)
    #print(model.predict(X_test))
    #print(predicts_labels)
    true_labels.append(Y_test)
    #!!w macierzy sa predykcje i etykiety, w csv_logger są wyniki walidacji itp (loss i accurracy)!!

    #Y predict aby miec etykiety
    #doloz predyckje i etykiety do macierzy aby miec wszystkow jednym miejsuc
    #!!Czy te predykcje i etykiety maja byc dla kazdego folda, epoki, i odmrozonej warstwy? Wtedy lista 5D lub 4D jesli tylko dla folda !!

#zachowuje model
#!!czy zachować model dla kazdego foldu?, może tylko dla najlepszej!!?

model.save(f"model_natural.model", save_format="h5")


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
