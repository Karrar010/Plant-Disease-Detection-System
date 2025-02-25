import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNEL = 3
EPOCH = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\PlantVillage\strawberryplant",
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

class_names = dataset.class_names
class_names

len(dataset)

#Visualizing the data
plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")

train_size = 0.8
train_data_size = int(train_size*len(dataset))
print(f"Train Size : {train_data_size}")

train_dataset = dataset.take(train_data_size) #80% of the data
print(len(train_dataset))

test_dataset = dataset.skip(train_data_size) #10% of the data
len(test_dataset)

val_size = 0.1
real_val_size = int(len(dataset)*val_size)
print(real_val_size)

val_dataset = test_dataset.take(real_val_size) #5% of the data
len(val_dataset)

test_dataset = test_dataset.skip(real_val_size)
len(test_dataset)

def get_dataset_partition_tf(ds, train_split = 0.8, val_split = 0.1 , test_split = 0.1, shuffle = True, shuffle_size =10000):
    dataset_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_dataset = ds.skip(train_size)
    val_dataset = ds.skip(train_size).take(val_size)
    test_dataset = ds.skip(train_size).skip(val_size)

    return train_dataset , val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = get_dataset_partition_tf(dataset)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

#CNN MODEL
input_shape = (BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
n_classes = 3
model = models.Sequential([
    layers.Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(38, activation = "softmax")
])

model.summary()

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = EPOCH,
    batch_size = BATCH_SIZE,
    verbose = 1,
    )

score = model.evaluate(test_dataset)
score

import numpy as np

for image_batch,labels_batch in test_dataset.take(1):
    first_image = image_batch[0].numpy().astype("uint8")
    first_label = labels_batch[0].numpy()

    print("Image to predict")
    plt.imshow(first_image)
    print("Actual Label: ", class_names[first_label])

    batch_prediction = model.predict(image_batch)
    print("Predicted Label: ", class_names[np.argmax(batch_prediction[0])])

import json

# Save class names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

model.save('mymodel.keras')