import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16


batch_size = 32
img_height = 180
img_width = 180
#Update data root to point at valence dataset 
data_root="/Users/sallyann/Documents/Fall 2020/Capstone/aff_wild_annotations_bboxes_landmarks_new/videos/dataset_valence/"


#get raw data into dataset object to train model
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_root+'train',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#gets raw data for validation dataset (gives unbiased estimate of the the final model)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_root+'train',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#gets raw data for testing model
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_root+'train',
    seed=123,
    image_size=(img_height, img_width),
    batch_size = 200
)


#gets three classs of valence (high,low,neutral)
class_names = train_ds.class_names
print(class_names)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


#mac doesn't use certificates need a work around to use VGG
model = VGG16()
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)




#result = model.evaluate(test_ds)
#dict(zip(model.metrics_names, result))

test_images = []
test_labels = []
predictions = []

for image, label in test_ds.take(1):
  test_images.append(image.numpy())
  test_labels.append(label.numpy())
  predictions.append(np.argmax(model.predict(test_images), axis=1))

test_labels = np.array(test_labels)
predictions = np.array(predictions)

y_true = test_labels

test_acc = sum(predictions[0] == y_true[0]) / len(y_true[0])
print(f'Test set accuracy: {test_acc:.0%}')





print(class_names)
confusion_mtx = tf.math.confusion_matrix(y_true[0], predictions[0]) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, yticklabels = class_names, xticklabels = class_names,annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()