import os 
import datetime
import os
import zipfile 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import Tensor
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard



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

#normalization of layers by scaling and adding offset
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]

# model =tf.keras.applications.ResNet50(
#     include_top=False,
#     weights=None,
#     input_tensor=None,
#     input_shape=((180,180,3)),
#     pooling=None,
#     classes=3,
    
# )

# #Set epochs to 5 for inital prototype 
# model.compile(
#   optimizer='adam',
#   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])
# model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=5
# )



def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(180, 180, 3))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = create_res_net()
model.summary()
#result = model.evaluate(test_ds)
#dict(zip(model.metrics_names, result))

# timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# name = 'cifar-10_res_net_30-'+timestr # or 'cifar-10_plain_net_30-'+timestr

# checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# os.system('mkdir {}'.format(checkpoint_dir))

# # save model after each epoch
# cp_callback = ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1
# )
# tensorboard_callback = TensorBoard(
#     log_dir='tensorboard_logs/'+name,
#     histogram_freq=1
# )

model.fit(
    train_ds,
    epochs=1,
    verbose=1,
    validation_data=val_ds,
    batch_size=32,
    ##callbacks=[cp_callback, tensorboard_callback]
)

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