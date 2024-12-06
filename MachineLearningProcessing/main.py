# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import os
import datetime

# %load_ext tensorboard

print(tf.__version__)
print(keras.__version__)
print(tf.config.list_physical_devices('GPU'))

"""
## Klasyfikator obrazów ze zbioru ImageNet
### Pobranie klasyfikatora
"""

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

inception_v1 = "https://www.kaggle.com/models/google/inception-v1/TensorFlow2/classification/2"
inception_v2 = "https://www.kaggle.com/models/google/inception-v2/TensorFlow2/classification/2"
inception_resnet_v2 = "https://www.kaggle.com/models/google/inception-resnet-v2/TensorFlow2/classification/2" #299

resnet50_v1 = "https://www.kaggle.com/models/google/resnet-v1/TensorFlow2/50-classification/2"
resnet101_v1 = "https://www.kaggle.com/models/google/resnet-v1/TensorFlow2/101-classification/2"
resnet152_v1 = "https://www.kaggle.com/models/google/resnet-v1/TensorFlow2/152-classification/2"

resnet50_v2 = "https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/50-classification/2"
resnet101_v2 = "https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/101-classification/2"
resnet152_v2 = "https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/152-classification/2"

efficientnet_b0_v1 = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b0-classification/1"
efficientnet_b4_v1 = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b4-classification/1"
efficientnet_b7_v1 = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b7-classification/1"

efficientnet_b0_v2 = "https://www.kaggle.com/models/google/efficientnet-v2/TensorFlow2/imagenet1k-b0-classification/2"
nasnet_mobile = "https://www.kaggle.com/models/google/nasnet/TensorFlow2/mobile-classification/2"

classifier_model = resnet152_v2
IMAGE_SIZE = 224 #299
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
num_classes = 1000

classifier = keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

classifier.summary()

url_list = []
url_list.append("https://ocdn.eu/lamoda-web-transforms/1/zUQk9k8aW1hZ2VzL29mZmVycy8yODc0MTQ0MjQvNHN0S2s0Yzk1RUg5UjBOZEFWV2Npbm5vODFxbzBoM1guanBnkpUCzQL4AMLDlAbM_8z_zP-BAAE")
url_list.append("https://static.reserved.com/media/catalog/product/cache/850/a4e40ebdc3e371adff845072e1c73f37/7/4/745BE-00X-010-1-850309_4.jpg")
url_list.append("https://api.modago.pl/img/jpg/400/1024/resize/catalog/product/3/0/30fe9d6bc0aab598df1dd6f1ed20063c_5266733.webp")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/1/9/190AQ-00X-010-1-790100_4.jpg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/1/6/168AQ-99X-010-1-837990_8.jpg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/8/5/8525W-99X-010-1-644549_5.jpg")
url_list.append("https://ocdn.eu/photo-offers-prod-transforms/1/Gi6k9k3b2ZmZXJzLzIzNjkxNjEwMTdfMzY3MTdhM2RkYmU4Y2I0ZDI3NWI1Zjg1MjEwYzg4Mjcud2VicJOVAsz6AMLDlQIAzPrCw5MJpjA2MzZhNwbeAAGhMAU/reserved-plaszcz-z-welna-niebieski.webp")
url_list.append("https://lb5.dstatic.pl/images/product-details/175490763-plaszcz-damski-reserved-jesienny.jpg")
url_list.append("https://image.ceneostatic.pl/data/products/175457286/p-reserved-welniany-plaszcz-czarny.jpg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/2/0/206BF-59X-010-1-887409_4.jpg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/4/2/424BW-83X-010-1-882293_4.jpg")
url_list.append("https://api.modago.pl/img/jpg/400/1024/resize/catalog/product/c/3/c33b8a9ee3eff730a5d2dfe1f4a0427b_6338340.webp")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/4/2/423BC-99X-010-1-875440_4.jpg")
url_list.append("https://image.ceneostatic.pl/data/products/149322157/p-reserved-torba-shopper-bezowy.jpg")
url_list.append("https://ocdn.eu/lamoda-web-transforms/1/lUpk9k8aW1hZ2VzL29mZmVycy8yODQ5ODY1NzQvZnNHd1g2VnRoSzdBUjBud29PWlNxamZCRlpCdmRJSDAuanBnkpUCzQL4AMLDlAbM_8z_zP-BAAE")
url_list.append("https://cdn.aboutstatic.com/file/images/69c8fa95623f114e671355745ea203b6.jpeg")
url_list.append("https://m.media-amazon.com/images/I/41GiezhcfoL._AC_UY1000_.jpg")
url_list.append("https://i5.walmartimages.com/seo/Women-s-Casual-Dresses-Long-Summer-Dresses-Women-s-V-Neck-Casual-Pleated-Button-Solid-Color-Short-Sleeved-Dress_342aadea-7784-4284-b6cd-9a736e1b45d0.4548fefcb0bc8245e1e65cde84a77937.jpeg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/X/E/XE921-99X-010-1-719700_21.jpg")
url_list.append("https://ocdn.eu/photo-offers-prod-transforms/1/t30k9k3b2ZmZXJzLzIzOTcxMjQ2OTJfYTAwNjFlZjg1NzVhYTI0NDFiYzNiZTgwYTkwZDhiM2Qud2VicJGTCaYyMWQzMmIG3gABoTAF/reserved-t-shirt-z-efektem-sprania-ciemnoszary.webp")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/8/9/897BD-33X-010-1-857304_5.jpg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/8/7/8769S-99X-010-1-716905_2.jpg")
url_list.append("https://static.reserved.com/media/catalog/product/cache/1200/a4e40ebdc3e371adff845072e1c73f37/7/3/7396S-76X-010-1-705879.jpg")

imgs = []
index = 0
for url in url_list:
  path = url
  path = keras.utils.get_file('image_{}.jpg'.format(index), path)
  imgs.append(np.array( Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE)) ) / 255.0)
  # plt.figure()
  # plt.imshow(imgs[index])
  # plt.grid(False)
  # plt.show()
  index += 1

index = 0
for img in imgs:
  plt.figure()
  plt.imshow(img)
  plt.grid(False)
  result = classifier.predict(img[np.newaxis, ...])
  result.shape
  predicted_class = tf.math.argmax(result[0], axis=-1)
  predicted_class

  labels_path = keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
  imagenet_labels = np.array(open(labels_path).read().splitlines())
  plt.imshow(img)
  plt.axis('off')
  predicted_class_name = imagenet_labels[predicted_class+1-result.shape[1]%num_classes]
  _ = plt.title("Prediction: " + predicted_class_name.title())
  index += 1


"""### Uruchomienie klasyfikatora na pojedyńczym obrazie"""

sandal1_model = 'https://static.reserved.com/media/catalog/product/4/6/463BU-99X-002-1-832324_2.jpg'
path = sandal1_model
path = keras.utils.get_file('image.jpg', path)

img = np.array( Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE)) ) / 255.0
os.remove(path)

plt.figure()
plt.imshow(img)
plt.grid(False)
plt.show()

img.shape

result = classifier.predict(img[np.newaxis, ...])

result.shape

predicted_class = tf.math.argmax(result[0], axis=-1)

predicted_class

"""### Dekodowanie predykcji"""

labels_path = keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(img)
plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class+1-result.shape[1]%num_classes]

_ = plt.title("Prediction: " + predicted_class_name.title())

"""## Import i wstępne przygotowanie danych"""

class ImgAug(keras.layers.Layer):

  def call(self, img):

    offset=int((IMAGE_SIZE-img.shape[0])/2)
    return tf.image.grayscale_to_rgb(
        tf.image.pad_to_bounding_box(img, offset, offset, IMAGE_SIZE, IMAGE_SIZE)
    )

resize_and_rescale = keras.Sequential([
 #keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
 keras.layers.Rescaling(1./255),
 ImgAug()
])

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'fashion_mnist',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False):

  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),
])

class AddColor(keras.layers.Layer):
    def call(self, images):
        # Dynamiczne pobieranie rozmiaru batcha
        batch_size = tf.shape(images)[0]
        # Generowanie kolorów w zakresie [0.5, 1.0]
        colors = tf.random.uniform([batch_size, 1, 1, 3], 0.5, 1.0)
        # Zastosowanie koloru do obrazu
        return images * colors

add_color_layer = AddColor()

class AddGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev):
        super(AddGaussianNoise, self).__init__()
        self.stddev = stddev

    def call(self, images):
        noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=self.stddev)
        return tf.clip_by_value(images + noise, 0.0, 1.0)

noise_layer = AddGaussianNoise(0.1)

train_ds = prepare(train_ds, shuffle=True)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (add_color_layer(x), y))
train_ds = train_ds.map(lambda x, y: (noise_layer(x), y))
test_ds = prepare(test_ds)
val_ds = prepare(val_ds)

num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(get_label_name(labels_batch[n]))
  plt.axis('off')
_ = plt.suptitle("Train batch")

result = classifier.predict(image_batch)
print(result.shape)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  predicted_class = tf.math.argmax(result[n], axis=-1)
  predicted_class_name = imagenet_labels[predicted_class+1-result.shape[1]%num_classes]
  plt.title(predicted_class_name.title())
  plt.axis('off')
_ = plt.suptitle("Predicted classes")

"""## Trening modelu"""

mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)

model = keras.Sequential([
  feature_extractor_layer,
  keras.layers.Dense(num_classes)
])

model.summary()

model.compile(
  optimizer=keras.optimizers.Adam(),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

NUM_EPOCHS = 3

history = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)

"""## Dekodowanie predykcji wytrenowanego modelu"""

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  predicted_class_name = get_label_name(predicted_label)
  true_class_name = get_label_name(true_label)
  plt.xlabel("{} {:2.0f}% ({})".format(predicted_class_name.title(),
                                100*np.max(predictions_array),
                                true_class_name.title()),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

probability_model = keras.Sequential([model,
                                         keras.layers.Softmax()])

result = probability_model.predict(img[np.newaxis, ...])

result.shape

for label in range(10):
    print(label, " ",get_label_name(label))

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, result[i], [5], img[np.newaxis, ...])
plt.subplot(1,2,2)
plot_value_array(i, result[i],  [5])
plt.show()

result = probability_model.predict(image_batch)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, result[i], labels_batch, image_batch)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, result[i], labels_batch)
plt.tight_layout()
plt.show()

"""## Zadania

1. Pobierz z Internetu co najmniej kilkanaście fotografii (do 30) należących do klas ze zbioru Fasion-MNIST, zbuduj z nich niewielki zbiór i sprawdź na nim rezultaty kilku klasyfikatorów wytrenowanych na zbiorze ImageNet. Wybierz najlepszy z nich i użyj go do treningu. (1 pkt)
2. Dodaj modyfikacje danych treningowych polegające na losowych zmianach rozmiaru i położenia. (1 pkt)
3. W zbiorze Fashion-MNIST wszystkie obrazy są monochromatyczne. Wzbogać przykłady treningowe o kolory. (1 pkt)
4. Zweryfikuj wyniki na wcześniej przygotowanym, własnym zbiorze obrazów a następnie wprowdź kolejne przekształcenia obrazów treningowych, tak aby zwiększyć celność predykcji. (2 pkt)



## Źródła

- [Transfer learning with TensorFlow Hub](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)
- [Data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

"""