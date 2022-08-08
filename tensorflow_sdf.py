from sys import base_exec_prefix
from typing_extensions import dataclass_transform
from sklearn import datasets
from tensorflow import keras
import tensorflow as tf

from keras import layers as KL
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import signed_distance_fields as sdf
import random as r

# centroid cv2 and other things
#https://learnopencv.com/blob-detection-using-opencv-python-c/

# Try running on videos and see what happens to get data
#   plot average over time


keras.backend.clear_session()

global sdf_generator

tf.get_logger().setLevel("FATAL")
#print("\n\n\n")

batch_size = 128
img_height = 32
img_width = 32

def load_training_data(original_photos, annotated_photos,max_size = None):
  print("Loading Training")
  paths = []
  anno_paths = []
  correct_images = []
  sdf_for_images = []
  for i in range(3,21):
    orig_folder = os.path.join(original_photos+"/"+str(i))
    anno_folder = os.path.join(annotated_photos+"/"+str(i))
    if not os.path.exists(orig_folder) or not os.path.exists(anno_folder):
      continue
    print(i)


    for file in os.listdir(orig_folder):
      base_img_path = os.path.join(orig_folder,file)
      anno_img_path = os.path.join(anno_folder,"Annotated_"+file)
      if os.path.exists(anno_img_path):
        paths.append(base_img_path); anno_paths.append(anno_img_path)
      else:
        continue


  if max_size is None:
    path_list = paths
  else:
    path_list = r.sample(paths,max_size)

  for base_img_path in path_list:

    anno_img_path = anno_paths[paths.index(base_img_path)]
    cur_sdf = sdf.identify_worm(base_img_path,anno_img_path)
    if cur_sdf is None:
      continue

    cur_img = cv2.imread(base_img_path,cv2.IMREAD_GRAYSCALE)
    cur_sdf = cv2.resize(cur_sdf,(256,256))
    cur_img = cv2.resize(cur_img,(256,256))
    stacked_img = np.stack((cur_img,)*3, axis=-1)
    stacked_sdf = np.stack((cur_sdf,)*3, axis=-1)

    correct_images.append(stacked_img)
    sdf_for_images.append(stacked_sdf)

  return np.array(correct_images), np.array(sdf_for_images)

# Initial weights
INIT = keras.initializers.RandomNormal(stddev=0.02)
# Kernel size in convolution
KERNEL_SIZE = (4,4)
# Size of stride in convolution
STRIDE_SIZE=(2,2)
# Alpha of LeakyReLU
ALPHA = 0.2

def make_discriminator(image_shape:tuple):
  print("Making Discriminator")
  """
  Creates a 70x70 PatchGAN discriminator

  Args:
      image_shape (tuple): A tuple of the dimensions of the input images
  """
  first_image = KL.Input(shape = image_shape)
  second_image = KL.Input(shape = image_shape)

  # Take both images as a single input
  merged = KL.Concatenate()([first_image,second_image])
  base_size = 64


  # C64
  L1 = KL.Conv2D(base_size,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer = INIT)(merged)
  L1_act = KL.LeakyReLU(alpha=ALPHA)(L1)
  prev_layer = L1
  i = 2

  # C128-512
  while i * base_size <= 512:
    layer = KL.Conv2D(i*base_size,KERNEL_SIZE,strides = STRIDE_SIZE, padding = "same",kernel_initializer = INIT)(prev_layer)
    layer = KL.BatchNormalization()(layer)
    layer = KL.LeakyReLU(alpha=ALPHA)(layer)

    prev_layer = layer
    i*=2

  # Prep for output
  prep = KL.Conv2D(512,KERNEL_SIZE,padding="same",kernel_initializer=INIT)(prev_layer)
  prep = KL.BatchNormalization()(prep)
  prep = KL.LeakyReLU(alpha=ALPHA)(prep)
  # Output
  output = KL.Conv2D(1,KERNEL_SIZE,padding = "same",kernel_initializer = INIT)(prep)
  output = KL.Activation("sigmoid")(output)

  model = keras.models.Model([first_image, second_image], output)

  opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer = opt, loss_weights = [0.5])
  return model

def make_encoder_block(prev_layer, n_filters,batchnorm = True):
  """
  Creates a block that reduces the image as in a CNN

  Args:
      prev_layer keras.layers.Layer: The previous layer
      n_filters int: The number of filters
      batchnorm (bool, optional): Whether to add a batch normalization. Defaults to True.

  Returns:
      keras.layers.Layer: The last layer in this block
  """
  downsample_layer = KL.Conv2D(n_filters,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer=INIT)(prev_layer)
  if batchnorm:
    downsample_layer = KL.BatchNormalization()(downsample_layer)
  activation_layer = KL.LeakyReLU(alpha=ALPHA)(downsample_layer)
  return activation_layer

def make_decoder_block(prev_layer,skip_in,n_filters,dropout=True):
  """
  Creates a block that reduces the image as in a CNN

  Args:
      prev_layer keras.layers.Layer: The previous layer
      skip_in keras.layers.Layer: The layer to add after the convolution
      n_filters int: The number of filters
      dropout (bool, optional): Whether to add a dropout layer. Defaults to True.

  Returns:
      keras.layers.Layer: The last layer in this block
  """
  upsample_layer = KL.Conv2DTranspose(n_filters,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer=INIT)(prev_layer)
  batch = KL.BatchNormalization()(upsample_layer)
  if dropout:
    batch = KL.Dropout(0.5)(batch, training=True)

  add_skip = KL.Concatenate()([batch, skip_in])
  activation_layer = KL.Activation("linear")(add_skip)
  return activation_layer

def make_generator(image_shape=(256,256,3),gen_path = None):
  if not gen_path is None:
    return keras.models.load_model(gen_path)
  print("Creating Generator")
  """
  Creates a model that makes images from images

  Args:
      image_shape (tuple, optional): The size of image to input and output. Defaults to (256,256,3).
  """
  input_image = KL.Input(shape=image_shape)
  e1 = make_encoder_block(input_image,64,batchnorm=False)
  e2 = make_encoder_block(e1,128)
  e3 = make_encoder_block(e2,256)
  e4 = make_encoder_block(e3,512)
  e5 = make_encoder_block(e4,512)
  e6 = make_encoder_block(e5,512)
  e7 = make_encoder_block(e6,512)

  # Image is currently fully encoded
  # One convolution layer and activation
  #print(e7.get_shape())
  mid = KL.Conv2D(512,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer=INIT)(e7)
  actv_layer = KL.Activation("linear")(mid)

  # Now start decoding
  d1 = make_decoder_block(actv_layer, e7, 512)
  d2 = make_decoder_block(d1, e6, 512)
  d3 = make_decoder_block(d2, e5, 512)
  d4 = make_decoder_block(d3, e4, 512,dropout=False)
  d5 = make_decoder_block(d4, e3, 256,dropout=False)
  d6 = make_decoder_block(d5, e2, 128,dropout=False)
  d7 = make_decoder_block(d6, e1, 64,dropout=False)

  # Get output
  conv = KL.Conv2DTranspose(3,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer=INIT)(d7)
  output_image = KL.Activation("linear")(conv)

  model = keras.models.Model(input_image, output_image)
  return model

def make_binary_generator(image_shape=(256,256,3),gen_path = None):
  if not gen_path is None:
    return keras.models.load_model(gen_path)
  print("Creating Generator")
  """
  Creates a model that makes images from images and outputs between 0 and 1

  Args:
      image_shape (tuple, optional): The size of image to input and output. Defaults to (256,256,3).
  """
  input_image = KL.Input(shape=image_shape)
  e1 = make_encoder_block(input_image,64,batchnorm=False)
  e2 = make_encoder_block(e1,128)
  e3 = make_encoder_block(e2,256)
  e4 = make_encoder_block(e3,512)
  e5 = make_encoder_block(e4,512)
  e6 = make_encoder_block(e5,512)
  e7 = make_encoder_block(e6,512)

  # Image is currently fully encoded
  # One convolution layer and activation
  #print(e7.get_shape())
  mid = KL.Conv2D(512,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer=INIT)(e7)
  actv_layer = KL.Activation("sigmoid")(mid)

  # Now start decoding
  d1 = make_decoder_block(actv_layer, e7, 512)
  d2 = make_decoder_block(d1, e6, 512)
  d3 = make_decoder_block(d2, e5, 512)
  d4 = make_decoder_block(d3, e4, 512,dropout=False)
  d5 = make_decoder_block(d4, e3, 256,dropout=False)
  d6 = make_decoder_block(d5, e2, 128,dropout=False)
  d7 = make_decoder_block(d6, e1, 64,dropout=False)

  # Get output
  conv = KL.Conv2DTranspose(3,KERNEL_SIZE,strides=STRIDE_SIZE,padding="same",kernel_initializer=INIT)(d7)
  output_image = KL.Activation("sigmoid")(conv)

  model = keras.models.Model(input_image, output_image)
  return model

def make_gan(discriminator,generator,image_shape):
  """
  Creates the GAN model for training the generator


  Args:
      discriminator (keras.models.Model): The discriminator model to tell whether images are acceptable
      generator (keras.models.Model): The generator model to create new images
      image_shape (keras.models.Model): The shape of images inputted and outputted
  """

  print("Making GAN")

  # When we use the GAN, we don't want to train the discriminator
  for layer in discriminator.layers:
    if not isinstance(layer, KL.BatchNormalization):
      layer.trainable = False

  input_image = KL.Input(shape = image_shape)
  generator_output = generator(input_image)
  generator_output._name = "gen_out"
  discriminator_output = discriminator([input_image, generator_output])
  discriminator_output._name = "disc_out"

  model = keras.models.Model(input_image, [discriminator_output, generator_output])

  opt = keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5)
  print("Compiling GAN")
  model.compile(loss = ["binary_crossentropy","mae"],optimizer=opt, loss_weights=[1,100])
  print("Finished compiling")
  return model

def gen_real_images(dataset, n_samples, patch_shape):
  orig_imgs, pred_imgs = dataset
  # Random examples
  indices = np.random.randint(0, orig_imgs.shape[0], n_samples)
  orig_exs = orig_imgs[indices]
  pred_exs = pred_imgs[indices]
  expctd_output = np.ones((n_samples,patch_shape,patch_shape,1))
  return [orig_exs, pred_exs], expctd_output

def gen_fake_images(generator, samples, patch_shape):
  X = generator.predict(samples)
  Y = np.zeros((len(X),patch_shape,patch_shape,1))
  return X, Y


def train(discriminator, generator, gan, data, n_epochs = 50, n_batch = 1, n_patch = 16):
  print("training")
  orig_images, expected_images = data
  batches_per_epoch = int(len(orig_images) / n_epochs)
  n_steps = batches_per_epoch * n_epochs


  for i in range(n_steps):
    [x_realA,x_realB], y_real = gen_real_images(data, n_batch, n_patch)
    x_fakeB,y_fake = gen_fake_images(generator,x_realA,n_patch)
    d_loss1 = discriminator.train_on_batch([x_realA,x_realB], y_real)
    d_loss2 = discriminator.train_on_batch([x_realA,x_fakeB], y_fake)
    g_loss, *_ = gan.train_on_batch(x_realA, [y_real, x_realB])
    print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))


def generate_single_sdf(img_matrix):
  x, y, channels = img_matrix.shape
  img_matrix = np.array([cv2.resize(img_matrix,(256,256))])
  #print(img_matrix.shape)
  sdf_matrix = sdf_generator.predict(img_matrix,verbose = 0)[0]
  sdf_matrix = np.mean(sdf_matrix, axis = -1)
  #plt.imshow(sdf_matrix)
  #plt.show()
  sdf_matrix = cv2.resize(sdf_matrix,(y,x))
  return sdf_matrix

def generate_many_sdf(img_array):
  resized_array = []
  orig_size = []
  for img in img_array:
    resized_array.append(cv2.resize(img,(256,256)))
    orig_size.append(img.shape)
  resized_array = np.array(resized_array)
  sdf_array = sdf_generator.predict(resized_array, verbose = 0)

  out_arr = []
  for img,orig_shape in zip(sdf_array,orig_size):
    img = np.mean(img, axis=-1)
    img = cv2.resize(img,(orig_shape[1],orig_shape[0]))
    out_arr.append(img)
  return out_arr

sdf_generator = make_generator((256,256,3),"./worm_binary_segmentation_model2")

if __name__ == "__main__":
  #generator = make_binary_generator((256,256,3))
  #generator.save("./worm_binary_segmentation_model2")

  for i in range(10):
    keras.backend.clear_session()
    discriminator = make_discriminator((256,256,3))
    #generator = make_generator((256,256,3),"./worm_segmentation_model")
    #generator = make_binary_generator((256,256,3),"./worm_binary_segmentation_model2")
    generator = make_binary_generator((256,256,3),"./worm_binary_segmentation_model2")

    gan_model = make_gan(discriminator,generator,(256,256,3))

    #org_path = "C:/Users/cdkte/Downloads/keras_training/tf_data_test"
    #anno_path = "C:/Users/cdkte/Downloads/keras_training/tf_anno_data_test"
    org_path = "C:/Users/cdkte/Downloads/320_anno/orig_"
    anno_path = "C:/Users/cdkte/Downloads/320_anno/anno"

    dataset = load_training_data(org_path,anno_path,2000)

    img = np.mean(dataset[1][0], axis = -1)
    #plt.imshow(img)

    #plt.show()

    train(discriminator, generator, gan_model, dataset,n_batch = 4)
    generator.save("./worm_binary_segmentation_model2")

"""
  pred_values = discriminator.predict(dataset)
  print(np.mean(pred_values))
  print(dataset[0].shape)
  pred_imgs = generator.predict(dataset[0])
  first_img = pred_imgs[0]
  first_img = np.mean(first_img, axis = -1)
  plt.imshow(first_img)
  plt.show()"""

