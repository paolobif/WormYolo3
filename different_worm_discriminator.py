from tensorflow import keras
from keras import layers as KL
import numpy as np
import os
import random as r
import cv2
from matplotlib import pyplot as plt

batch_size = 128
img_height = 32
img_width = 32

# Initial weights
INIT = keras.initializers.RandomNormal(stddev=0.02)
# Kernel size in convolution
KERNEL_SIZE = (4,4)
# Size of stride in convolution
STRIDE_SIZE=(2,2)
# Alpha of LeakyReLU
ALPHA = 0.2

def load_triplet_training_data(orig_path, max_size = None, n_patch = 16):
  photo1 = []
  photo2 = []
  data = []
  output = []

  fold_paths = []

  for folder in os.listdir(orig_path):
    f_path = os.path.join(orig_path,folder)
    for file in os.listdir(f_path):
      file_ending = file.split(".")[0][-4:]
      if len(photo1) > max_size:
          break
      if "a" in file_ending and not "data" in file_ending:
        f_id = "a".join(file.split("a")[:-1])
        pre_path = os.path.join(f_path, file)
        pre_img = cv2.imread(pre_path,cv2.IMREAD_GRAYSCALE)
        try:
          pre_img = cv2.resize(pre_img,(256,256))
        except Exception as E:
          print(pre_path)
          raise E
        pre_img = np.stack((pre_img,)*3, axis=-1)

        cur_path = os.path.join(f_path, f_id+"b.png")
        cur_img = cv2.imread(cur_path,cv2.IMREAD_GRAYSCALE)
        try:
          cur_img = cv2.resize(cur_img,(256,256))
        except:
          print(cur_path)
          cur_img = cv2.resize(cur_img,(256,256))
        cur_img = np.stack((cur_img,)*3, axis=-1)


        photo1.append(pre_img)
        photo2.append(cur_img)

        photo2.append(pre_img)
        photo1.append(cur_img)

        # Load new Data
        data_path = os.path.join(f_path, f_id+"_data.txt")
        data_file = open(data_path, "r")
        data_info = data_file.read()
        data_file.close()
        data_info = [float(i) for i in data_info.split(",")]
        size = int(len(data_info)/2)
        alt_data_info = data_info[size:] + data_info[0:size]
        data.append(data_info)
        data.append(alt_data_info)

        if "good" in folder:
          output.append(1)
          output.append(1)
        elif "bad" in folder:
          output.append(0)
          output.append(0)




  return np.array(photo1), np.array(photo2), np.array(data), np.array(output)

def gen_real_images(dataset, n_samples, patch_shape):
  orig_imgs, pred_imgs, data, expctd = dataset
  # Random examples
  indices = np.random.randint(0, orig_imgs.shape[0], n_samples)
  orig_exs = orig_imgs[indices]
  pred_exs = pred_imgs[indices]
  data_exs = data[indices]
  expctd_output = expctd[indices]
  #print(orig_exs.shape,pred_exs.shape,expctd_output.shape)
  return [orig_exs, pred_exs, data_exs], expctd_output


def make_discriminator(image_shape:tuple,additional_info_shape:tuple, gen_path = None):
  print("Making Discriminator")
  """
  Creates a 70x70 PatchGAN discriminator

  Args:
      image_shape (tuple): A tuple of the dimensions of the inpeut images
  """

  if not gen_path is None:
    if os.path.exists(gen_path):
      return keras.models.load_model(gen_path)


  first_image = KL.Input(shape = image_shape)
  second_image = KL.Input(shape = image_shape)

  additional_info = KL.Input(shape = additional_info_shape)

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

  flatten_output = KL.Flatten()(output)

  # Reduce to reason
  data_layer = KL.Dense(8, activation = "sigmoid")(additional_info)

  all_output = KL.Concatenate()([flatten_output,data_layer])


  single_output = KL.Dense(1,activation="sigmoid")(all_output)

  model = keras.models.Model([first_image, second_image, additional_info], single_output)

  opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer = opt, loss_weights = [0.5])
  return model

def train(discriminator, data, n_epochs = 50, n_batch = 1, n_patch = 16):

  orig_images1, orig_images2, additional_info, expctd_out = data

  batches_per_epoch = int(len(orig_images1) / n_epochs)
  n_steps = batches_per_epoch * n_epochs

  for i in range(n_steps):
    [x_realA,x_realB, additional_info], y_real = gen_real_images(data, n_batch, n_patch)
    d_loss1 = discriminator.train_on_batch([x_realA,x_realB, additional_info], y_real)
    print('>%d, d1[%.3f]' % (i+1, d_loss1))

def predict_single(img_matrix1, img_matrix2, info):
    global use_discriminator
    img_matrix1 = np.array([cv2.resize(img_matrix1,(256,256))])
    img_matrix2 = np.array([cv2.resize(img_matrix2,(256,256))])
    info = np.array([info])
    #info = info.reshape((8,1))

    return use_discriminator.predict([[img_matrix1],[img_matrix2],[info]],verbose=0)
    #return use_discriminator.predict([img_matrix1,img_matrix2,info],verbose=0)


def generate_many_sdf(img_array1, img_array2, info_array):
  global use_discriminator
  resized_array1 = []
  resized_array2 = []
  for img1, img2 in zip(img_array1,img_array2):
    resized_array1.append(cv2.resize(img1,(256,256)))
    resized_array2.append(cv2.resize(img2,(256,256)))
  resized_array1 = np.array(resized_array1)
  resized_array2 = np.array(resized_array2)
  out_arr = use_discriminator.predict([resized_array1,resized_array2,np.array(info_array)], verbose = 0)

  return out_arr

def make_resnet_discriminator(image_shape:tuple,additional_info_shape:tuple, gen_path = None):
  if not gen_path is None:
    if os.path.exists(gen_path):
      return keras.models.load_model(gen_path)

  first_img_model = keras.applications.ResNet101V2(include_top = False, input_shape = image_shape)
  second_img_model = keras.applications.ResNet101V2(include_top = False, input_shape = image_shape)

  additional_info = KL.Input(shape = additional_info_shape)

  first_img_start = first_img_model.get_layer(index=0).output
  second_img_start = second_img_model.get_layer(index=0).output


  first_img_end = first_img_model.get_layer(index=100).output
  second_img_end = second_img_model.get_layer(index=100).output

  for layer in second_img_model.layers:
    layer._name = layer.name + str("_2")

  first_img_end = KL.Activation("sigmoid")(first_img_end)
  second_img_end = KL.Activation("sigmoid")(second_img_end)

  output = KL.Concatenate()([first_img_end,second_img_end])
  flatten_output = KL.Flatten()(output)

  data_layer = KL.Dense(8, activation = "sigmoid")(additional_info)
  all_output = KL.Concatenate()([flatten_output,data_layer])
  single_output = KL.Dense(1,activation="sigmoid")(all_output)

  model = keras.models.Model([first_img_start, second_img_start,additional_info],single_output)

  opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer = opt, loss_weights = [0.5])

  return model

path = "./worm_binary_same_worm_data_before_merge"
#path = "./worm_resnet_binary"
global use_discriminator
use_discriminator = make_discriminator((256,256,3),(8),gen_path = path)

if __name__ == "__main__":
  #generator = make_binary_generator((256,256,3))
  #generator.save("./worm_binary_segmentation_model2")

  for i in range(5):
    keras.backend.clear_session()
    discriminator = make_discriminator((256,256,3),(8),gen_path = path)

    #org_path = "C:/Users/cdkte/Downloads/match_between_vids_training"
    org_path = "C:/Users/cdkte/Downloads/match_between_vids_again_halloween"
    dataset = load_triplet_training_data(org_path,1100)
    print(dataset[2].shape)

    train(discriminator, dataset,n_batch = 10)

    discriminator.save(path)