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
  output = []

  fold_paths = []
  for folder in os.listdir(orig_path):
    f_path = os.path.join(orig_path,folder)
    if not folder in ["Dead", "Alive"]:
      continue

    for sub_folder in os.listdir(f_path):
      s_path = os.path.join(f_path,sub_folder)
      fold_paths.append(s_path)

  if max_size is None:
    fold_paths = fold_paths
  else:
    #print(len(fold_paths),max_size)
    fold_paths = r.sample(fold_paths,max_size)

  for folder in fold_paths:
    assert os.path.exists(os.path.join(folder,"pre.jpg")), folder
    pre_img = cv2.imread(os.path.join(folder,"pre.jpg"),cv2.IMREAD_GRAYSCALE)
    pre_img = cv2.resize(pre_img,(256,256))
    pre_img = np.stack((pre_img,)*3, axis=-1)

    cur_img = cv2.imread(os.path.join(folder,"cur.jpg"),cv2.IMREAD_GRAYSCALE)
    cur_img = cv2.resize(cur_img,(256,256))
    cur_img = np.stack((cur_img,)*3, axis=-1)


    photo1.append(pre_img)
    photo2.append(cur_img)

    photo2.append(pre_img)
    photo1.append(cur_img)

    if "Alive" in folder:
      #print(folder,s_path)
      """
      fig, ax = plt.subplots(3)
      ax[0].imshow(pre_img)
      ax[1].imshow(cur_img)
      ax[2].imshow(cv2.imread(cv2.imread(os.path.join(s_path,"cur.jpg"),cv2.IMREAD_GRAYSCALE)))
      plt.show()
      """
      output.append(1)
      output.append(1)
    elif "Dead" in folder:
      output.append(0)
      output.append(0)

  return np.array(photo1), np.array(photo2), np.array(output)

def gen_real_images(dataset, n_samples, patch_shape):
  orig_imgs, pred_imgs, expctd = dataset
  # Random examples
  indices = np.random.randint(0, orig_imgs.shape[0], n_samples)
  orig_exs = orig_imgs[indices]
  pred_exs = pred_imgs[indices]
  expctd_output = expctd[indices]
  #print(orig_exs.shape,pred_exs.shape,expctd_output.shape)
  return [orig_exs, pred_exs], expctd_output


def make_discriminator(image_shape:tuple, gen_path = None):
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
  single_output = KL.Dense(1,activation="sigmoid")(flatten_output)

  model = keras.models.Model([first_image, second_image], single_output)

  opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer = opt, loss_weights = [0.5])
  return model

def train(discriminator, data, n_epochs = 50, n_batch = 1, n_patch = 16):

  orig_images1, orig_images2, expctd_out = data

  batches_per_epoch = int(len(orig_images1) / n_epochs)
  n_steps = batches_per_epoch * n_epochs

  for i in range(n_steps):
    [x_realA,x_realB], y_real = gen_real_images(data, n_batch, n_patch)
    d_loss1 = discriminator.train_on_batch([x_realA,x_realB], y_real)
    print('>%d, d1[%.3f]' % (i+1, d_loss1))


path = "./worm_binary_discrimination"
global use_discriminator
use_discriminator = make_discriminator((256,256,3),gen_path = path)

def predict_single(img_matrix1, img_matrix2):
    global use_discriminator
    img_matrix1 = np.array([cv2.resize(img_matrix1,(256,256))])
    img_matrix2 = np.array([cv2.resize(img_matrix2,(256,256))])
    return use_discriminator.predict([[img_matrix1],[img_matrix2]],verbose=0)

def generate_many_sdf(img_array1, img_array2):
  global use_discriminator
  resized_array1 = []
  resized_array2 = []
  for img1, img2 in zip(img_array1,img_array2):
    resized_array1.append(cv2.resize(img1,(256,256)))
    resized_array2.append(cv2.resize(img2,(256,256)))
  resized_array1 = np.array(resized_array1)
  resized_array2 = np.array(resized_array2)
  out_arr = use_discriminator.predict([resized_array1,resized_array2], verbose = 0)

  return out_arr

if __name__ == "__main__":
  #generator = make_binary_generator((256,256,3))
  #generator.save("./worm_binary_segmentation_model2")

  for i in range(10):
    keras.backend.clear_session()
    discriminator = make_discriminator((256,256,3),gen_path = path)

    org_path = "C:/Users/cdkte/Downloads/triple_worms_2/triple_worms/triple_output"
    dataset = load_triplet_training_data(org_path,1700)
    print(dataset[2].shape)

    train(discriminator, dataset,n_batch = 4)

    discriminator.save(path)