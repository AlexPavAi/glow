import numpy as np
import os
from PIL import Image
import imageio



def normalize():
    img_folder = os.path.join('/home/nird/glow/demo/', 'Female_Combinations')
    path_to_save = os.path.join('/home/nird/glow/demo/', 'Female_Combinations_normalized')
    images_list = []
    images_names = []


    for img in os.listdir(img_folder):
        images_names.append(img)
        img = os.path.join(img_folder, img)
        img = Image.open(img)
        img = np.array(img)
        images_list.append(img)


    images_np_array = np.array(images_list)

    max_value = images_np_array.max()
    min_value = images_np_array.min()
    for np_image, img_name in zip(images_np_array, images_names):
        np_image_normalized = (np_image - min_value) / (max_value - min_value)
        save_name = os.path.join(path_to_save, img_name+'_normalized.jpg')
        imageio.imwrite(save_name, np_image_normalized)


def normalize_and_get_table_of_latents():
    img_folder = os.path.join('/home/nird/glow/demo/', 'Female_Combinations')
    path_to_save = os.path.join('/home/nird/glow/demo/', 'Female_Combinations_normalized')
    images_list = []
    images_names = []
    latents_list = []

    for img in os.listdir(img_folder):
        images_names.append(img)
        img = os.path.join(img_folder, img)
        img = Image.open(img)
        img = np.array(img)
        images_list.append(img)

    images_np_array = np.array(images_list)

    max_value = images_np_array.max()
    min_value = images_np_array.min()
    for np_image, img_name in zip(images_np_array, images_names):
        np_image_normalized = (np_image - min_value) / (max_value - min_value)
        save_name = os.path.join(path_to_save, img_name + '_normalized.jpg')
        imageio.imwrite(save_name, np_image_normalized)



def get_table_of_latents():
    img_folder = os.path.join('/home/nird/glow/demo/', 'Female_Combinations_normalized')
    latents_list = []
    file_name_to_save = os.path.join('/home/nird/glow/demo/', 'Female_head_combinations_normalized_latents')

    for img in os.listdir(img_folder):
        img = os.path.join(img_folder, img)
        img = Image.open(img)
        img = np.array(img)
        curr_latent = encode(img)
        latents_list.append(curr_latent)

    latents_list = np.array(latents_list)
    latents_list = np.squeeze(latents_list, axis=1)
    save_obj(latents_list, file_name_to_save)
    print("object saved")