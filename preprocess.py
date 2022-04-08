from data_preprocessing import DataPreprocessing
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# ---- Get the list of tags ---- #
tags_filepath = "tags.txt"

print(f"--- Loading tags from {tags_filepath} ...")
with open(tags_filepath) as tags_file:
    tags = [line.rstrip() for line in tags_file]
tags = tags[:5]
tags = [tag.lower() for tag in tags]
onehot_tags = DataPreprocessing.map_tags_into_onehot(tags)
print(f"--- {len(tags)} tags loaded ---")
# ------------------------------- #

# ---- Load icon tags from file and convert them to onehot vectors ---- #
""" Tags need to be separated by a ',' """
ic_tags_filepath = "ic_tags_5.txt"
class_num = 5

with open(ic_tags_filepath) as ic_tags_file:
    ic_tags = [line.rstrip().split(',') for line in ic_tags_file]
tags_data = DataPreprocessing.tags_to_onehots(ic_tags, onehot_tags, class_num)
# --------------------------------------------------------------------- #

# --- Convert png icons to binary matrices --- #
num_ic = 4500
ic_size = 32
png_dirpath = "./data_5/"
jpg_dirpath = "./data_5_jpg/"

print("--- Converting png icons into binary matrices ... ")
data_array = np.zeros((num_ic, ic_size, ic_size))
for i in range(num_ic):
    try:
        print(f'Processing ic_{i}')
        DataPreprocessing.png_to_jpg(f'{png_dirpath}ic_{i}.png', f'{jpg_dirpath}ic_{i}.jpg')
        img = load_img(f'{jpg_dirpath}ic_{i}.jpg', color_mode="grayscale")
        img_array = img_to_array(img)
        img_array = np.reshape(img_array, (ic_size, ic_size))
        img_array = np.where(img_array != 0, 1, 0)
        data_array[i] = img_array
    except:
        print(f'Could not convert ic_{i}')
        data_array[i] = np.zeros((ic_size, ic_size))
print(f"--- Conversion ended ---")
# ---------------------------------------------- #


# ----- Clean empty icon data ------ #
ic_data = np.loadtxt("data_5.csv")
clean_ic_data, clean_tags_data = DataPreprocessing.clean_empty_icons(ic_data, tags_data)
# ------------------------------------- #

# --- Save into files ---- #

# Save onehots vectors or flatten icon data
np.savetxt('tags_data_5.csv', tags_data, fmt='%i')

# Save array images
reshaped_data = data_array.reshape(data_array.shape[0], -1)  # 3D -> 2D
np.savetxt("data_5.csv", reshaped_data, fmt='%i')

# Save clean data
np.savetxt("data_5_clean.csv", clean_ic_data, fmt='%i')
np.savetxt('tags_data_5_clean.csv', clean_tags_data, fmt='%i')

# ------------------------ #
