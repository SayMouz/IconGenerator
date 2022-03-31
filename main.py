from flaticon_api import FlaticonApi
from data_preprocessing import DataPreprocessing
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# Get the list of tags
with open("tags.txt") as tags_file:
    tags = [line.rstrip() for line in tags_file]
tags = tags[:23]
tags = [tag.lower() for tag in tags]
onehot_tags = DataPreprocessing.map_tags_into_onehot(tags)

with open("ic_tags.txt") as ic_tags_file:
    ic_tags = [line.rstrip().split(',') for line in ic_tags_file]
ic_tags = ic_tags[:4600]

tags_data = np.zeros((4600, 23))
for i in range(4600):
    tags_data[i] = DataPreprocessing.tags_to_onehot(ic_tags[i], onehot_tags)

np.savetxt('tags_data.csv', tags_data, fmt='%i')


"""
data_array = np.zeros((4600, 32, 32))
for i in range(4600):
    try:
        print(i)
        img = load_img(f'./data/ic_{i}.png', color_mode="grayscale")
        img_array = img_to_array(img)
        img_array = np.reshape(img_array, (32, 32))
        img_array = np.where(img_array != 0, 1, 0)
        data_array[i] = img_array
    except:
        data_array[i] = np.zeros((32, 32))

# reshaping the array from 3D
# matrice to 2D matrice and save it to file.
arr_reshaped = data_array.reshape(data_array.shape[0], -1)
np.savetxt("data.csv", arr_reshaped, fmt='%i')
"""
"""
loaded_arr = np.loadtxt("geekfile.txt")
load_original_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // array.shape[2], array.shape[2])
"""

"""
# For each tag, search for the icons and download them
api = FlaticonApi()
limit_per_tag = 200
icon_size = "32"
all_ic_urls = []
all_ic_tags = []
tags = tags[10:]
for tag in tags:
    data, metadata = api.get_black_icons(tag, 200)
    ic_urls = [d["images"][icon_size] for d in data]
    ic_tags = [d["tags"].split(',') for d in data]
    # The tag searched may not appear in the tags, so we add it
    for ft in ic_tags:
        if tag not in ft:
            ft.append(tag)
    all_ic_urls = [*all_ic_urls, *ic_urls]
    all_ic_tags = [*all_ic_tags, *ic_tags]

with open("ic_tags.txt") as ic_tags_file:
    ic_tags_f = [line.rstrip() for line in ic_tags_file]

ic_tags_f = ic_tags_f[:2000]
all_ic_tags = [*ic_tags_f, *all_ic_tags]
# Write all tags in a file
with open('ic_tags.txt', 'w') as f:
    for item in all_ic_tags:
        f.write("%s\n" % item)

# Download all icons
dp = DataPreprocessing(data_path='./data', filecount=2000)
dp.download_images(all_ic_urls)
"""
"""
img = load_img('./data/ic_119.png', color_mode="grayscale")
# convert to numpy array
img_array = img_to_array(img)
img_array = np.reshape(img_array, (32, 32))
img_array = np.where(img_array != 0, 1, 0)
np.savetxt('data.csv', img_array, fmt='%i')
print("fin")
img1 = load_img('./data/ic_120.png', color_mode="grayscale")
# convert to numpy array
img_array2 = img_to_array(img1)
img_array2 = np.reshape(img_array2, (32, 32))
img_array2 = np.where(img_array2 != 0, 1, 0)
np.savetxt('data.csv', img_array2, fmt='%i')
print("fin")

final_array = np.empty((1, 32, 32))
final_array = np.append(final_array, img_array)
final_array = np.append(final_array, img_array2)
print("x")
"""

