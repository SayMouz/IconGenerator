from flaticon_api import FlaticonApi
import requests
import urllib.request
import numpy as np
from PIL import Image


class DataPreprocessing:

    def __init__(self, data_path='./data', filecount=0):
        self.filecount = filecount
        self.data_path = data_path

    def download_images(self, urls):
        for url in urls:
            try:
                urllib.request.urlretrieve(url, f'{self.data_path}/ic_{self.filecount}.png')
                print(f"ic_{self.filecount}  downloaded")
            except:
                print("Error when downloading icon : ", self.filecount)
            self.filecount += 1

    @staticmethod
    def map_tags_into_onehot(tags):
        vectors = np.eye(len(tags))
        mapped_tags = {}
        i = 0
        for tag in tags:
            mapped_tags[tag] = vectors[i]
            i += 1
        return mapped_tags

    @staticmethod
    def tags_to_onehot(tags, mapped_tags):
        onehot = np.zeros(len(mapped_tags))
        for tag in tags:
            if tag in mapped_tags:
                onehot = np.add(onehot, mapped_tags[tag])
        return onehot

    @staticmethod
    def tags_to_onehots(tags_lists, mapped_tags, class_num=None):
        if class_num is None:
            class_num = len(mapped_tags)
        tags_onehots = np.zeros((len(tags_lists), class_num))
        for i in range((len(tags_lists))):
            tags_onehots[i] = DataPreprocessing.tags_to_onehot(tags_lists[i], mapped_tags)
        return tags_onehots

    @staticmethod
    def png_to_jpg(filepath, new_path=None):
        png_img = Image.open(filepath)
        png_img.load()
        background = Image.new("RGB", png_img.size, (255, 255, 255))
        background.paste(png_img, mask=png_img.split()[3])
        if new_path is None:
            new_path = rf'{filepath}.jpg'
        background.save(new_path, 'JPEG', quality=100)

    @staticmethod
    def rgba_to_rgb(filepath, new_path=None):
        rgba_image = Image.open(filepath)
        rgb_image = rgba_image.convert('RGB')
        if new_path is None:
            new_path = filepath
        rgb_image.save(new_path)

    @staticmethod
    def clean_empty_icons(ic_data, tags_data):
        to_delete = []
        for i in range(np.shape(ic_data)[0]):
            if np.all((ic_data == 0)):
                to_delete.append(i)
        clean_data = np.delete(ic_data, to_delete, axis=0)
        clean_tags_data = np.delete(tags_data, to_delete, axis=0)
        return clean_data, clean_tags_data


