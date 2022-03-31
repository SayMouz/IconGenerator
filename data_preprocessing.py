from flaticon_api import FlaticonApi
import requests
import urllib.request
import numpy as np


class DataPreprocessing:

    def __init__(self, data_path='./data', filecount=0):
        self.filecount = filecount
        self.data_path = data_path

    def download_images(self, urls):
        for url in urls:
            urllib.request.urlretrieve(url, f'{self.data_path}/ic_{self.filecount}.png')
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

