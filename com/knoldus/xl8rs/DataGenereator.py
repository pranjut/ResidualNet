import tensorflow as tf
import numpy as np
import cv2
import os


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, labels, image_dir, batch_size=32,
                 img_h=256, img_w=512, shuffle=True):
        self.list_ids = list_ids
        self.labels = labels
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids)) / self.batch_size)

    def __getitem__(self, index):
        'generate one batch of data'
        print("Generating data .....")
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # get list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(list_ids_temp)
        # return data
        return X, y

    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 1))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 4))

        for idx, id in enumerate(list_ids_temp):
            file_path = os.path.join(self.image_dir, id)
            image = cv2.imread(file_path, 0)
            image_resized = cv2.resize(image, (self.img_w, self.img_h))
            image_resized = np.array(image_resized, dtype=np.float64)
            # standardization of the image
            image_resized -= image_resized.mean()
            image_resized /= image_resized.std()

            mask = np.empty((self.img_h, self.img_w, 4))

            for idm, image_class in enumerate(['1', '2', '3', '4']):
                rle = self.labels.get(id + '_' + image_class)
                # if there is no mask create empty mask
                if rle is None:
                    class_mask = np.zeros((1600, 256))
                else:
                    class_mask = self.rle_to_mask(rle, width=1600, height=256)

                class_mask_resized = cv2.resize(class_mask, (self.img_w, self.img_h))
                mask[..., idm] = class_mask_resized

            X[idx,] = np.expand_dims(image_resized, axis=2)
            y[idx,] = mask

        # normalize Y
        y = (y > 0).astype(int)
        return X, y

    def rle_to_mask(self, rle_string, height, width):
        rows, cols = height, width
        if rle_string == -1:
            return np.zeros((height, width))
        else:
            rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
            rlePairs = np.array(rleNumbers).reshape(-1, 2)
            img = np.zeros(rows * cols, dtype=np.uint8)
            for index, length in rlePairs:
                index -= 1
                img[index:index + length] = 255
            img = img.reshape(cols, rows)
            img = img.T
            return img