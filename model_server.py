# model_server.py
# Created by: Dale Best
# Created on: April 30th, 2022

# The new model server script that will handle training the models

# The included python libraries
import os
import time
import base64
import datetime
from typing import List

# The local python libraries
from generate_fake_image import augmentation
from image_classifier import ImageClassifier
import optical_character_recognition as ocr

# The external python libraries installed via pip
import cv2
import numpy
import requests
from rich import print
import tensorflow as tf
import tensorflow_hub as hub


class ModelServer(object):
    _BATCH_SIZE: int = 32
    _FAKE_COUNT: int = 250

    DATA_PATH: str = "data"
    IMAGE_PATH: str = "data/images"
    MODEL_PATH: str = "data/models"

    _IMAGE_SIZE: int = 512
    _IMAGE_SHAPE: tuple = (512, 512)
    _MODEL_NAME: str = "efficientnetv2-xl-21k"
    _MODEL_HANDLE: str = "https://tfhub.dev/google/imagenet/" \
                         "efficientnet_v2_imagenet21k_xl/feature_vector/2"

    _NUM_EPOCHS: int = 10
    _AUGMENT_DATA: bool = False

    _PENDING_FORM_IMAGE_URL: str = "/api/pending/get-form/%d/image/"

    def __init__(self,
                 api_url: str,
                 fake_images: int = _FAKE_COUNT,
                 epochs: int = _NUM_EPOCHS,
                 augment_data: bool = _AUGMENT_DATA):

        self._api_url: str = api_url
        self._fake_images: int = fake_images
        self._epochs: int = epochs
        self._augment_data: bool = augment_data

        # TODO: Create a state json that is loaded here,
        # Currently it will just check if the images directory is there
        self._base_forms: dict = {}
        self._trained_forms: List[str] = self._get_form_image_list()

        self._classified_forms: List[int] = []

        self._model_path: str = self._get_latest_model_path()
        print("Loading latest model:", self._model_path)
        self._image_classifier = ImageClassifier(self._model_path,
                                                 self._IMAGE_SIZE)

        return

    def run(self) -> None:
        server_running: bool = True
        while server_running:
            try:
                found_new_forms: bool = self.update_base_forms()
                new_pending_forms: List[int] = self.get_pending_forms()

                if found_new_forms:
                    self.create_fake_images()
                    self.run_training()
                    self._update_classifier()

                for new_pending_form in new_pending_forms:
                    self.classify_form(new_pending_form)

                time.sleep(1)

            except Exception as server_error:
                server_running = False
                print("Got a model server error of:", server_error)

        return

    # Step 1: Check for new forms and create a dict of the new forms
    def update_base_forms(self) -> bool:
        new_forms_found: bool = False
        current_base_forms = self._get_form_dict()
        for base_form in current_base_forms:
            if base_form not in self._trained_forms:
                new_forms_found = True
        self._base_forms = current_base_forms
        self._trained_forms = list(self._base_forms.keys())
        return new_forms_found

    # Step 2: Create fake images from base forms
    def create_fake_images(self) -> None:
        form_image_list = self._get_form_image_list()
        for form_name in self._base_forms:
            if form_name not in form_image_list:
                self._create_fake_form_images(form_name)
                print("Created the fake images for:", form_name)
            else:
                pass

        return

    def run_training(self) -> None:
        # TODO: Clean up this function
        train_dataset: tf.data.Dataset
        test_dataset: tf.data.Dataset
        train_size: int
        test_size: int

        train_dataset, test_dataset, train_size, test_size = self._get_datasets(self._IMAGE_SHAPE, self._BATCH_SIZE)

        model_config = self._create_model_config()
        model = tf.keras.Sequential(model_config)
        model.build((None,) + self._IMAGE_SHAPE + (3,))
        model.summary()

        optimizer_config = tf.keras.optimizers.SGD(learning_rate=0.005,
                                                   momentum=0.9)
        loss_config = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                              label_smoothing=0.1)
        model.compile(loss=loss_config,
                      optimizer=optimizer_config,
                      metrics=['accuracy'])

        model.fit(train_dataset,
                  epochs=self._epochs,
                  steps_per_epoch=train_size // self._BATCH_SIZE,
                  validation_data=test_dataset,
                  validation_steps=test_size // self._BATCH_SIZE)

        timestamp: int = int(datetime.datetime.utcnow().timestamp())
        model_path = "data/models/%s_%d" % (self._MODEL_NAME, timestamp)
        tf.saved_model.save(model, model_path)

        # Save the list of class names
        class_name_path: str = model_path + "/class_names.txt"
        with open(class_name_path, "w+") as class_name_file:
            class_name_list = list(self._base_forms.keys())
            class_name_list.sort()
            for class_name in class_name_list:
                class_name_file.write(class_name + "\n")
        return

    # Step 5: Query API for new filled out forms
    def get_pending_forms(self) -> List[int]:
        new_pending_forms: List[int] = []
        pending_form_list: List[int] = self._get_pending_form_list()

        for pending_form in pending_form_list:
            if pending_form not in self._classified_forms:
                new_pending_forms.append(pending_form)
            else:
                pass

        return new_pending_forms

    # Step 6: Classify the form
    def classify_form(self, form_id: int) -> None:
        form_image: numpy.ndarray = self._get_pending_form_image(form_id)

        form_type = self._image_classifier.make_inference(form_image)
        if form_type != "":
            print("Classified pending form:", form_id, "as", form_type)
            ocr_results: dict = self._run_ocr(form_type, form_image)
            print("Got the ocr results of:", ocr_results)
            self._update_api(ocr_results)
            print("Updated the api!")
            self._classified_forms.append(form_id)
        return

    def _get_form_dict(self) -> dict:
        form_dict: dict = {}
        try:
            response = requests.get(self._api_url + "/api/base/all-forms/")
            if response.ok:
                response_json: dict = response.json()
                form_names = list(response_json.keys())
                for form_name in form_names:
                    form_list: list = response_json.get(form_name)
                    form_info: dict = form_list[0]
                    form_bounding_boxes: dict = form_info.get("bboxes")
                    form_dict.update({form_name: form_bounding_boxes})

        except Exception as get_form_error:
            print("Got an error getting the forms:", get_form_error)

        return form_dict

    def _get_form_image_list(self) -> List[str]:
        image_list: List[str] = []
        try:
            image_list = os.listdir(self.IMAGE_PATH)

        except Exception as image_list_error:
            print("Did not find the image directory!:", image_list_error)

        return image_list

    def _create_fake_form_images(self, form_name: str) -> None:
        print("Creating fake images for:", form_name)
        form_image_directory: str = self.IMAGE_PATH + "/" + form_name

        if not os.path.exists(form_image_directory):
            os.makedirs(form_image_directory)

        form_image_info: dict = self._get_form_image_info(form_name)

        base_image_path: str = form_image_directory + "/" + form_name + ".jpg"
        form_image: numpy.ndarray = self._create_form_image(form_image_info)
        cv2.imwrite(base_image_path, form_image)

        creator_info: dict = {"bboxes": form_image_info.get("bboxes"),
                              "imageX": form_image_info.get("image_width"),
                              "imageY": form_image_info.get("image_height")}

        # Generate the fake form images
        augmentation(form_name,
                     form_image_directory,
                     base_image_path,
                     creator_info,
                     self._fake_images)
        return

    def _get_form_image_info(self, form_name: str) -> dict:
        form_data = {}
        try:
            get_image_extension = "/api/base/get-form/%s/image/" % form_name
            get_image_url: str = self._api_url + get_image_extension
            print("Got an image url of:", get_image_url)
            response = requests.get(get_image_url)

            if response.ok:
                form_data: dict = response.json()

            else:
                print("Was unable to get form data because:", response.reason)
        except Exception as get_data_error:
            print("Got an error getting the form data:", get_data_error)

        return form_data

    @staticmethod
    def _create_form_image(form_data: dict) -> numpy.ndarray:
        frame_string: str = form_data.get("frame_string")
        image_height: int = form_data.get("image_height")
        image_width: int = form_data.get("image_width")
        image_depth: int = form_data.get("image_depth")

        encoded_array = frame_string.encode()
        decoded_bytes = base64.b64decode(encoded_array)
        decoded_buffer = numpy.frombuffer(decoded_bytes, numpy.uint8)
        image_shape: tuple = (image_height, image_width, image_depth)
        form_image = numpy.resize(decoded_buffer, image_shape)
        return form_image

    def _build_datasets(self, subset, image_size):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.IMAGE_PATH,
            validation_split=.30,
            subset=subset,
            label_mode="categorical",
            shuffle=True,
            seed=123,
            image_size=image_size,
            batch_size=1)

    def _get_datasets(self, image_size: tuple, batch_size: int) -> (tf.data.Dataset, tf.data.Dataset, int, int):
        # Seed needs to provided when using validation_split and shuffle = True.
        # A fixed seed is used so that the validation set is stable across runs.

        train_dataset = self._build_datasets("training", image_size)
        train_size = train_dataset.cardinality().numpy()
        train_dataset = train_dataset.unbatch().batch(batch_size)
        train_dataset = train_dataset.repeat()

        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        preprocessing_model = tf.keras.Sequential([normalization_layer])

        train_dataset = train_dataset.map(lambda images, labels:
                                          (preprocessing_model(images), labels))

        test_data_set = self._build_datasets("validation", image_size)
        test_size = test_data_set.cardinality().numpy()
        test_data_set = test_data_set.unbatch().batch(batch_size)
        test_data_set = test_data_set.map(lambda images, labels:
                                          (normalization_layer(images), labels))

        return train_dataset, test_data_set, train_size, test_size

    def _create_model_config(self) -> list:
        num_classes: int = len(self._base_forms)
        input_layer_shape = self._IMAGE_SHAPE + (3,)
        regularizers = tf.keras.regularizers.l2(0.0001)

        input_layer = tf.keras.layers.InputLayer(input_shape=input_layer_shape)
        model_handle_layer = hub.KerasLayer(self._MODEL_HANDLE, trainable=False)
        dropout_layer = tf.keras.layers.Dropout(rate=0.2)
        dense_layer = tf.keras.layers.Dense(num_classes,
                                            kernel_regularizer=regularizers)

        model_config = [  # preprocessing_model,
            input_layer,
            model_handle_layer,
            dropout_layer,
            dense_layer]
        return model_config

    @staticmethod
    def _create_preprocessing_model(augment_data: bool = False):
        flip_mode: str = "horizontal"
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        preprocessing_model = tf.keras.Sequential([normalization_layer])
        if augment_data:
            preprocessing_model.add(tf.keras.layers.RandomRotation(40))
            preprocessing_model.add(tf.keras.layers.RandomZoom(0.2, 0.2))
            preprocessing_model.add(tf.keras.layers.RandomTranslation(0, 0.2))
            preprocessing_model.add(tf.keras.layers.RandomTranslation(0.2, 0))
            preprocessing_model.add(tf.keras.layers.RandomFlip(mode=flip_mode))

        return preprocessing_model

    def _update_classifier(self) -> None:
        self._model_path: str = self._get_latest_model_path()
        self._image_classifier = ImageClassifier(self._model_path,
                                                 self._IMAGE_SIZE)
        return

    def _get_latest_model_path(self) -> str:
        latest_model_path: str = ""
        try:
            model_name_list: List[str] = os.listdir(self.MODEL_PATH)
            latest_model_name = model_name_list[-1]
            latest_model_path = self.MODEL_PATH + "/" + latest_model_name
        except Exception as image_list_error:
            print("Did not find the image directory!:", image_list_error)
        return latest_model_path

    def _get_pending_form_list(self) -> List[int]:
        current_state: List[int] = []
        try:
            # TODO: Need this API call to filter out already classified forms
            # TODO: Need this API call to filter out OCR'd forms
            all_form_id_url: str = self._api_url + "/api/pending/all-form-ids/"
            response = requests.get(all_form_id_url)

            if response.ok:
                response_json: dict = response.json()
                form_types: list = response_json.get("all_forms")
                current_state = form_types

        except Exception as get_list_error:
            print("Had an error getting current list:", get_list_error)

        return current_state

    def _get_pending_form_image(self, form_id: int) -> numpy.ndarray:

        form_image_endpoint: str = self._PENDING_FORM_IMAGE_URL % form_id
        form_image_url: str = self._api_url + form_image_endpoint

        form_image_response: requests.Response
        form_image_response = requests.get(form_image_url)

        form_image_data = form_image_response.json()
        form_image = self._create_form_image(form_image_data)

        return form_image

    def _run_ocr(self, form_type: str, form_image: numpy.ndarray) -> dict:
        ocr_results: dict = {}
        try:
            # TODO: Want a more efficient API endpoint for this
            base_info_url: str = self._api_url + "/api/base/get-form/%s/image/"
            base_info_url %= form_type

            form_info_response = requests.get(base_info_url)
            form_info: dict = form_info_response.json()
            form_bounding_boxes: dict = form_info.get("bboxes")

            base_form_image: numpy.ndarray = self._create_form_image(form_info)
            base_form_height: int = form_info.get("image_height")
            base_form_width: int = form_info.get("image_width")

            aligned_image: numpy.ndarray = ocr.align_images(form_image,
                                                            base_form_image,
                                                            base_form_height,
                                                            base_form_width)

            extracted_text: dict = ocr.extract_text(aligned_image,
                                                    form_bounding_boxes)
            ocr_results = {"ocr_results": extracted_text,
                           "classified_form": form_type}

        except Exception as ocr_error:
            print("Got an error running ocr of:", ocr_error)

        return ocr_results

        # Step 4:

    def _update_api(self, ocr_results: dict) -> None:
        ocr_result_url: str = self._api_url + "/api/documents/send-ocr-results/"
        requests.post(ocr_result_url, json=ocr_results)
        return


if __name__ == '__main__':
    epochs: int = 1
    image_count: int = 1000
    augment_data: bool = False

    # Server URL to get the images for initial testing
    # url: str = "http://35.239.234.232"

    # Use this URL when testing locally
    url: str = "http://127.0.0.1:8000"

    if os.getenv("Api_Url"):
        url = os.getenv("Api_Url")

    if os.getenv("Image_Count"):
        try:
            image_count: int = int(os.getenv("Image_Count"))

        except Exception as load_error:
            print("Had an error loading image count setting:")

    if os.getenv("Epochs"):
        try:
            epochs: int = int(os.getenv("Epochs"))

        except Exception as load_error:
            print("Had an error loading epoch setting:")

    if os.getenv("Augment_Data"):
        try:
            augment_data: bool = os.getenv("Augment_Data") == "True"

        except Exception as load_error:
            print("Had an error loading epoch setting:")

    model_server = ModelServer(url, image_count, epochs, augment_data)
    model_server.run()

    print("Model Server finished")