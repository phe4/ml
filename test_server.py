import os
import time
import json
import base64
import datetime
from datetime import date
from pytz import timezone
from typing import List
import shutil

# The local python libraries
from generate_fake_image import augmentation
from TF_image_classifier import ImageClassifier
import optical_character_recognition as ocr

# The external python libraries installed via pip
import cv2
import numpy
import requests
from rich import print

# tflite library
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader


class ModelServer(object):
    _BATCH_SIZE: int = 32
    _FAKE_COUNT: int = 250

    DATA_PATH: str = "data"
    IMAGE_PATH: str = "data/images"
    MODEL_PATH: str = "data/models"

    _IMAGE_SIZE: int = 240
    _IMAGE_SHAPE: tuple = (240, 240)
    _MODEL_NAME: str = "tflite_model_maker"
    _MODEL_HANDLE: str = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2"

    _NUM_EPOCHS: int = 10
    _AUGMENT_DATA: bool = False
    _IMAGE_QUALITY: int = 60
    _FORM_IMG: List[str] = []

    _PENDING_FORM_IMAGE_URL: str = "/api/pending/get-form/%d/image/"

    def __init__(self,
                 fake_images: int = _FAKE_COUNT,
                 epochs: int = _NUM_EPOCHS,
                 augment_data: bool = _AUGMENT_DATA,
                 image_quality: int = _IMAGE_QUALITY,
                 form_img: List[str] = _FORM_IMG,
                 complex_background: float = 0.5):

        self._fake_images: int = fake_images
        self._epochs: int = epochs
        self._augment_data: bool = augment_data
        self._image_quality: int = image_quality
        self._form_img: List[str] = form_img
        
        self._api_url: str = "http://localhost:8000"
        self._test_forms: List[str] = []
        self._time_used: float = 0.0
        self.background_path: str = "data/backgrounds"
        self.complex_background: float = complex_background

        self._base_forms: dict = {}
        self._trained_forms: List[str] = self._get_form_image_list()

        self._classified_forms: List[int] = []

        return

    def run(self) -> None:
        try:
            current_img_data: List[str] = []
            for form_img in self._form_img:
                form_name = form_img.replace(".jpg", "")
                current_img_data.append(form_name)

            self._test_forms = current_img_data

            found_new_forms: bool = self.update_base_forms()
            if found_new_forms: 
                self.create_fake_images()
            else:
                print("All form images exists.")

            # Preprocess: move image/xml files to test folder:
            print("Moving data files...")
            for form_name in self._test_forms:
                shutil.move("data/images/image_files/%s" % form_name, "data/images/image_files/test")
                shutil.move("data/images/xml_files/%s" % form_name, "data/images/xml_files/test")


            print("==========Using-TensorFlow-lite==========")
            self.run_training_tflite()
            print("=========================================")

            # Postprocessing: move image/xml files back:
            print("Moving data files...")
            for form_name in self._test_forms:
                shutil.move("data/images/image_files/test/%s" % form_name, "data/images/image_files")
                shutil.move("data/images/xml_files/test/%s" % form_name, "data/images/xml_files")

        except Exception as running_error:
            print("Got a model running error of:", running_error)

        return

    # Check for new forms and create a dict of the new forms
    def update_base_forms(self) -> bool:
        new_forms_found: bool = False
        current_base_forms = self._get_form_dict()
        for base_form in current_base_forms:
            # if there is no corresponding imaged folder in the generated images folder
            if base_form not in self._trained_forms:
                new_forms_found = True
        self._base_forms = current_base_forms
        self._trained_forms = list(self._base_forms.keys())
        return new_forms_found

    # Create fake images from base forms
    def create_fake_images(self) -> None:
        form_image_list = self._get_form_image_list()
        for form_name in self._base_forms:
            if form_name not in form_image_list:
                self._create_fake_form_images(form_name)
                print("Created the fake images for:", form_name)
            else:
                pass

        return

    def run_training_tflite(self) -> None: 
        """
            Create and train model with tflite_model_maker and save it as Saved_Model.
            See code at: https://www.tensorflow.org/lite/models/modify/model_maker/image_classification
            TODO:
                Step 1: load fake images, with DataLoader
                Step 2: split dataset
                Step 3: Create model and train, with image_classifier.create()
                Step 4: evaluate the model
                Step 5: save the model and the class names, with quantization_config
        """  
        # Step 1: load fake images    
        print("Loading Images......")
        data = DataLoader.from_folder("data/images/image_files/test")
        print("Loaded %d images" % (data.size))

        # Step 2: split dataset
        train_dataset, test_dataset = data.split(0.7)
        
        # Step 3: Create model and train
        print("Training......")

        start_new = time.time()

        # change the model spec here:
        model = image_classifier.create(train_dataset,
                                model_spec=model_spec.get('efficientnet_lite1'),
                                epochs=self._epochs)                        
        end_new = time.time()
        self._time_used = end_new - start_new

        # Step 4: evaluate the model
        loss, accuracy = model.evaluate(test_dataset)
        print("Got a model loss of:", loss, "and accuracy of:", accuracy)

        # Step 5: save the model
        config = QuantizationConfig.for_float16()
        config.representative_data = train_dataset
        timestamp: int = int(datetime.datetime.utcnow().timestamp())
        model_path = "data/models" 
        model_file_name = "%s_%d" % (self._MODEL_NAME, timestamp)
        class_names_path = model_path + '/' + model_file_name
        model.export(export_dir = model_path,
            #  tflite_filename='label_model_1.tflite',
             quantization_config=config,
             export_format = ExportFormat.SAVED_MODEL,
             saved_model_filename = model_file_name)
        model.export(export_dir = class_names_path, 
                    export_format=ExportFormat.LABEL,
                    label_filename='class_names.txt')
        
        # Step 6: json file for the training record
        self.save_info(img_count=data.size, path_to_class_file=class_names_path, avg_loss=loss, avg_accuracy=accuracy)

        return
    
    # Save model info to json
    def save_info(self, img_count: int, path_to_class_file: str, avg_loss, avg_accuracy) -> None:
        """Save the information of the model just trained.

        Args:
            img_count: how many images used for training in total.
            path_to_class_file: path to the saved class_names.txt file.
            Accuracy: average accuracy.
            Loss: average loss.

        Result including:
            Complete_time: When the training was done.
            Class_names: What are the classes the model was trained on.
            Total_images: How many fake images were used to create it.
            Epoch: How many epochs.
            Accuracy and loss: average accuracy and loss
            Training time: self._time_used
            Imaged_quality: self._image_quality
        """
        today = date.today()
        _date = today.strftime("%b-%d-%Y")
        print("Date =", _date)
        tz = timezone('EST')
        now = datetime.datetime.now(tz)
        _time = now.strftime("%H:%M:%S")
        print("EST =", _time)

        _date_time = _date + ', ' + _time
        _epochs = self._epochs
        _image_count = img_count
        _path_to_classes = path_to_class_file + "/class_names.txt"

        f = open(_path_to_classes)
        content = f.read()
        classnames = content.split("\n")
        f.close()

        model_info = {
            "Complete_time" : _date_time,
            "Epoch" : _epochs,
            "Total_images" : _image_count,
            "Class_names_file" : _path_to_classes,
            "Class_names" : classnames,
            "Average_loss" : avg_loss,
            "Average_accuracy" : avg_accuracy,
            "Training_time" : self._time_used,
            "Imaged_quality" : self._image_quality
        }
        json_path = path_to_class_file + "/model_info.json"
        with open(json_path, "w") as out_file:
            json.dump(model_info, out_file, indent=4)

        return

    # Classify the form
    def classify_form(self, form_image) -> dict:
        """
            Classify the form with the trained model.
            Args:
                form_image: the image of the form to be classified.
            Returns:
                A dictionary with the form id and the class name.
        """

        model_path: str = self._get_latest_model_path()
        print("Loading latest model:", model_path)

        m_spec = model_spec.get('efficientnet_lite1')
        _image_classifier = ImageClassifier(model_path=model_path, use_augmentation=False, model_spec=m_spec)

        form_type = _image_classifier.make_inference(form_image)
        f_type = form_type["detected_class"]
        print(f_type)

        if form_type != "":
            ocr_results: dict = self._run_ocr(f_type, form_image)

        return ocr_results
    
    def _run_ocr(self, form_type: str, form_image: numpy.ndarray) -> dict:
        """
            Run OCR on the form image and return the results
            Args:
                form_type: the type of the form
                form_image: the image array of the form
            Returns:
                A dictionary with the form id and the ocr results.
        """
        ocr_results: dict = {}
        try:
            # server based
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

    def _get_form_dict(self) -> dict:
        """Get the form dictionary from the API/local json file"""
        form_dict: dict = {}
        try:            
            # If connected to server:
            response = requests.get(self._api_url + "/api/base/all-forms/")
            if response.ok:
                response_json: dict = response.json()

                form_names = list(response_json.keys())
                for form_name in form_names:
                    if form_name in self._test_forms:
                        form_list: list = response_json.get(form_name)
                        form_info: dict = form_list[0]
                        form_bounding_boxes: dict = form_info.get("bboxes")
                        form_dict.update({form_name: form_bounding_boxes})

        except Exception as get_form_error:
            print("Got an error getting the forms:", get_form_error)

        return form_dict

    def _get_form_image_list(self) -> List[str]:
        """
            Get the list of form images
            Returns:
                A list of form images
        """
        image_list: List[str] = []
        try:
            image_list = os.listdir(self.IMAGE_PATH + "/image_files")

        except Exception as image_list_error:
            print("Did not find the image directory!:", image_list_error)

        return image_list

    def _create_fake_form_images(self, form_name: str) -> None:
        """
        Creates fake form images
        Args:
            form_name: the name of the form
        """
        print("Creating fake images for:", form_name)
        form_image_directory: str = self.IMAGE_PATH + "/image_files/" + form_name
        form_xml_directtory: str = self.IMAGE_PATH + "/xml_files/" + form_name

        if not os.path.exists(form_image_directory):
            os.makedirs(form_image_directory)
        if not os.path.exists(form_xml_directtory):
            os.makedirs(form_xml_directtory)
        if not os.path.exists(self.IMAGE_PATH + "/image_files/test"):
            os.makedirs(self.IMAGE_PATH + "/image_files/test")
            os.makedirs(self.IMAGE_PATH + "/xml_files/test")

        form_image_info: dict = self._get_form_image_info(form_name)

        # test edit
        print("img info get!!!")

        base_image_path: str = form_image_directory + "/" + form_name + ".jpg"
        form_image: numpy.ndarray = self._create_form_image(form_image_info)
        # compression the original image:
        cv2.imwrite(base_image_path, form_image, [int(cv2.IMWRITE_JPEG_QUALITY), self._image_quality])

        creator_info: dict = {"bboxes": form_image_info.get("bboxes"),
                              "imageX": form_image_info.get("image_width"),
                              "imageY": form_image_info.get("image_height")}

        # Generate the fake form images
        create_xml: bool = True
        augmentation(form_name,
                     form_image_directory,
                     base_image_path,
                     self.background_path,
                     creator_info,
                     self._fake_images,
                     self._image_quality,
                     create_xml,
                     self.complex_background)

        return

    def _get_form_image_info(self, form_name: str) -> dict:
        """
        Get the form image info from the server/local json file
        Args:
            form_name: the name of the form
        Returns:
            A dictionary with the form image info
        """
        form_data = {}
        try:
            # request image info
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
        """
        Creates a form image from the form data
        Args:
            form_data: the form data
        Returns:
            A numpy array of the form image
        """
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

    def _get_latest_model_path(self) -> str:
        """
        Gets the latest model path
        Returns:
            The latest model path
        """
        latest_model_path: str = ""
        try:
            model_name_list: List[str] = os.listdir(self.MODEL_PATH)
            latest_model_name = model_name_list[-1]
            latest_model_path = self.MODEL_PATH + "/" + latest_model_name
        except Exception as image_list_error:
            print("Did not find the image directory!:", image_list_error)
        return latest_model_path

def download_form_req_json() -> None:
    """ Download all base form request json files for testing """
    # local url
    try:
        all_form_response = requests.get("http://127.0.0.1:8000/api/base/all-forms/")
        if all_form_response.ok:
            all_form_json: dict = all_form_response.json()
            print("Got updated All-form.")
        with open("data/dict_form_req.json", 'w', encoding='utf-8') as dict_form_req:
            json.dump(all_form_json, dict_form_req, ensure_ascii=False, indent=4)
        form_list: list = list(all_form_json.keys()) 
        for form_name in form_list:
            form_img_response = requests.get("http://127.0.0.1:8000/api/base/get-form/%s/image/" % form_name)
            if form_img_response.ok:
                form_img_json: dict = form_img_response.json()
                print("Got updated form info for ", form_name)
            with open("data/form_img_req/%s.json" % form_name, 'w', encoding='utf-8') as form_img_req:
                json.dump(form_img_json, form_img_req, ensure_ascii=False, indent=4)
    except Exception as download_error:
        print("Got an error when downloading json files: ", download_error)

    return    

def clear_stored_data() -> None:
    """ Clear all the stored data in the data folder.
    """
    shutil.rmtree("data/images/image_files")
    print("Image_files removed...")

    shutil.rmtree("data/images/xml_files")
    print("Xml_files removed...")

    print("Temp data cleared!")

    return

def proceed_config(clear_data, epoch_list, 
                    image_count, augment_data, image_quality_list,
                    manual_forms, complex_background):
    """Deal with the config parameters and start the test.

    Args:
        clear_data (bool): remove the local files for fake images
        epoch_list (List of int): number of epochs
        image_count (int): number of images for each form
        augment_data (bool): augment data
        image_quality_list (List of int): image quality of the fake images
        manual_forms (list of string): user's choice of forms, length > 1
    """
    if clear_data:
        clear_stored_data()

    # Load local json file
    forms_path = "data/dict_form_req.json"
    with open(forms_path) as json_file:
        form_dict: dict = json.load(json_file)
    form_list: list = list(form_dict.keys()) 

    # find all combinations
    if len(manual_forms) > 1:
        # keep this, since we want to train the model with all forms
        for form_s in manual_forms:
            for epochs in epoch_list:
                print("Running test for combination", str(form_s))
                print("Epoch = ", epochs)
                for image_quality in image_quality_list:
                    print("Image quality = ", image_quality)
                    model_server = ModelServer(image_count, epochs, augment_data, image_quality, form_s, complex_background)
                    model_server.run()
    else: 
        # can be removed
        print("traning %d forms--" % (len(form_list)))
        for epochs in epoch_list:
            print("Running test for combination", str(form_list))
            print("Epoch = ", epochs)
            for image_quality in image_quality_list:
                print("Image quality = ", image_quality)
                model_server = ModelServer(image_count, epochs, augment_data, image_quality, form_list, complex_background)
                model_server.run()

def request_run() -> dict:
    """Request the config parameters and start the test.
    """
    try:
        response = requests.get("http://localhost:8000/api/run_model")
        if response.ok:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return {}

def get_pending_form_info(api_url: str, form_id: int) -> dict:
    """Get the pending form info from the server.
    """
    try:
        request_url = api_url + "api/pending/get-form/%d/image/" % form_id
        response = requests.get(request_url)
        if response.ok:
            form_image_response = response.json()
            form_image = ModelServer._create_form_image(form_image_response)
    except Exception as get_form_error:
        print("Error when getting form info: ", get_form_error)
    
    return form_image

def new_pending_form(api_url: str)-> dict:
    """Server: Request a new pending form from the Api. Local: manual input image path. Both work.
    """
    try:
        # read classfication history, create file if not exist
        if not os.path.exists("data/classification_history.txt"):
            f = open("data/classification_history.txt", 'w')
            f.close()
        with open("data/classification_history.txt", 'r') as classified_images:
            classified_images_list = classified_images.readlines()
        classified_images_list = [x.strip() for x in classified_images_list]
        
        # create a fake ModelServer instance
        model_server = ModelServer(1, 1, False, 1, ["fake"], False)
        inf_result: dict = {}

        # from server
        response = requests.get(api_url + "api/pending/all-form-ids/")
        if response.ok:
            response_json = response.json()
            form_ids: list = response_json["all_forms"]
            print("Form ids: ", form_ids)
        for form_id in form_ids:
            if str(form_id) not in classified_images_list:
                print("Found a new pending form: ", form_id)
                form_image: numpy.ndarray = get_pending_form_info(api_url, form_id)
                print("Got the image of the form: ", form_id)
                inf_result["form_id"] = model_server.classify_form(form_image)
                # add the form_id to the classified_images.txt
                with open("data/classification_history.txt", "a") as classified_images:
                    classified_images.write(form_id + "\n")

    except Exception as new_pending_error:
        print("Got an error when requesting new pending form: ", new_pending_error)                  

    return inf_result

def server_pipeline():
    """Server pipe_config.
    """
    server_running = True
    saved_pipe_config = {}
    pipe_config = {}
    while server_running:
        try:
            # 1. server for config
            # get the config from the server
            with open("data/config.json") as json_file:
                pipe_config = json.load(json_file)
            # pipe_config = request_run()

            if pipe_config != saved_pipe_config:
                saved_pipe_config = pipe_config
                # construct_pipeline(pipe_config)
                proceed_config(**pipe_config)
                print("Pipeline finished!")

            # 2. server for classification and OCR
            pending_form_api = "http://localhost:8000/"
            pending_inf_result = new_pending_form(pending_form_api)
            if pending_inf_result:
                print("Analyze done: ", pending_inf_result)
            # save the result to local
            with open("data/pending_inf_result.json", "w") as pending_inf_result_file:
                json.dump(pending_inf_result, pending_inf_result_file)

            time.sleep(2)
        
        except Exception as e:
            print(e)
            server_running = False                  

def construct_pipeline(pipe_config):
    """Construct the pipeline for the server.
    """
    clear_data = pipe_config["clear_data"]
    epoch_list = pipe_config["epoch_list"]
    image_count = pipe_config["image_count"]
    augment_data = pipe_config["augment_data"]
    image_quality_list = pipe_config["image_quality_list"]
    manual_forms = pipe_config["manual_forms"]
    complex_background = pipe_config["complex_background"]

    proceed_config(clear_data, epoch_list, 
                    image_count, augment_data, image_quality_list,
                    manual_forms, complex_background)

if __name__ == '__main__':
    server_pipeline()

    print("Model Server finished")