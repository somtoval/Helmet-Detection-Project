import os
import sys
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from helmet.logger import logging
from helmet.exception import HelmetException
from helmet.ml.feature.helmet_detection import HelmetDetection # This is the class that inherits pytorch dataset module to create our dataset
from helmet.constants import *
from helmet.utils import save_object
from helmet.entity.config_entity import DataTransformationConfig
from helmet.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def number_of_classes(self):
        try:
            coco = COCO(os.path.join(self.data_ingestion_artifact.train_file_path, ANNOTATIONS_COCO_JSON_FILE))
            ''' This so "coco.cats" whill give use something like this 
            coco_cats_sample = {
                1: {'id': 1, 'name': 'person', 'supercategory': 'human'},
                2: {'id': 2, 'name': 'car', 'supercategory': 'vehicle'},
                3: {'id': 3, 'name': 'dog', 'supercategory': 'animal'},
            }'''
            categories = coco.cats 
            '''categories.items() would give us a list of all the items of the "categories" dictionary so we can loop, it's something like this;
            dict_items([(1, {'id': 1, 'name': 'person', 'supercategory': 'human'}), (2, {'id': 2, 'name': 'car', 'supercategory': 'vehicle'}), (3, {'id': 3, 'name': 'dog', 'supercategory': 'animal'})])
            So for each iteration we take the the 2nd index which would be for instance "{'id': 1, 'name': 'person', 'supercategory': 'human'}" then we access the dictionary as normal which is passing in the key you want to get the value and so we passed in name and got for example "person"
            This would happen for all of the categories to get a simple list with the names of the categories we have
            '''
            classes = [i[1]['name'] for i in categories.items()]
            # Just checking the lengthe of the list "class"
            n_classes = len(classes)

            # The return value of this function is the number of categories our annotation data caries
            return n_classes
        except Exception as e:
            raise HelmetException(e, sys) from e
        
    # This is the transformation of our data, if we have an argument "train" which is set to default as False, anytime we want to use it if this argument is False it just resizes the data and return in tensors but it is a train data that means we would do some more operations on it
    def get_transforms(self, train=False):
        try: 
            # If we specify train as True
            if train:
                # Creating a transformation object and performing operations on it using the varibles in our constants directory file and then returns tensor output
                transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE),
                    A.HorizontalFlip(p=HORIZONTAL_FLIP),
                    A.VerticalFlip(p=VERTICAL_FLIP),
                    A.RandomBrightnessContrast(p=RANDOM_BRIGHTNESS_CONTRAST),
                    A.ColorJitter(p=COLOR_JITTER),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format=BBOX_FORMAT))
            # If we specify train as Fale
            else:
                # Creating a transformation object and performing operation and just resizing it and return tensor output
                transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE), 
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format=BBOX_FORMAT))

            # Returns the transformtion object
            return transform
        except Exception as e:
            raise HelmetException(e, sys) from e

    # Initiating the data transformation and hoping to get an output which will be our DataTransformationArtifacts
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            
            # To get the number of classes we have using our created method above
            n_classes = self.number_of_classes()
            print(n_classes)
            logging.info(f"Total number of classes: {n_classes}")

            # Creating our dataset officically using the pytorch dataset module, we have implemented HelmetDetection class which inherits from it, it's contained in the "ml" directory
            # This class takes in the the root directory of the data ingestion which we specified as ROOT_DIR in our data transformation entity, it also takes in the split here it is the train split so it is "train",
            # it joins thes two things and uses this to find the location of our annotations.json file, then the last arguments for transformations using our created transformation class
            train_dataset = HelmetDetection(root=self.data_transformation_config.ROOT_DIR,
                                            split=self.data_transformation_config.TRAIN_SPLIT,
                                            transforms=self.get_transforms(train=True))
            # Logging that we have prepared our dataset
            logging.info(f"Training dataset prepared")

            # We do the same here to obtain our test dataset
            test_dataset = HelmetDetection(root=self.data_transformation_config.ROOT_DIR,
                                           split=self.data_transformation_config.TEST_SPLIT,
                                           transforms=self.get_transforms(train=False))
            logging.info(f"Testing dataset prepared")

            # Now using our utility functions in utils directory we will save our transformed train and test datasets respectively in the specified location in our configuration
            save_object(self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, train_dataset)
            save_object(self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH, test_dataset)
            logging.info("Saved the train transformed object")

            # We now create our DataTransformationArtifacts as specified in our artifact entity and then we pass in the file path of the train dataset object and the test dataset object
            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                transformed_test_object=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH,
                number_of_classes=n_classes)

            logging.info("Exited the initiate_data_transformation method of Data transformation class")

            # We return the data transformation artifact which contains the file path of the train and test dataset objects respectively
            return data_transformation_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e




