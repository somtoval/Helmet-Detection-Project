import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm # This library is used to add progress bars to your loops in the terminal.
import torch
from torchvision import models
from torch.utils.data import DataLoader # Creates an iterator to load our dataset based on batchsize we specified
from helmet.logger import logging
from helmet.exception import HelmetException
from helmet.utils import load_object
from helmet.ml.models.model_optimiser import model_optimiser
from helmet.entity.config_entity import ModelTrainerConfig
from helmet.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                    model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifacts: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model trainer
        """

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    # Our model traning method, it takes in some parameter whose values are in the constant directory
    def train(self, model, optimizer, loader, device, epoch):
        try:
            # This line moves the model to the specified device. In PyTorch, a device can be a CPU (torch.device('cpu')) or a GPU (torch.device('cuda:0') or torch.device('cuda:1') for different GPUs).
            # Remember we would use pytorch fasterrcnn model just like we use scikitlearn's linear regression 
            model.to(device) 
            # This line sets the model to training mode. In PyTorch, some layers (like dropout or batch normalization) behave differently during training and evaluation. Calling model.train() is crucial to activate these layers for training. Conversely, you would use model.eval() when you're evaluating or testing your model to deactivate those layers.
            model.train() 
            # We define a list and dictionary to store our losses
            all_losses = []
            all_losses_dict = []

            # This iterating the images and targets in the data loader iterator which normally contains different batches, Now we are able to do this like this because we have only one batch so we just looped through the data loader since there is just one batch else we first have to loop through the batch and then loop through the images and targets that are there
            for images, targets in tqdm(loader):
                # This line iterates through each image in the images list and moves each image tensor to the specified device (device). The resulting images list will contain the same images but now residing on the specified device.
                images = list(image.to(device) for image in images)
                # This line iterates through each dictionary in the targets list. For each dictionary (which represents a target annotation), it creates a new dictionary where each value tensor is moved to the specified device (device). This is done to ensure that both input images and target annotations are on the same device.
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass: Here we pass our image which is a tensor into the model compute the model's loss
                loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
                # This line calculates the total loss by summing up individual losses stored in the loss_dict. The loss_dict typically contains various components of the total loss, such as classification loss, regression loss, etc., depending on the nature of the model.
                losses = sum(loss for loss in loss_dict.values())

                # This line creates a new dictionary (loss_dict_append) where the values are converted to Python scalars using the item() method. This is often done to facilitate further processing, such as appending the losses to a list or converting them to NumPy arrays.
                loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                # This line extracts the scalar value from the total losses tensor (losses). This is necessary because, in PyTorch, the result of a reduction operation like sum() is typically a scalar value wrapped in a tensor. Using .item() retrieves the actual Python scalar.
                loss_value = losses.item()
                
                # This line appends the total scalar loss value (sum of individual losses) to the all_losses list. This list is likely used to keep track of the overall training loss at each iteration or epoch.
                all_losses.append(loss_value)
                #  This line appends the dictionary loss_dict_append to the all_losses_dict list. This dictionary contains individual loss components, each converted to a Python scalar using item()
                all_losses_dict.append(loss_dict_append)
                
                # This condition checks if loss_value is not finite, meaning it is either NaN (Not a Number) or infinity.
                if not math.isfinite(loss_value):
                    # If the loss is not finite, this line prints a message indicating the problematic loss value and that the training is being stopped.
                    print(f"Loss is {loss_value}, stopping training")  # train if loss becomes infinity
                    # loss_dict contains information about individual loss components. 
                    print(loss_dict)
                    # Finally, the script exits with a status code of 1, indicating an abnormal termination. This will stop the execution of the training script.
                    sys.exit(1)
                
                # These lines are part of the standard training loop for a PyTorch model then exist in pytorch module.
                # This line zeroes the gradients of all model parameters. 
                optimizer.zero_grad()
                # This line computes the gradients of the model parameters with respect to the computed loss. The gradients are then used for updating the model parameters during optimization. 
                losses.backward()
                # This line updates the model parameters based on the computed gradients and the optimization algorithm. The optimizer's step() function is responsible for updating the weights according to the optimization strategy (e.g., SGD, Adam, etc.).
                optimizer.step()
            # Forming a dataframe from the all_losses dictionary
            all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing

            # The print statement you provided outputs a summary of the training statistics for a given epoch.
            print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                # Epoch {},: Prints the current epoch number. lr: {:.6f},: Prints the learning rate used in the optimizer. The optimizer.param_groups[0]['lr'] extracts the learning rate from the optimizer's parameter groups.
                epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
                # Prints the mean of the 'loss_classifier' component from the all_losses_dict. The 'loss_classifier' is assumed to be one of the components of the total loss.
                all_losses_dict['loss_classifier'].mean(),
                # Prints the mean of the 'loss_box_reg' component from all_losses_dict. This component typically represents the regression loss for bounding box predictions.
                all_losses_dict['loss_box_reg'].mean(),
                # loss_rpn_box: {:.6f},: Prints the mean of the 'loss_rpn_box_reg' component from all_losses_dict. This might be related to the region proposal network (RPN) bounding box regression loss.
                all_losses_dict['loss_rpn_box_reg'].mean(),
                # loss_object: {:.6f}": Prints the mean of the 'loss_objectness' component from all_losses_dict. This component is often related to objectness prediction in the model
                all_losses_dict['loss_objectness'].mean()
            ))

        except Exception as e:
            raise HelmetException(e, sys) from e
        
    # The collate_fn method you've provided is a custom collating function for your train DataLoader. This function is used to process the individual samples in a batch before they are passed to your model. It's particularly useful when dealing with datasets where each sample can have different sizes or structures.
    '''
        In a layman's way, we use a collate function in PyTorch's DataLoader to make sure that the batches of data our model receives during training are organized in a way that it can understand and process. The collate function helps handle situations where the individual pieces of data in our dataset, like images and labels, might have different shapes or structures.

        Imagine you have a collection of images and their corresponding labels, but the images are not all the same size. Without a collate function, the DataLoader might struggle to put these images into batches for your model because it expects all images in a batch to have the same dimensions.

        The collate function steps in and tells the DataLoader exactly how to group the images and labels together. It helps create consistent batches where images and labels are neatly organized, making it easier for your neural network to learn and make predictions. So, in simple terms, we use a collate function to ensure that our data is ready and organized for effective training.
    '''
    # Ensure that your dataset returns samples in a format that this collate function expects (i.e., as tuples), and that it aligns with your model's input requirements.
    @staticmethod
    def collate_fn(batch):
        """
        This is our collating function for the train dataloader, 
        it allows us to create batches of data that can be easily pass into the model
        """
        try:
            # What happens here is that we would have batches like this with image and corresponding labels
            '''
                dataset = [
                    (image1_tensor, label1),
                    (image2_tensor, label2),
                    (image3_tensor, label3),
                    # ...
                ]
            '''
            # Using this each batch will contain a tuple where the first element is a tensor containing all the images, and the second element is a tensor containing all the labels
            '''
                batch_after = (
                    torch.stack([image1_tensor, image2_tensor, image3_tensor, ...]),
                    torch.tensor([label1, label2, label3, ...])
                )
            '''
            # This line assumes that your batch is a list of tuples, and it transposes the batch.
            #  This is useful when you have a list of samples, where each sample is a tuple of input data and its corresponding target (e.g., an image and its label). By transposing the batch, it groups all input data together in the first element of the returned tuple and all targets in the second element.
            return tuple(zip(*batch))
        except Exception as e:
            raise HelmetException(e, sys) from e
        
    # Intiating the Model Training
    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            # Loading the transformed object we saved in data transformation by using the path in data transformation entity 
            train_dataset = load_object(self.data_transformation_artifacts.transformed_train_object)
            # We create a data loader object for our train and test respectively by specifying the transformed train dataset which is now in pytorch format, batch size and others
            train_loader = DataLoader(train_dataset,
                                     batch_size=self.model_trainer_config.BATCH_SIZE,
                                     shuffle=self.model_trainer_config.SHUFFLE,
                                     num_workers=self.model_trainer_config.NUM_WORKERS,
                                     collate_fn=self.collate_fn
                                     )
            # Also here we load the test dataset and create a data loader object
            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)
            test_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=self.model_trainer_config.SHUFFLE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS,
                                      collate_fn=self.collate_fn
                                      )
            logging.info("Loaded training data loader object")

            # Here we intantiate our model which is the pytorch models from the torchvsion library, 
            # we set pretrained = True so it will use the pretrained weights of the COCO dataset. This means that the model has already been trained on a large dataset and can be used for various detection tasks out of the box.
            model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
            logging.info("Loaded faster Rcnn  model")

            # This line is extracting the number of input features to the classification layer (cls_score) of the region of interest (ROI) heads in the Faster R-CNN model. Specifically, it retrieves the number of input features to the classification layer, which is crucial for later modifications or customization.
            # model.roi_heads: Accesses the region of interest (ROI) heads of the Faster R-CNN model. The ROI heads are responsible for region-based operations, including classification and bounding box regression.
            # .box_predictor: Accesses the box predictor within the ROI heads. The box predictor is responsible for predicting bounding box coordinates and class scores.
            # Accesses the classification layer of the box predictor. This layer outputs class scores for each region of interest.
            # .in_features: Retrieves the number of input features to the classification layer. This value represents the dimensionality of the feature vector input to the classification layer.
            in_features = model.roi_heads.box_predictor.cls_score.in_features  # we need to change the head

            # This line is replacing the existing box predictor in the region of interest (ROI) heads of the Faster R-CNN model with a new one. The new box predictor is an instance of FastRCNNPredictor from torchvision.models.detection.faster_rcnn.
            # model.roi_heads.box_predictor: Accesses the box predictor within the ROI heads of the Faster R-CNN model.
            # models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.data_transformation_artifacts.number_of_classes): Assigns a new instance of FastRCNNPredictor to replace the existing box predictor. The FastRCNNPredictor is initialized with the specified number of input features (in_features) and the number of output classes (self.data_transformation_artifacts.number_of_classes).
            # This operation is commonly used when you want to customize the final classification layer of the Faster R-CNN model, especially when adapting the model to a specific task or dataset. The FastRCNNPredictor class allows you to define a new fully connected layer for classification and bounding box regression.
            # Make sure that self.data_transformation_artifacts.number_of_classes is set to the correct number of classes for your specific task. 
            model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.data_transformation_artifacts.number_of_classes)

            # Initializing our model optimizer using our mode_optimiser function in the "ml" directory
            optimiser = model_optimiser(model)
            logging.info("loaded optimiser")

            # An epoch refers to one complete pass through the entire training dataset.
            # We update our parameters like weight based on the number of epoch
            for epoch in range(self.model_trainer_config.EPOCH):
                # calling the train method for each epoch and passing in necessary parameter
                self.train(model, optimiser, train_loader, self.model_trainer_config.DEVICE, epoch)

            # Making the direcory of where the model will be saved
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            # Saving the model in the directory specified in our config
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)

            logging.info(f"Saved the trained model")

            # Creating a ModelTrainerArtifacts which which contains our model path 
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            # Returning the Model Artifacts which is the path of the model we have just trained
            return model_trainer_artifacts

        except Exception as e:
            raise HelmetException(e, sys) from e