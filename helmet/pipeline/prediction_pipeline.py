import os
import io
import sys
from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from helmet.exception import HelmetException
from helmet.logger import logging
from helmet.constants import *

class PredictionPipeline:
    def __init__(self):
        pass

    def image_loader(self, image_bytes):
        """load image, returns cuda tensor"""
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            # image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = Image.open(io.BytesIO(image_bytes))
            convert_tensor = transforms.ToTensor()
            tensor_image = convert_tensor(image)
            # image = image[:3]
            image_int = torch.tensor(tensor_image * 255, dtype=torch.uint8)
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return tensor_image, image_int

        except Exception as e:
            raise HelmetException(e, sys) from e
        
    def get_model(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel", TRAINED_MODEL_NAME)
            logging.info("Exited the get_model method of PredictionPipeline class")
            return predict_model_path

        except Exception as e:
            raise HelmetException(e, sys) from e
        
    def prediction(self, best_model_path: str, image_tensor, image_int_tensor) -> float:
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            model = torch.load(best_model_path, map_location=torch.device(DEVICE))
            model.eval()
            with torch.no_grad():
                prediction = model([image_tensor.to(DEVICE)])
                pred = prediction[0]

            bbox_tensor = draw_bounding_boxes(image_int_tensor,
                                pred['boxes'][pred['scores'] > 0.8],
                                [PREDICTION_CLASSES[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()],
                                width=4).permute(0, 2, 1)

            transform = transforms.ToPILImage()
            img = transform(bbox_tensor)

            # Convert the saved image to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Add the required prefix
            img_str = f"data:image/jpeg;base64,{img_str}"

            # Save the image to the specified output path
            img.save('predicted_image.jpg')
            
            # img_str = base64.b64encode(buffered.getvalue())
            logging.info(f'This is the base64 format of the predicted image: {img_str}')

            logging.info("Exited the prediction method of PredictionPipeline class")
            return img_str

        except Exception as e:
            raise HelmetException(e, sys) from e
        
    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image, image_int = self.image_loader(data)
            print(image.shape)
            print(image_int.shape)
            best_model_path: str = os.path.join('C:/Users/user/My ML Projects/Helmet Prediction/artifacts/01_08_2024_11_18_07/TrainedModel/model.pt')
            detected_image = self.prediction(best_model_path, image, image_int)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image
        except Exception as e:
            raise HelmetException(e, sys) from e
