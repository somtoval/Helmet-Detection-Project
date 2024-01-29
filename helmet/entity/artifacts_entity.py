# This artifacts entity is what we expect from each of the component at the end of their individaul stage
from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str 
    test_file_path: str
    valid_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str 
    transformed_test_object: str
    number_of_classes: int

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

@dataclass
class ModelEvaluationArtifacts:
    # is_model_accepted: bool
    all_losses: str