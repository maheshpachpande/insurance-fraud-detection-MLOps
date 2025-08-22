
from insurance_src.logger import logging
import sys
from insurance_src.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline
# from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
# from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
# from src.pipeline.stage_04_model_trainer import ModelTrainingPipeline
# from src.pipeline.stage_05_model_evolution import ModelEvolutionTrainingPipeline
# from src.pipeline.stage_06_model_pusher import ModelPusherTrainingPipeline

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter errors




STAGE_NAME = "Data Ingestion stage"


try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e


# STAGE_NAME = "Data Validation stage"

# try:
#     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataValidationTrainingPipeline()
#     obj.main()
#     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     sys.exit(str(e))
    
    
# STAGE_NAME = "Data Transformation stage"

# try:
#     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataTransformationTrainingPipeline()
#     obj.main()
#     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     sys.exit(str(e))
    
    
# STAGE_NAME = "Model Trainer stage"

# try:
#     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = ModelTrainingPipeline()
#     obj.main()
#     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     sys.exit(str(e))


    
# STAGE_NAME = "Model Evolution stage"

# try:
#     model_evolution_pipeline = ModelEvolutionTrainingPipeline()
#     model_evolution_pipeline.main()
# except Exception as e:
#     logging.error(f"Error during model training and evaluation: {e}")
#     sys.exit(1)
            
# STAGE_NAME = "Model Pusher stage"
# try:
#     model_evolution_pipeline = ModelPusherTrainingPipeline()
#     model_evolution_pipeline.main()
# except Exception as e:
#     logging.error(f"Error during model training and evaluation: {e}")
#     sys.exit(1)