
from insurance_src.logger import logging
from insurance_src.exceptions import CustomException
from insurance_src.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from insurance_src.pipeline.stage_02_data_validation_pipeline import DataValidationPipeline
from insurance_src.pipeline.stage_03_data_transformation_pipeline import DataTransformationPipeline
# from insurance_src.pipeline.stage_04_model_trainer import ModelTrainingPipeline
# from insurance_src.pipeline.stage_05_model_evolution import ModelEvolutionTrainingPipeline
# from insurance_src.pipeline.stage_06_model_pusher import ModelPusherTrainingPipeline

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter errors




STAGE_NAME = "Data Ingestion stage"


try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.run()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e)


STAGE_NAME = "Data Validation stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationPipeline()
    obj.run()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e)
    
    
STAGE_NAME = "Data Transformation stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationPipeline()
    obj.run()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e)
    
    
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