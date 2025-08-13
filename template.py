import os
from pathlib import Path

project_name = "src"

list_of_files = [
    f"{project_name}/__init__.py",
    
    # Components
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",  
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    
    # Configuration
    f"{project_name}/config/__init__.py",
    
    # Cloud Storage
    f"{project_name}/cloud_storage/__init__.py",
    
    # Data Access
    f"{project_name}/data_access/__init__.py",
    
    # Constants
    f"{project_name}/constants/__init__.py",
    
    # Entities
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    
    # Exceptions & Logging
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/logger/__init__.py",
    
    # Pipelines
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    
    # Utilities
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/file_utils.py",
    
    # Root level files
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "pyproject.toml",
    
    # Config files
    "config/model.yaml",
    "config/schema.yaml",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"File already present at: {filepath}")
