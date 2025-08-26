import sys
from pandas import DataFrame
from typing import Any
from insurance_src.entity.config_entity import PredictorConfig
from insurance_src.entity.s3_estimator import S3_InsuranceEstimator
from insurance_src.exceptions import CustomException
from insurance_src.logger import logging
from insurance_src.utils.main_utils import read_yaml_file


class InsuranceData:
    def __init__(self,
                 months_as_customer: int,
                 age: int,
                 policy_state: object,
                 policy_csl: object,
                 policy_deductable: int,
                 policy_annual_premium: float,
                 umbrella_limit: int,
                 insured_sex: object,
                 insured_education_level: object,
                 insured_occupation: object,
                 insured_hobbies: object,
                 insured_relationship: object,
                 capital_gains: int,
                 capital_loss: int,
                 incident_type: object,
                 collision_type: object,
                 incident_severity: object,
                 authorities_contacted: object,
                 incident_state: object,
                 incident_hour_of_the_day: int,
                 number_of_vehicles_involved: int,
                 property_damage: object,
                 bodily_injuries: int,
                 witnesses: int,
                 police_report_available: object,
                 total_claim_amount: int,
                 injury_claim: int,
                 property_claim: int,
                 vehicle_claim: int,
                 auto_make: object,
                 auto_year: int
                 ):
        """
        Insurance Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.months_as_customer = months_as_customer
            self.age = age
            self.policy_state = policy_state
            self.policy_csl = policy_csl
            self.policy_deductable = policy_deductable
            self.policy_annual_premium = policy_annual_premium
            self.umbrella_limit = umbrella_limit
            self.insured_sex = insured_sex
            self.insured_education_level = insured_education_level
            self.insured_occupation = insured_occupation
            self.insured_hobbies = insured_hobbies
            self.insured_relationship = insured_relationship
            self.capital_gains = capital_gains
            self.capital_loss = capital_loss
            self.incident_type = incident_type
            self.collision_type = collision_type
            self.incident_severity = incident_severity
            self.authorities_contacted = authorities_contacted
            self.incident_state = incident_state
            self.incident_hour_of_the_day = incident_hour_of_the_day
            self.number_of_vehicles_involved = number_of_vehicles_involved
            self.property_damage = property_damage
            self.bodily_injuries = bodily_injuries
            self.witnesses = witnesses
            self.police_report_available = police_report_available
            self.total_claim_amount = total_claim_amount
            self.injury_claim = injury_claim
            self.property_claim = property_claim
            self.vehicle_claim = vehicle_claim
            self.auto_make = auto_make
            self.auto_year = auto_year

        except Exception as e:
            raise CustomException(e)

    def get_insurance_input_data_frame(self) -> DataFrame:
        """
        Returns a DataFrame from InsuranceData class input
        """
        try:
            insurance_input_dict = self.get_insurance_data_as_dict()
            return DataFrame(insurance_input_dict)
        except Exception as e:
            raise CustomException(e)

    def get_insurance_data_as_dict(self):
        """
        Returns a dictionary from InsuranceData class input
        """
        logging.info("Entered get_insurance_data_as_dict method of InsuranceData class")
        try:
            input_data = {
                "months_as_customer": [self.months_as_customer],
                "age": [self.age],
                "policy_state": [self.policy_state],
                "policy_csl": [self.policy_csl],
                "policy_deductable": [self.policy_deductable],
                "policy_annual_premium": [self.policy_annual_premium],
                "umbrella_limit": [self.umbrella_limit],
                "insured_sex": [self.insured_sex],
                "insured_education_level": [self.insured_education_level],
                "insured_occupation": [self.insured_occupation],
                "insured_hobbies": [self.insured_hobbies],
                "insured_relationship": [self.insured_relationship],
                "capital-gains": [self.capital_gains],
                "capital-loss": [self.capital_loss],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "incident_state": [self.incident_state],
                "incident_hour_of_the_day": [self.incident_hour_of_the_day],
                "number_of_vehicles_involved": [self.number_of_vehicles_involved],
                "property_damage": [self.property_damage],
                "bodily_injuries": [self.bodily_injuries],
                "witnesses": [self.witnesses],
                "police_report_available": [self.police_report_available],
                "total_claim_amount": [self.total_claim_amount],
                "injury_claim": [self.injury_claim],
                "property_claim": [self.property_claim],
                "vehicle_claim": [self.vehicle_claim],
                "auto_make": [self.auto_make],
                "auto_year": [self.auto_year],
            }
            logging.info("Created insurance data dict")
            logging.info("Exited get_insurance_data_as_dict method of InsuranceData class")
            return input_data
        except Exception as e:
            raise CustomException(e)


class InsuranceClassifier:
    def __init__(self, prediction_pipeline_config: PredictorConfig = PredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise CustomException(e)

    def predict(self, dataframe) -> Any:
        """
        Predict insurance fraud
        """
        try:
            logging.info("Entered predict method of InsuranceClassifier class")
            model = S3_InsuranceEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise CustomException(e)