import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from StepRepresentation import StepRepresentation

class PersistenceManager:
    def __init__(self):
        uri = os.getenv('MONGO_URI')
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client['fake-gpt']

    def save_step(self, step:StepRepresentation):
        self.db['training_steps'].insert_one(step.to_dict())