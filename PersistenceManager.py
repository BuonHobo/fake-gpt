import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import json

class PersistenceManager:
    def __init__(self):
        uri = os.getenv('MONGO_URI')
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client['fake-gpt']

    def save_step(self, step):
        print(json.dumps(step,indent="\t"),flush=True)
        self.db['training_batches'].insert_one(step)

    def save_instance(self, instance):
        print(json.dumps(instance,indent="\t"),flush=True)