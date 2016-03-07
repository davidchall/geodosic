# system imports
import json

# project imports
from .dicom import DicomCollection


class Patient(object):
    """docstring for Patient"""
    def __init__(self, config_path):
        super(Patient, self).__init__()
        with open(config_path) as f:
            self.config = json.load(f)

        self.dicom = DicomCollection(self.config['dicom_dir'])
