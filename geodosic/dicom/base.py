class DicomBase(object):
    """docstring for DicomBase"""

    def __init__(self, ds):
        self.ds = ds

    def GetPatientInfo(self):
        """Return the patient metadata."""

        # Set up some sensible defaults for demographics
        patient = {'name': 'N/A',
                   'id': 'N/A',
                   'birth_date': None,
                   'gender': 'N/A'}
        if 'PatientName' in self.ds:
            if PY2:
                self.ds.decode()
            name = self.ds.PatientName
            patient['name'] = name
            patient['given_name'] = name.given_name
            patient['middle_name'] = name.middle_name
            patient['family_name'] = name.family_name
        if 'PatientID' in self.ds:
            patient['id'] = self.ds.PatientID
        if 'PatientSex' in self.ds:
            if (self.ds.PatientSex == 'M'):
                patient['gender'] = 'M'
            elif (self.ds.PatientSex == 'F'):
                patient['gender'] = 'F'
            else:
                patient['gender'] = 'O'
        if 'PatientBirthDate' in self.ds:
            if len(self.ds.PatientBirthDate):
                patient['birth_date'] = str(self.ds.PatientBirthDate)

        return patient

    def GetStudyInfo(self):
        """Return the study information of the current file."""

        study = {}
        if 'StudyDescription' in self.ds:
            desc = self.ds.StudyDescription
        else:
            desc = 'No description'
        study['description'] = desc
        if 'StudyDate' in self.ds:
            date = self.ds.StudyDate
        else:
            date = None
        study['date'] = date
        # Don't assume that every dataset includes a study UID
        if 'StudyInstanceUID' in self.ds:
            study['id'] = self.ds.StudyInstanceUID
        else:
            study['id'] = str(random.randint(0, 65535))

        return study

    def GetSeriesInfo(self):
        """Return the series information of the current file."""

        series = {}
        if 'SeriesDescription' in self.ds:
            desc = self.ds.SeriesDescription
        else:
            desc = 'No description'
        series['description'] = desc
        series['id'] = self.ds.SeriesInstanceUID
        # Don't assume that every dataset includes a study UID
        series['study'] = self.ds.SeriesInstanceUID
        if 'StudyInstanceUID' in self.ds:
            series['study'] = self.ds.StudyInstanceUID
        series['referenceframe'] = self.ds.FrameOfReferenceUID \
            if 'FrameOfReferenceUID' in self.ds \
            else str(random.randint(0, 65535))
        if 'Modality' in self.ds:
            series['modality'] = self.ds.Modality
        else:
            series['modality'] = 'OT'

        return series
