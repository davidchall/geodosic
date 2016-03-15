from geodosic.dicom.base import DicomBase


class DicomRtPlan(DicomBase):
    """docstring for DicomRtPlan"""
    def __init__(self, ds):
        super(DicomRtPlan, self).__init__(ds)
