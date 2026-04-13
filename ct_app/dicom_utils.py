import datetime

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid


def create_dicom(image_data, patient_name, patient_id, comments):
    """Zapisuje zrekonstruowany obraz jako obiekt DICOM."""
    normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8)
    pixel_data = (normalized_image * 65535).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
    ds.ImageComments = comments

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = pixel_data.shape[1]
    ds.Rows = pixel_data.shape[0]

    ds.PixelData = pixel_data.tobytes()
    return ds
