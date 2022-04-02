import os
import json
import SimpleITK as sitk

from radiomics import featureextractor, getTestCase

image_name, mask_name = getTestCase("brain1")

settings = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "verbose": True
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
feature = extractor.execute(image_name, mask_name)

for k, v in feature.items():
    print(k, v)
