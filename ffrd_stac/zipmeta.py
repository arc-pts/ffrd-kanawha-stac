import s3fs
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Iterator, Optional, Tuple, Dict
from mypy_boto3_s3.service_resource import Object, ObjectSummary
import zipfile
import json
import re

def scan_s3_zip(fs: s3fs.S3FileSystem, s3_zip_file: str):
    contents = []
    ras_models = []
    with fs.open(s3_zip_file, "rb") as zip_file:
        with zipfile.ZipFile(BytesIO(zip_file.read())) as zip_ref:
            info = zip_ref.infolist()
            for i, file in enumerate(info):
                logging.info(f"scan_s3_zip | {i} {file.filename}")
                contents.append(file.filename)
                if Path(file.filename).suffix == ".prj":
                    file_bytes = zip_ref.read(file.filename)
                    logging.debug(
                        f"scan_s3_zip | prj data for {file.filename}: {file_bytes.decode()}"
                    )
                    if "Proj Title" in file_bytes.decode():
                        ras_models.append(file.filename)

    return contents, ras_models

def filter_zip_archive_objects(pattern: str = None, zip_info_list: list = None) -> List[str]:
    compiled_pattern = re.compile(pattern) if pattern else None
    objects = []
    for obj in zip_info_list:
        if compiled_pattern:
            if re.search(compiled_pattern, obj):
                objects.append(obj)
        else:
            objects.append(obj)
    return objects

def sort_ras_model_files(ras_prj_file:str, objects: list) -> dict:
    ras_model_objects = {"ras_prj_file": ras_prj_file}
    other_files = []
    for obj in objects:
        match = re.search(r'\.([a-zA-Z]+)\d+$', obj)
        if match:
            group = match.group(1)
            if group in ras_model_objects:
                ras_model_objects[group].append(obj)
            else:
                ras_model_objects[group] = [obj]
        else:
            other_files.append(obj)

    ras_model_objects['geom_hdfs'] = filter_zip_archive_objects(r'g\.?\d+\.hdf$', other_files)
    ras_model_objects['plan_hdfs'] = filter_zip_archive_objects(r'p\.?\d+\.hdf$', other_files)
    ras_model_objects['tmp_hdfs'] = filter_zip_archive_objects(r'tmp\.hdf$', other_files)

    hdf_files = ras_model_objects['geom_hdfs']  + ras_model_objects['plan_hdfs'] + ras_model_objects['tmp_hdfs']
    ras_model_objects['other'] = [item for item in other_files if item not in hdf_files]
    
    ras_model_objects['other'].remove(ras_prj_file)

    return ras_model_objects

class ZipReaderError(Exception):
    def __init__(self, message="Error extracting data from zip"):
        self.message = message
        super().__init__(self.message)

class S3Zip:
    def __init__(self, bucket: str, key: str, fs: s3fs.S3FileSystem):
        self.bucket = bucket
        self.key = key
        self.fs = fs

        try:
            contents, ras_model_prjs = scan_s3_zip(self.fs, f"{self.bucket}/{self.key}")
            self._contents = contents
            self._ras_model_prjs = ras_model_prjs
        except Exception as e:
            raise ZipReaderError(
                f"Cannot read or list contents of {self.bucket}/{self.key}: {e}"
            )

    @property
    def contents(self):
        return self._contents

    @property
    def ras_models(self):
        ras_models = []
        for ras_model in self._ras_model_prjs:
            model_files = [f for f in self.contents if ras_model.strip(".prj") in f]
            ras_model_files = sort_ras_model_files(ras_model, model_files)
            ras_models.append(ras_model_files)
        return ras_models

    @property
    def contains_ras_models(self):
        if len(self.ras_models) >= 1:
            return True
        return False

    @property
    def shapefiles(self):
        return [f for f in self.contents if Path(f).suffix in [".shp"]]

    @property
    def contains_shapefiles(self):
        if len(self.shapefiles) >= 1:
            return True
        return False

    @property
    def rasters(self):
        return [f for f in self.contents if Path(f).suffix in [".tif"]]

    @property
    def contains_rasters(self):
        if len(self.rasters) >= 1:
            return True
        return False

    def shapefile_parts(self, filename: str):
        """
        Return list of auxilary files (assumed) to be parts of a shapefile
        """
        if Path(filename).suffix != ".shp":
            raise ValueError(
                f"filename ext must be `.shp` not {Path(filename).suffix}"
            )
        return [
            f
            for f in self.contents
            if f[:-4] == filename[:-4] and Path(f).suffix != ".shp"
        ]

    def __repr__(self):
        return json.dumps(
            {
                "S3ZIP": {
                    "bucket": self.bucket,
                    "key": self.key,
                    "shapefiles": len(self.shapefiles),
                    "rasters": len(self.rasters),
                    "ras_models": len(self.ras_models),
                }
            }
        )