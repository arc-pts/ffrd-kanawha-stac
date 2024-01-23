from ffrd_stac.utils import filter_objects, list_ras_model_names
from ffrd_stac.rasmeta import RasGeomHdf, RasPlanHdf

import boto3
from dotenv import load_dotenv
import fsspec
from mypy_boto3_s3.service_resource import Bucket, Object

import argparse
import json
import logging
import os
from pathlib import Path
import shapely
from typing import List


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
file_handler = logging.FileHandler('meta.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)

load_dotenv()


BUCKET_NAME = "kanawha-pilot"
RAS_MODELS_PREFIX = "FFRD_Kanawha_Compute/ras"
OUTPUT_PATH = "FFRD_Kanawha_Compute/runs/{simulation}/ras/{ras_model}/{ras_model}.p01.hdf"
SIMULATIONS = 1001


def get_simulation_metadata(bucket: Bucket, ras_model: str, simulation: int) -> dict:
    plan_hdf_key = OUTPUT_PATH.format(simulation=simulation, ras_model=ras_model)
    s3url = f"s3://{bucket.name}/{plan_hdf_key}"
    s3f = fsspec.open(s3url, mode="rb")
    metadata = {
        "cloud_wat:realization": 1,
        "cloud_wat:simulation": simulation,
    }
    try:
        plan_hdf = RasPlanHdf(s3f.open(), mode="r")
        plan_attrs = plan_hdf.get_plan_attrs()
        results_attrs = plan_hdf.get_plan_results_attrs()
        metadata.update(plan_attrs)
        metadata.update(results_attrs)
    except FileNotFoundError as e:
        logger.error(f"File not found: {plan_hdf_key}")
    return metadata


def get_ras_model_metadata(bucket: Bucket, ras_model: str) -> List[dict]:
    results = []
    for simulation in range(1, SIMULATIONS + 1):
        logger.info(f"Getting RAS simulation output metadata for {ras_model} sim {simulation} of {SIMULATIONS}")
        metadata = get_simulation_metadata(bucket, ras_model, simulation)
        results.append(metadata)
    return results


def get_ras_model_geom(bucket: Bucket, ras_model: str) -> dict:
    logger.info(f"Getting RAS model geometry: {ras_model}")
    plan_hdf_key = OUTPUT_PATH.format(simulation=1, ras_model=ras_model)
    ras_plan_hdf = RasGeomHdf.open_url(f"s3://{bucket.name}/{plan_hdf_key}")
    polygon = ras_plan_hdf.get_2d_flow_area_perimeter()
    return json.loads(shapely.to_geojson(polygon))


def main(bucket_name: str, geometry_only: bool = False):
    meta_path = Path('./meta')
    if not meta_path.exists():
        meta_path.mkdir()
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    ras_model_names = list_ras_model_names(bucket, RAS_MODELS_PREFIX)
    logger.info(ras_model_names)
    geoms = {}
    for ras_model in ras_model_names:
        logger.info(f"Processing {ras_model}")
        if geometry_only:
            geom = get_ras_model_geom(bucket, ras_model)
            geoms[ras_model] = geom
            with open(meta_path / "geoms.json", "w") as f:
                json.dump(geoms, f)
        else:
            metadata = get_ras_model_metadata(bucket, ras_model)
            ndjson = "\n".join([json.dumps(m) for m in metadata])
            model_meta_path = meta_path / f"{ras_model}.ndjson"
            logger.info(f"Writing metadata to {model_meta_path}")
            with open(model_meta_path, "w") as f:
                f.write(ndjson)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", help="S3 bucket name", default=BUCKET_NAME)
    parser.add_argument("--geometry-only", help="Only get 2D flow area perimeter data", action="store_true")
    args = parser.parse_args()
    main(args.bucket, args.geometry_only)
