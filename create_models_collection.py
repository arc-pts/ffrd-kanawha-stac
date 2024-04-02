from ffrd_stac.utils import list_ras_model_names, filter_objects

import boto3
from dotenv import load_dotenv
import pystac
import requests

import argparse
import json
import logging
from pathlib import Path
from typing import List
from urllib.parse import urljoin


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
file_handler = logging.FileHandler('kanawha.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)

load_dotenv()


BUCKET_NAME = "kanawha-pilot"
BUCKET = boto3.resource('s3').Bucket(BUCKET_NAME)


def create_models_collection(models: List[pystac.Item]) -> pystac.Collection:
    extent = pystac.Extent.from_items(models)
    return pystac.Collection(
        id="kanawha-models-march-2024",
        description="Kanawha HEC-RAS Models (March 2024)",
        title="Kanawha HEC-RAS Models (March 2024)", 
        extent=extent,
    )


def read_geometry_items() -> List[pystac.Item]:
    objects = filter_objects(BUCKET, pattern=r".*/geometry\.json$", prefix="stac/ressim/")
    items = []
    for obj in objects:
        content = obj.get()["Body"].read().decode("utf-8")
        item = pystac.Item.from_dict(json.loads(content))
        items.append(item)
    return items


def create_simulations_collection(extent: pystac.Extent) -> pystac.Collection:
    return pystac.Collection(
        id="kanawha-simulations-march-2024",
        description="Kanawha HEC-RAS Simulations (March 2024)",
        title="Kanawha HEC-RAS Simulations (March 2024)",
        extent=extent,
    )


def create_collection(endpoint: str, collection: pystac.Collection):
    collections_url = urljoin(endpoint, "collections")
    response = requests.post(collections_url, json=collection.to_dict())
    if response.status_code == 409:
        logger.info(f"Collection {collection.id} already exists")
        collection_update_url = urljoin(collections_url, collection.id)
        logger.error("Retrying with PUT...")
        response = requests.put(collection_update_url, json=collection.to_dict())


def create_item(endpoint: str, collection_id: str, item: pystac.Item):
    items_url = urljoin(endpoint, f"collections/{collection_id}/items")
    response = requests.post(items_url, json=item.to_dict())
    if response.status_code == 409:
        logger.info(f"Item {item.id} already exists")
        item_update_url = urljoin(items_url, item.id)
        logger.error("Retrying with PUT...")
        response = requests.put(item_update_url, json=item.to_dict())
    if not response.ok:
        logger.error(f"Response from STAC API: {response.status_code}")
        logger.error(response.text)


def load_simulation_items(endpoint: str, collection_id: str):
    objects = filter_objects(BUCKET, pattern=r".*/sims/\w*-\d+\.json$", prefix="stac/ressim/")
    for obj in objects:
        content = obj.get()["Body"].read().decode("utf-8")
        model_name, sim_number = Path(obj.key).name.split(".")[0].split("-")
        new_id = f"{model_name}-{int(sim_number):04d}"
        item = pystac.Item.from_dict(json.loads(content))
        item.id = new_id
        logger.info(item)
        create_item(endpoint, collection_id, item)


def main(endpoint: str):
    model_items = read_geometry_items()
    models_collection = create_models_collection(model_items)
    print(models_collection)
    create_collection(endpoint, models_collection)

    for model_item in model_items:
        create_item(endpoint, models_collection.id, model_item)

    simulations_collection = create_simulations_collection(models_collection.extent)
    print(simulations_collection)
    create_collection(endpoint, simulations_collection)

    load_simulation_items(endpoint, simulations_collection.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint', help='STAC API endpoint')
    args = parser.parse_args()
    main(args.endpoint)
