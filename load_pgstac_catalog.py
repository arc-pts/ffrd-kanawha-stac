from dotenv import load_dotenv
import pystac
import requests

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
file_handler = logging.FileHandler('load-pgstac.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)


load_dotenv()


def fix_derived_from_links(item: dict) -> dict:
    links: Optional[List[dict]] = item.get('links')
    if links is None:
        return item
    fixed_item = deepcopy(item)
    for link in links:
        if link.get('rel') == 'derived_from':
            href_split = link.get('href').split('/')
            collection = href_split[2]
            ref_id = href_split[3]
            link["href"] = f"../../collections/{collection}/items/{ref_id}"
    fixed_item['links'] = links
    return fixed_item


def load_collection_items(endpoint: str, collection_id: str, items_dir: Path) -> None:
    for path in items_dir.iterdir():
        if path.is_dir():
            logger.info(f"Loading STAC item from {path}")
            item_json = (path / path.name).with_suffix('.json')
            with open(item_json, 'r') as f:
                item = json.load(f)
                item = fix_derived_from_links(item)
            items_url = urljoin(endpoint, f"collections/{collection_id}/items")
            logger.info(f"Posting item to {items_url}")
            response = requests.post(items_url, json=item)
            if response.status_code == 409:
                logger.info(f"Item {item['id']} already exists")
                item_update_url = items_url + f"/{item['id']}"
                logger.info(f"Updating item at {item_update_url}")
                response = requests.put(urljoin(items_url, item["id"]), json=item)
            if not response.ok:
                logger.error(f"Response from pgstac: {response.status_code}")
                logger.error(response.text)


def load_collection(endpoint: str, collection_dir: Path) -> None:
    collection_path = collection_dir / 'collection.json'
    with open(collection_path, 'r') as f:
        collection_json = json.load(f)
    collections_url = urljoin(endpoint, "collections")
    response = requests.post(collections_url, json=collection_json)
    logger.info(f"Response from pgstac: {response.status_code}")
    logger.info(response.text)
    collection = pystac.Collection.from_dict(collection_json)
    load_collection_items(endpoint, collection.id, collection_dir)


def main(endpoint: str, directory: str):
    t1 = datetime.now()
    logging.info("Loading STAC data into Postgres")
    pgstac_dir = Path(directory)
    for path in pgstac_dir.iterdir():
        if path.is_dir():
            logger.info(f"Loading STAC collection from {path}")
            load_collection(endpoint, path)
    t2 = datetime.now()
    logger.info(f"Finished loading STAC data in {(t2 - t1).total_seconds()/60:.2f} mins")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint', help='STAC API endpoint')
    parser.add_argument('--directory', help='directory to upload', default='./pgstac/')
    args = parser.parse_args()
    main(args.endpoint, args.directory)
