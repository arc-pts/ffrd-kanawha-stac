from ffrd_stac.utils import filter_objects, list_ras_model_names, get_dict_values
from ffrd_stac.rasmeta import RasGeomHdf, RasPlanHdf, parse_duration

import boto3
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import pystac
import shapely

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
from typing import List, Optional
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
file_handler = logging.FileHandler('pgstac.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)

load_dotenv()

BUCKET_NAME = "kanawha-pilot"
BUCKET = boto3.resource('s3').Bucket(BUCKET_NAME)

CATALOG_TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M')
ROOT_HREF = f"./pgstac/pgstac-kanawha-models-{CATALOG_TIMESTAMP}"

REALIZATION = 1


def create_catalog() -> pystac.Catalog:
    return pystac.Catalog(
        id="kanawha-pilot-ras",
        description="pgstac catalog for the Kanawha produced under an FFRD pilot project",
        title="Kanawha HEC-RAS Models (pgstac)"
    )


def create_model_item(ras_model_name: str) -> pystac.Item:
    logger.info(f"Creating STAC item for model {ras_model_name}")
    ras_geom_hdf_url = f"s3://{BUCKET_NAME}/FFRD_Kanawha_Compute/sims/ressim/1/ras/{ras_model_name}/{ras_model_name}.p01.hdf"
    ras_hdf = RasGeomHdf.open_url(ras_geom_hdf_url)
    perimeter = ras_hdf.get_2d_flow_area_perimeter(simplify=100.0)  # simplify with tolerance in model units
    properties = ras_hdf.get_geom_attrs()
    geometry_time = properties.get("geometry:geometry_time")
    model_id = ras_model_name
    # model_id = str(uuid.uuid1())
    item = pystac.Item(
        id=model_id,
        geometry=json.loads(shapely.to_geojson(perimeter)),
        bbox=perimeter.bounds,
        datetime=datetime.fromisoformat(geometry_time),
        properties=properties,
    )
    return item


def create_models_collection() -> pystac.Collection:
    logger.info("Creating STAC collection for models")
    ras_model_names = list_ras_model_names(BUCKET, "FFRD_Kanawha_Compute/ras")
    items: List[pystac.Item] = []
    for ras_model_name in ras_model_names:
        item = create_model_item(ras_model_name)
        items.append(item)
    extent = pystac.Extent.from_items(items)
    collection = pystac.Collection(
        id="kanawha-pilot-ras-models",
        description="Kanawha HEC-RAS models",
        extent=extent,
    )
    for ras_model_name, item in zip(ras_model_names, items):
        collection.add_item(item, title=ras_model_name)
    return collection


def get_realization_stats(props: List[dict]) -> dict:
    logger.info("Getting realization stats...")
    computation_times = get_dict_values(props, "results_summary:computation_time_total")
    computation_time_total_minutes = [parse_duration(t) for t in computation_times]
    run_time_windows = get_dict_values(props, "results_summary:run_time_window")
    run_time_starts = [i[0] for i in run_time_windows]
    run_time_stops = [i[1] for i in run_time_windows]
    error_percents = get_dict_values(props, 'volume_accounting:error_percent')
    solutions = get_dict_values(props, 'results_summary:solution')
    stats = {
        "cloud_wat:simulations": len(props),
        "cloud_wat:min_computation_time_mins": min(computation_time_total_minutes).total_seconds() / 60,
        "cloud_wat:max_computation_time_mins": max(computation_time_total_minutes).total_seconds() / 60,
        "cloud_wat:avg_computation_time_mins": np.mean(computation_time_total_minutes).total_seconds() / 60,
        "cloud_wat:total_computation_time_hrs": np.sum(computation_time_total_minutes).total_seconds() / 3600,
        "cloud_wat:run_time_window": [min(run_time_starts), max(run_time_stops)],
        "cloud_wat:min_volume_error_percent": min(error_percents),
        "cloud_wat:max_volume_error_percent": max(error_percents),
        "cloud_wat:avg_volume_error_percent": np.mean(error_percents),
        "cloud_wat:unsuccessful_runs": len([s for s in solutions if s != "Unsteady Finished Successfully"]),
    }
    logger.info(stats)
    return stats


def create_model_realization_item(ras_model_name: str, realization: int, model_item: pystac.Item, simulation_items: List[pystac.Item]) -> pystac.Item:
    stats = get_realization_stats([item.properties for item in simulation_items])
    start_datetime = datetime.fromisoformat(stats["cloud_wat:run_time_window"][0])
    end_datetime = datetime.fromisoformat(stats["cloud_wat:run_time_window"][1])
    item = pystac.Item(
        id=f"{ras_model_name}-r{realization:04d}",
        properties=stats,
        bbox=model_item.bbox,
        geometry=model_item.geometry,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        datetime=start_datetime
    )
    return item


def create_model_simulation_item(ras_model_name: str, model_item: pystac.Item, results_meta: dict) -> Optional[pystac.Item]:
    realization = results_meta["cloud_wat:realization"]
    simulation = results_meta["cloud_wat:simulation"]
    logger.debug(f"Creating STAC item for model simulation {realization}:{simulation} - {ras_model_name}")
    if len(results_meta.keys()) == 2:
        logger.warning(f"No results for {realization}:{simulation} - {ras_model_name}")
        return None
    model_sim_id = f"{ras_model_name}-r{realization:04d}-s{simulation:04d}"
    runtime_window = results_meta.get("results_summary:run_time_window")
    start_datetime = datetime.fromisoformat(runtime_window[0])
    end_datetime = datetime.fromisoformat(runtime_window[1])
    item = pystac.Item(
        id=model_sim_id,
        geometry=model_item.geometry,
        bbox=model_item.bbox,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        datetime=start_datetime,
        properties=results_meta,
    )
    return item


def create_model_simulation_items(ras_model_name: str, model_item: pystac.Item, limit: Optional[int] = None) -> List[pystac.Item]:
    logger.info(f"Creating STAC items for model simulations of {ras_model_name}")
    items = []
    with open(f"./meta/{ras_model_name}.ndjson") as f:
        metadata = [json.loads(line) for line in f.readlines()]
        if limit:
            metadata = metadata[:limit + 1]
        for meta in metadata:
            item = create_model_simulation_item(ras_model_name, model_item, meta)
            if item is not None:
                # item_json = json.dumps(item.to_dict())
                items.append(item)
    return items


def main(limit: Optional[int] = None):
    stac_path = Path('./pgstac')
    if stac_path.exists():
        shutil.rmtree(stac_path)
    stac_path.mkdir(exist_ok=True) 

    catalog = create_catalog()
    ras_models_collection = create_models_collection()
    catalog.add_child(ras_models_collection)

    model_realizations = []
    model_simulations = []

    for ras_model in ras_models_collection.get_items():
        model_simulation_items = create_model_simulation_items(ras_model.id, ras_model, limit=limit)

        realization_item = create_model_realization_item(ras_model.id, REALIZATION, ras_model, model_simulation_items)
        realization_item.add_derived_from(ras_model)

        for model_sim in model_simulation_items:
            model_sim.add_derived_from(ras_model)
            model_sim.add_derived_from(realization_item)

        model_simulations.extend(model_simulation_items)
        model_realizations.append(realization_item)

    model_realizations_collection = pystac.Collection(
        id="kanawha-pilot-ras-model-realizations",
        description="Kanawha HEC-RAS model realizations",
        extent=pystac.Extent.from_items(model_realizations),
    )
    model_simulations_collection = pystac.Collection(
        id="kanawha-pilot-ras-model-simulations",
        description="Kanawha HEC-RAS model simulations",
        extent=pystac.Extent.from_items(model_simulations),
    )

    model_realizations_collection.add_items(model_realizations)
    model_simulations_collection.add_items(model_simulations)

    catalog.add_child(model_realizations_collection)
    catalog.add_child(model_simulations_collection)

    catalog.normalize_hrefs(root_href=ROOT_HREF)

    catalog.normalize_and_save(root_href=ROOT_HREF, catalog_type=pystac.CatalogType.SELF_CONTAINED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build pgstac catalog')
    parser.add_argument('--limit', type=int, help='Limit the number of simulations to process')
    args = parser.parse_args()
    main(args.limit)
