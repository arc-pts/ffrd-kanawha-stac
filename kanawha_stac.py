import boto3
from dotenv import load_dotenv
import re
import pystac
from pathlib import Path
import shutil
import json
import os
from datetime import datetime
from typing import List, Iterator
import shapely
from shapely.geometry import shape
import rasterio
from rasterio.session import AWSSession
import rasterio.warp
import fsspec
import h5py
import sys


load_dotenv()

s3 = boto3.resource('s3')
BUCKET_NAME = 'kanawha-pilot'
BUCKET = s3.Bucket(BUCKET_NAME)

ROOT_HREF = "./stac/kanawha-models"

MODELS_CATALOG_ID = "kanawha-models"
RAS_MODELS_COLLECTION_ID = f"{MODELS_CATALOG_ID}-ras"

SIMULATIONS = 4 
DEPTH_GRIDS = 5

AWS_SESSION = AWSSession(boto3.Session())


def create_catalog():
    catalog = pystac.Catalog(
        id=MODELS_CATALOG_ID,
        description="Models for the Kanawha produced under an FFRD pilot project",
        title="Kanawha Models"
    )
    return catalog


def get_fake_extent() -> pystac.Extent:
    spatial_extent = pystac.SpatialExtent([[0.0, 0.0, 1.0, 1.0]])
    temporal_extent = pystac.TemporalExtent(intervals=[datetime.now(), datetime.now()])
    fake_extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)
    return fake_extent


def get_fake_geometry():
    fake_geometry = shapely.Polygon([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0]
    ])
    return fake_geometry


def bbox_to_polygon(bbox) -> shapely.Polygon:
    min_x, min_y, max_x, max_y = bbox
    return shapely.Polygon([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ])


def get_realization_string(r: int) -> str:
    realization = f"r{str(r).zfill(4)}"
    return realization


def get_simulation_string(r: int) -> str:
    simulation = f"s{str(r).zfill(4)}"
    return simulation 


def create_ras_models_parent_collection():
    collection = pystac.Collection(
        id=RAS_MODELS_COLLECTION_ID,
        title="HEC-RAS Models",
        description="HEC-RAS Models for the Kanawha",
        extent=get_fake_extent(),
    )
    return collection


def create_ras_model_collection(key_base: str):
    model_objs = BUCKET.objects.filter(Prefix=key_base)
    basename = os.path.basename(key_base)
    collection = pystac.Collection(
        id=f"{RAS_MODELS_COLLECTION_ID}-{basename}",
        title=f"{basename}",
        description=f"HEC-RAS Model: {basename}",
        extent=get_fake_extent(),
    )
    for obj in model_objs:
        filename = os.path.basename(obj.key)
        asset = pystac.Asset(
            href=obj.key,
            title=filename,
        )
        collection.add_asset(key=filename, asset=asset)
    return collection


def create_ras_model_realization_collection(key_base: str, r: int):
    basename = os.path.basename(key_base)
    realization = f"r{str(r).zfill(4)}"
    collection = pystac.Collection(
        id=f"{RAS_MODELS_COLLECTION_ID}-{basename}-{realization}",
        title=f"{basename}-{realization}",
        description=f"Realization {realization} of HEC-RAS model {basename}",
        extent=get_fake_extent(),
    )
    return collection


def get_ras_output_assets(key_base: str, r: int, s: int) -> List[pystac.Asset]:
    print('filtering objects')
    basename = os.path.basename(key_base)
    ras_output_objs = filter_objects(
        pattern=rf"^FFRD_Kanawha_Compute\/runs\/{s}\/ras\/{basename}\/.*$",
        prefix=f"FFRD_Kanawha_Compute/runs/{s}/ras/{basename}"
    )
    assets = []
    for obj in ras_output_objs:
        print(obj.key)
        filename = os.path.basename(obj.key)
        s = int(obj.key.split('/')[-4])
        simultation = get_simulation_string(s)
        realization = get_realization_string(r)
        simulation_filename = f"{realization}-{simultation}-{filename}"
        asset = pystac.Asset(
            href=obj.key, # TODO: s3 url
            title=simulation_filename,
        )
        if obj.key.endswith('.p01.hdf'):
            unsteady_summary = get_results_attrs(obj.key)
            asset.extra_fields = unsteady_summary
        assets.append(asset)
    return assets


def create_realization_ras_results_item(key_base: str, r: int):
    basename = os.path.basename(key_base)
    realization = f"r{str(r).zfill(4)}"
    fake_bbox = get_fake_extent().spatial.bboxes[0]
    fake_geometry = get_fake_geometry()
    item = pystac.Item(
        id=f"{RAS_MODELS_COLLECTION_ID}-{basename}-{realization}-ras",
        properties={},
        bbox=fake_bbox,
        datetime=datetime.now(),
        geometry=json.loads(shapely.to_geojson(fake_geometry)),
    )
    print('getting assets')
    for s in range(1, SIMULATIONS):
        assets = get_ras_output_assets(key_base, r, s)
        for asset in assets:
            item.add_asset(key=asset.title, asset=asset)
    return item


def depth_grids_for_model_run(key_base: str, s: int):
    print('filtering objects')
    basename = os.path.basename(key_base)
    return filter_objects(
        pattern=rf"^FFRD_Kanawha_Compute\/runs\/{s}\/depth-grids\/{basename}\/.*\.tif$",
        prefix=f"FFRD_Kanawha_Compute/runs/{s}/depth-grids/{basename}"
    )


def gather_depth_grid_items(key_base: str, r: int):
    basename = os.path.basename(key_base)
    realization = f"r{str(r).zfill(4)}"
    depth_grid_items = {}
    for s in range(1, SIMULATIONS):
        simulation = get_simulation_string(s)
        depth_grids = depth_grids_for_model_run(key_base, s)
        for depth_grid in depth_grids[:DEPTH_GRIDS]:
            filename = os.path.basename(depth_grid.key)
            if not filename in depth_grid_items.keys():
                bbox = get_raster_bounds(depth_grid.key)
                geometry = bbox_to_polygon(bbox)
                depth_grid_items[filename] = pystac.Item(
                    id=f"{basename}-{realization}-depth-grids-{filename}",
                    # title=f"{basename}-{realization}-{filename}"
                    properties={},
                    bbox=bbox,
                    datetime=datetime.now(),
                    geometry=json.loads(shapely.to_geojson(geometry)),
                )
            dg_asset = pystac.Asset(
                href=depth_grid.key,
                title=f"{realization}-{simulation}-{basename}-{filename}",
            )
            depth_grid_items[filename].add_asset(key=dg_asset.title, asset=dg_asset)
    return depth_grid_items.values()


def create_depth_grids_collection(key_base: str, r: int):
    basename = os.path.basename(key_base)
    realization = get_realization_string(r)
    items = gather_depth_grid_items(key_base, r)
    bboxes = [item.bbox for item in items]
    spatial_extent = pystac.SpatialExtent(bboxes)
    temporal_extent = pystac.TemporalExtent(intervals=[datetime.now(), datetime.now()])
    extent = pystac.Extent(spatial_extent, temporal_extent)
    collection = pystac.Collection(
        id=f"{basename}-{realization}-depth-grids",
        title=f"{basename}-{realization} Depth Grids",
        description=f"Depth grids for Realization {realization} of HEC-RAS model: {basename}",
        extent=extent,
    )
    collection.add_items(items)
    return collection


def filter_objects(pattern: str = None, prefix: str = None):
    compiled_pattern = re.compile(pattern) if pattern else None
    objects = []
    for obj in BUCKET.objects.filter(Prefix=prefix):
        if compiled_pattern:
            if re.match(compiled_pattern, obj.key):
                objects.append(obj)
        else:
            objects.append(obj)
    return objects


def list_ras_model_names():
    prefix = "FFRD_Kanawha_Compute/ras"
    plan_hdfs_pattern = r".*\.p01\.hdf$"
    ras_plan_hdfs = list(filter_objects(plan_hdfs_pattern, prefix))
    return [hdf.key[:-8] for hdf in ras_plan_hdfs]


def get_raster_bounds(s3_key: str):
    print(f"getting raster bounds: {s3_key}")
    s3_path = f"s3://{BUCKET_NAME}/{s3_key}"
    with rasterio.Env(AWS_SESSION):
        with rasterio.open(s3_path) as src:
            bounds = src.bounds
            crs = src.crs
            bounds_4326 = rasterio.warp.transform_bounds(crs, 'EPSG:4326', *bounds)
            return bounds_4326


def get_results_attrs(model_p01_key: str) -> dict:
    s3url = f"s3://{BUCKET_NAME}/{model_p01_key}"
    print(f'ffspec.open: {s3url}')
    s3f = fsspec.open(s3url, mode='rb')
    print(f'h5py.File')
    h5f = h5py.File(s3f.open(), mode='r')
    results_unsteady_summary_attrs = {}
    summary = h5f['Results']['Unsteady']['Summary']
    unsteady_results = h5f['Results']['Unsteady']
    for k, v in unsteady_results.attrs.items():
        results_unsteady_summary_attrs[str(k)] = str(v)
    summary = unsteady_results['Summary']
    for k, v in summary.attrs.items():
        results_unsteady_summary_attrs[str(k)] = str(v)
    for k, v in summary['Volume Accounting'].attrs.items():
        results_unsteady_summary_attrs[str(k)] = str(v)
    return results_unsteady_summary_attrs


def main():
    stac_path = Path('./stac')
    if stac_path.exists():
        shutil.rmtree(stac_path)
    stac_path.mkdir(exist_ok=True)
    catalog = create_catalog()
    ras_models_parent_collection = create_ras_models_parent_collection()

    ras_model_names = list_ras_model_names()
    for i, ras_model_key_base in enumerate(ras_model_names):
        ras_model_collection = create_ras_model_collection(ras_model_key_base)
        ras_models_parent_collection.add_child(ras_model_collection)

        realization_collection = create_ras_model_realization_collection(ras_model_key_base, 1)
        ras_model_collection.add_child(realization_collection)

        item = create_realization_ras_results_item(ras_model_key_base, 1)
        realization_collection.add_item(item)

        depth_grids_collection = create_depth_grids_collection(ras_model_key_base, 1)
        realization_collection.add_child(depth_grids_collection)


    catalog.add_child(ras_models_parent_collection)
    catalog.normalize_and_save(root_href=ROOT_HREF, catalog_type=pystac.CatalogType.SELF_CONTAINED)


if __name__ == "__main__":
    main()
