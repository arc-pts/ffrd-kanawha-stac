import boto3
from dotenv import load_dotenv
import re
import pystac
from pathlib import Path
import shutil
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Iterator, Optional, Tuple, Dict
import shapely
from shapely.geometry import shape, box
import shapely.ops
import rasterio
from rasterio.session import AWSSession
import rasterio.warp
import fsspec
import h5py
import sys
from dataclasses import dataclass
import numpy as np
import pyproj
from mypy_boto3_s3.service_resource import Object, ObjectSummary
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
file_handler = logging.FileHandler('stac.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)

load_dotenv()

s3 = boto3.resource('s3')
BUCKET_NAME = 'kanawha-pilot'
BUCKET = s3.Bucket(BUCKET_NAME)

CATALOG_TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M')
ROOT_HREF = f"./stac/kanawha-models-{CATALOG_TIMESTAMP}"

MODELS_CATALOG_ID = "kanawha-models"
RAS_MODELS_COLLECTION_ID = f"{MODELS_CATALOG_ID}-ras"

CATALOG_URL = f"https://radiantearth.github.io/stac-browser/#/external/wsp-kanawha-pilot-stac.s3.amazonaws.com/{MODELS_CATALOG_ID}-{CATALOG_TIMESTAMP}/catalog.json"

SIMULATIONS = 1001 + 1
# SIMULATIONS = 10
# DEPTH_GRIDS = 10

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


def obj_key_to_s3_url(obj_key: str) -> str:
    return f"s3://{BUCKET_NAME}/{obj_key}"


def get_ras_file_roles(filename: str) -> Optional[List[str]]:
    ext = ".".join(filename.split('.')[1:])
    ras_roles = {
        "g01": ["ras-geometry-text"],
        "g01.hdf": ["ras-geometry"],
        "p01": ["ras-plan"],
        "p01.hdf": ["ras-output"],
        "u01": ["ras-unsteady"],
        "prj": ["ras-project"],
    }
    return ras_roles.get(ext, None)


def create_ras_model_collection(key_base: str):
    logger.info(f"Creating RAS model collection: {key_base}")
    model_objs = BUCKET.objects.filter(Prefix=key_base)
    basename = os.path.basename(key_base)
    collection = pystac.Collection(
        id=f"{RAS_MODELS_COLLECTION_ID}-{basename}",
        title=f"{basename}",
        description=f"HEC-RAS Model: {basename}",
        extent=get_fake_extent(),
    )
    collection.ext.add("proj")
    collection.ext.add("file")
    for obj in model_objs:
        filename = os.path.basename(obj.key)
        asset = pystac.Asset(
            href=obj_key_to_s3_url(obj.key),
            title=filename,
        )
        asset.roles = get_ras_file_roles(filename)
        if filename.endswith('.g01.hdf'):
            geom_attrs = get_geom_attrs(obj.key)
            asset.extra_fields = geom_attrs
            geom_extents = ras_geom_extents(geom_attrs['geometry:extents'], geom_attrs['proj:wkt2'])
            spatial_extent = pystac.SpatialExtent([geom_extents.bounds])
            temporal_extent = pystac.TemporalExtent(intervals=[datetime.now(), datetime.now()])
            collection.extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)
            asset.media_type = pystac.MediaType.HDF5
        elif filename.endswith('.p01.hdf'):
            plan_attrs = get_plan_attrs(obj.key, results=False)
            asset.extra_fields = plan_attrs
            asset.media_type = pystac.MediaType.HDF5
        elif filename.endswith('.hdf'):
            asset.media_type = pystac.MediaType.HDF5
        elif filename.split('.')[-1] in ['b01', 'bco01', 'g01', 'p01', 'u01', 'x01', 'prj']:
            asset.media_type = pystac.MediaType.TEXT
        asset.extra_fields.update(get_basic_object_metadata(obj))
        asset.extra_fields = dict(sorted(asset.extra_fields.items()))
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
    logger.info(f"Getting RAS output assets: {r} {s} {key_base}")
    basename = os.path.basename(key_base)
    ras_output_objs = filter_objects(
        pattern=rf"^FFRD_Kanawha_Compute\/runs\/{s}\/ras\/{basename}\/.*$",
        prefix=f"FFRD_Kanawha_Compute/runs/{s}/ras/{basename}"
    )
    assets = []
    for obj in ras_output_objs:
        # print(obj.key)
        filename = os.path.basename(obj.key)
        s = int(obj.key.split('/')[-4])
        simulation = get_simulation_string(s)
        realization = get_realization_string(r)
        simulation_filename = f"{realization}-{simulation}_{filename}"
        asset = pystac.Asset(
            href=obj_key_to_s3_url(obj.key), # TODO: s3 url
            title=simulation_filename,
        )
        if obj.key.endswith('.p01.hdf'):
            results_attrs = get_plan_results_attrs(obj.key)
            asset.extra_fields = results_attrs
            asset.roles = ['ras-output']
            asset.media_type = pystac.MediaType.HDF5
            asset.title = f"{realization}-{simulation}-{filename}"
        elif obj.key.endswith('.log'):
            asset.roles = ['ras-output-logs']
            asset.media_type = pystac.MediaType.TEXT
            asset.title = f"{realization}-{simulation}-rasoutput.log"
        asset.extra_fields['cloud_wat:realization'] = r
        asset.extra_fields['cloud_wat:simulation'] = s
        asset.extra_fields.update(get_basic_object_metadata(obj))
        asset.extra_fields = dict(sorted(asset.extra_fields.items()))
        assets.append(asset)
    return assets


def get_ras_realization_metadata(key_base: str, r: int, s: int = 1) -> dict:
    logger.info(f"Getting RAS output metadata for realization: {r} {key_base} (s={s})")
    basename = os.path.basename(key_base)
    ras_output_objs = filter_objects(
        pattern=rf"^FFRD_Kanawha_Compute\/runs\/{s}\/ras\/{basename}\/.*\.p01.hdf$",
        prefix=f"FFRD_Kanawha_Compute/runs/{s}/ras/{basename}",
    )
    plan_attrs = {
        'cloud_wat:realization': r,
    }
    for obj in ras_output_objs:
        if obj.key.endswith(".p01.hdf"):
            plan_attrs.update(get_plan_attrs(obj.key, results=False))
            return plan_attrs
    return plan_attrs


def asset_field_values(assets: List[pystac.Asset], field: str, media_type: pystac.MediaType = pystac.MediaType.HDF5) -> list:
    values = []
    for asset in assets:
        if asset.media_type == media_type:
            value = asset.extra_fields.get(field)
            if value is not None:
                values.append(value)
    return values


def get_ras_simulation_stats(assets: List[pystac.Asset]) -> dict:
    logger.info("Getting RAS simulation stats")
    computation_time_total_minutes = asset_field_values(assets, 'results_summary:computation_time_total_minutes')
    run_time_windows = asset_field_values(assets, 'results_summary:run_time_window')
    run_time_starts = [i[0] for i in run_time_windows]
    run_time_stops = [i[1] for i in run_time_windows]
    error_percents = asset_field_values(assets, 'volume_accounting:error_percent')
    solutions = asset_field_values(assets, 'results_summary:solution')
    stats = {
        "cloud_wat:simulations": len([a for a in assets if a.has_role('ras-output') and a.media_type == pystac.MediaType.HDF5]),
        "cloud_wat:min_computation_time_mins": min(computation_time_total_minutes),
        "cloud_wat:max_computation_time_mins": max(computation_time_total_minutes),
        "cloud_wat:avg_computation_time_mins": np.mean(computation_time_total_minutes),
        "cloud_wat:total_computation_time_hrs": np.sum(computation_time_total_minutes) / 60,
        "cloud_wat:run_time_window": [min(run_time_starts), max(run_time_stops)],
        "cloud_wat:min_volume_error_percent": min(error_percents),
        "cloud_wat:max_volume_error_percent": max(error_percents),
        "cloud_wat:avg_volume_error_percent": np.mean(error_percents),
        "cloud_wat:unsuccessful_runs": len([s for s in solutions if s != "Unsteady Finished Successfully"]),
    }
    return stats


def create_realization_ras_results_item(key_base: str, r: int):
    logger.info(f"Creating realization RAS results item: {key_base}, {r}")
    basename = os.path.basename(key_base)
    realization = f"r{str(r).zfill(4)}"
    geometry = get_2d_flow_area_perimeter(key_base + '.g01.hdf')
    bbox = geometry.bounds
    properties = get_ras_realization_metadata(key_base, r)
    item = pystac.Item(
        id=f"{basename}-{realization}",
        properties=properties,
        bbox=bbox,
        datetime=datetime.now(),
        geometry=json.loads(shapely.to_geojson(geometry)),
    )
    item.ext.add("proj")
    for s in range(1, SIMULATIONS):
        assets = get_ras_output_assets(key_base, r, s)
        for asset in assets:
            item.add_asset(key=asset.title, asset=asset)
    item.properties.update(get_ras_simulation_stats(item.assets.values()))
    return item


def depth_grids_for_model_run(key_base: str, s: int):
    basename = os.path.basename(key_base)
    return filter_objects(
        pattern=rf"^FFRD_Kanawha_Compute\/runs\/{s}\/depth-grids\/{basename}\/.*\.tif$",
        prefix=f"FFRD_Kanawha_Compute/runs/{s}/depth-grids/{basename}"
    )


def get_basic_object_metadata(obj: ObjectSummary) -> dict:
    return {
        'file:size': obj.size,
        'e_tag': obj.e_tag,
        'last_modified': obj.last_modified.isoformat(),
        'storage:platform': 'AWS',
        'storage:region': obj.meta.client.meta.region_name,
        'storage:tier': obj.storage_class,
    }


def gather_depth_grid_items(key_base: str, r: int):
    basename = os.path.basename(key_base)
    realization = f"r{str(r).zfill(4)}"
    depth_grid_items: Dict[str, pystac.Item] = {}
    raster_bounds = {}
    for s in range(1, SIMULATIONS):
        simulation = get_simulation_string(s)
        logger.info(f"Gathering depth grid items for {simulation}, {basename}")
        depth_grids = depth_grids_for_model_run(key_base, s)
        # for depth_grid in depth_grids[:DEPTH_GRIDS]:
        for depth_grid in depth_grids:
            filename = os.path.basename(depth_grid.key)
            if not filename in depth_grid_items.keys():
                if not filename in raster_bounds:
                    bbox = get_raster_bounds(depth_grid.key)
                    raster_bounds[filename] = bbox
                else:
                    bbox = raster_bounds[filename]
                geometry = bbox_to_polygon(bbox)
                depth_grid_items[filename] = pystac.Item(
                    id=f"{basename}-{realization}-{filename}",
                    # title=f"{basename}-{realization}-{filename}"
                    properties={},
                    bbox=bbox,
                    datetime=datetime.now(),
                    geometry=json.loads(shapely.to_geojson(geometry)),
                )
            # non_null = not raster_is_all_null(depth_grid.key)
            dg_asset = pystac.Asset(
                href=obj_key_to_s3_url(depth_grid.key),
                title=f"{realization}-{simulation}-{basename}-{filename}",
                media_type=pystac.MediaType.GEOTIFF,
                roles=['ras-depth-grid'],
                extra_fields={
                    'cloud_wat:realization': r,
                    'cloud_wat:simulation': s,
                    # 'non_null': non_null,
                },
            )
            dg_asset.extra_fields.update(get_basic_object_metadata(depth_grid))
            dg_asset.extra_fields = dict(sorted(dg_asset.extra_fields.items()))
            # dg_metadata = get_raster_metadata(depth_grid.key)
            # if dg_metadata:
            #     dg_asset.extra_fields.update(dg_metadata)
            depth_grid_items[filename].add_asset(key=dg_asset.title, asset=dg_asset)
            depth_grid_items[filename].datetime = get_datetime_from_item_assets(depth_grid_items[filename])
    return depth_grid_items.values()


def get_items_temporal_extent(items: List[pystac.Item]) -> pystac.TemporalExtent:
    item_datetimes = [item.datetime for item in items]
    dt_min = min(item_datetimes)
    dt_max = max(item_datetimes)
    return pystac.TemporalExtent(intervals=[dt_min, dt_max])


def create_depth_grids_collection(key_base: str, r: int):
    logger.info(f"Creating depth grids collection: {key_base}, {r}")
    basename = os.path.basename(key_base)
    realization = get_realization_string(r)
    items = gather_depth_grid_items(key_base, r)
    bboxes = [item.bbox for item in items]
    spatial_extent = pystac.SpatialExtent(bboxes)
    # temporal_extent = pystac.TemporalExtent(intervals=[datetime.now(), datetime.now()])
    temporal_extent = get_items_temporal_extent(items)
    extent = pystac.Extent(spatial_extent, temporal_extent)
    collection = pystac.Collection(
        id=f"{basename}-{realization}-depth-grids",
        title=f"{basename}-{realization} Depth Grids",
        description=f"Depth grids for Realization {realization} of HEC-RAS model: {basename}",
        extent=extent,
    )
    collection.add_items(items)
    return collection


def filter_objects(pattern: str = None, prefix: str = None) -> List[Object]:
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
    # print(f"getting raster bounds: {s3_key}")
    s3_path = f"s3://{BUCKET_NAME}/{s3_key}"
    with rasterio.Env(AWS_SESSION):
        with rasterio.open(s3_path) as src:
            bounds = src.bounds
            crs = src.crs
            bounds_4326 = rasterio.warp.transform_bounds(crs, 'EPSG:4326', *bounds)
            return bounds_4326


def raster_is_all_null(s3_key: str) -> bool:
    """
    Opens a GeoTIFF file from an S3 URL using Rasterio.
    Returns False if any raster cells are non-null, True if all cells are null.
    """
    s3_path = f"s3://{BUCKET_NAME}/{s3_key}"
    # Open the GeoTIFF file from S3
    with rasterio.Env(AWS_SESSION):
        with rasterio.open(s3_path) as dataset:
            # Iterate over windows (chunks) of the dataset
            for ji, window in dataset.block_windows(1):
                # Read the data in the current window
                data = dataset.read(window=window)

                # Check if there are any non-null cells
                if np.any(data != dataset.nodata):
                    return False
    return True


def get_raster_metadata(s3_key: str) -> dict:
    s3_path = f"s3://{BUCKET_NAME}/{s3_key}"
    with rasterio.Env(AWS_SESSION):
        with rasterio.open(s3_path) as src:
            return src.tags(1)


# def get_raster_info(s3_key: str) -> dict:
    # print(f"getting raster bounds: {s3_key}")
    # s3_path = f"s3://{BUCKET_NAME}/{s3_key}"
    

def to_snake_case(text):
    """
    Convert a string to snake case, removing punctuation and other symbols.
    
    Args:
    text (str): The string to be converted.

    Returns:
    str: The snake case version of the string.
    """
    import re

    # Remove all non-word characters (everything except numbers and letters)
    text = re.sub(r'[^\w\s]', '', text)

    # Replace all runs of whitespace with a single underscore
    text = re.sub(r'\s+', '_', text)

    # Convert to lower case
    return text.lower()


def convert_hdf5_string(value: str):
    ras_datetime_format1_re = r"\d{2}\w{3}\d{4} \d{2}:\d{2}:\d{2}"
    ras_datetime_format2_re = r"\d{2}\w{3}\d{4} \d{2}\d{2}"
    s = value.decode('utf-8')
    if s == "True":
        return True
    elif s == "False":
        return False
    elif re.match(rf"^{ras_datetime_format1_re}", s):
        if re.match(rf"^{ras_datetime_format1_re} to {ras_datetime_format1_re}$", s):
            split = s.split(" to ")
            return [
                parse_ras_datetime(split[0]).isoformat(),
                parse_ras_datetime(split[1]).isoformat(),
            ]
        return parse_ras_datetime(s).isoformat()
    elif re.match(rf"^{ras_datetime_format2_re}", s):
        if re.match(rf"^{ras_datetime_format2_re} to {ras_datetime_format2_re}$", s):
            split = s.split(" to ")
            return [
                parse_ras_simulation_window_datetime(split[0]).isoformat(),
                parse_ras_simulation_window_datetime(split[1]).isoformat(),
            ]
        return parse_ras_simulation_window_datetime(s).isoformat()
    return s 


def convert_hdf5_value(value):
    # TODO (?): handle "8-bit bitfield" values in 2D Flow Area groups

    # Check for NaN (np.nan)
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    
    # Check for byte strings
    elif isinstance(value, bytes) or isinstance(value, np.bytes_):
        return convert_hdf5_string(value)
    
    # Check for NumPy integer or float types
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    
    # Leave regular ints and floats as they are
    elif isinstance(value, (int, float)):
        return value

    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 1:
            return [convert_hdf5_value(v) for v in value]
        else:
            return convert_hdf5_value(value[0])
    
    # Convert all other types to string
    else:
        return str(value) 


def hdf5_attrs_to_dict(attrs, prefix: str = None) -> dict:
    results = {}
    for k, v in attrs.items():
        value = convert_hdf5_value(v)
        if prefix:
            key = f"{to_snake_case(prefix)}:{to_snake_case(k)}"
        else:
            key = to_snake_case(k)
        results[key] = value
    return results


def parse_simulation_time_window(window: str) -> Tuple[datetime, datetime]:
    split = window.split(' to ')
    format = '%d%b%Y %H%M'
    begin = datetime.strptime(split[0], format)
    end = datetime.strptime(split[1], format)
    return begin, end


def parse_ras_datetime(datetime_str: str) -> datetime:
    format = '%d%b%Y %H:%M:%S'
    return datetime.strptime(datetime_str, format)


def parse_ras_simulation_window_datetime(datetime_str) -> datetime:
    format = '%d%b%Y %H%M'
    return datetime.strptime(datetime_str, format)


def parse_run_time_window(window: str) -> Tuple[datetime, datetime]:
    split = window.split(' to ')
    begin = parse_ras_datetime(split[0])
    end = parse_ras_datetime(split[1])
    return begin, end


def parse_duration(duration_str: str) -> timedelta:
    # Split the duration string into hours, minutes, and seconds
    hours, minutes, seconds = map(int, duration_str.split(':'))
    # Create a timedelta object
    duration = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return duration


def geom_to_4326(s: shapely.Geometry, proj_wkt: str) -> shapely.Geometry:
    source_crs = pyproj.CRS.from_wkt(proj_wkt)
    target_crs = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_proj(source_crs, target_crs, always_xy=True)
    return shapely.ops.transform(transformer.transform, s)


def ras_geom_extents(extents, proj_wkt: str) -> shapely.Polygon:
    # min_x, max_x, min_y, max_y = [float(x) for x in extents_str[1:-1].split()]
    min_x, max_x, min_y, max_y = extents
    # source_crs = pyproj.CRS.from_wkt(proj_wkt)
    # target_crs = pyproj.CRS.from_epsg(4326)
    # transformer = pyproj.Transformer.from_proj(source_crs, target_crs, always_xy=True)
    extents = shapely.Polygon([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ])
    # extents_transformed = shapely.ops.transform(transformer.transform, extents)
    extents_transformed = geom_to_4326(extents, proj_wkt)
    return extents_transformed


def open_s3_hdf5(s3_hdf5_key: str) -> h5py.File:
    s3url = f"s3://{BUCKET_NAME}/{s3_hdf5_key}"
    s3f = fsspec.open(s3url, mode='rb')
    return h5py.File(s3f.open(), mode='r')


def get_first_group(parent_group: h5py.Group) -> Optional[h5py.Group]:
    for _, item in parent_group.items():
        if isinstance(item, h5py.Group):
            return item
    return None


def get_geom_attrs(model_g01_key: str) -> dict:
    h5f = open_s3_hdf5(model_g01_key)    

    attrs = {}
    top_attrs = hdf5_attrs_to_dict(h5f.attrs)
    projection = top_attrs.pop("projection", None)
    if projection is not None:
        top_attrs["proj:wkt2"] = projection
    attrs.update(top_attrs)

    geometry = h5f['Geometry']
    geometry_attrs = hdf5_attrs_to_dict(geometry.attrs, prefix="geometry")
    attrs.update(geometry_attrs)

    structures = geometry['Structures']
    structures_attrs = hdf5_attrs_to_dict(structures.attrs, prefix="structures")
    attrs.update(structures_attrs)

    d2_flow_area = get_first_group(geometry['2D Flow Areas'])
    d2_flow_area_attrs = hdf5_attrs_to_dict(d2_flow_area.attrs, prefix="2d_flow_area")
    cell_average_size = d2_flow_area_attrs.get('2d_flow_area:cell_average_size', None)
    if cell_average_size is not None:
        d2_flow_area_attrs["2d_flow_area:cell_average_length"] = cell_average_size ** 0.5
    attrs.update(d2_flow_area_attrs)

    return attrs


def get_plan_attrs(model_p01_key: str, results: bool = True) -> dict:
    h5f = open_s3_hdf5(model_p01_key)    

    attrs = {}
    top_attrs = hdf5_attrs_to_dict(h5f.attrs)
    projection = top_attrs.pop("projection", None)
    if projection is not None:
        top_attrs["proj:wkt2"] = projection
    attrs.update(top_attrs)

    plan_data = h5f['Plan Data']

    plan_information = plan_data['Plan Information']
    plan_info_attrs = hdf5_attrs_to_dict(plan_information.attrs, prefix="Plan Information")
    attrs.update(plan_info_attrs)

    plan_parameters = plan_data['Plan Parameters']
    plan_param_attrs = hdf5_attrs_to_dict(plan_parameters.attrs, prefix="Plan Parameters")
    attrs.update(plan_param_attrs)

    precip = h5f['Event Conditions']['Meteorology']['Precipitation']
    precip_attrs = hdf5_attrs_to_dict(precip.attrs, prefix="Meteorology")
    precip_attrs.pop("meteorology:projection", None)
    attrs.update(precip_attrs)

    if results:
        plan_results_attrs = get_plan_results_attrs(model_p01_key, h5f=h5f)
        attrs.update(plan_results_attrs)
    return attrs


def get_plan_results_attrs(model_p01_key: str, h5f: Optional[h5py.File] = None) -> dict:
    if not h5f:
        h5f = open_s3_hdf5(model_p01_key)
    results_attrs = {}

    unsteady_results = h5f['Results']['Unsteady']
    unsteady_results_attrs = hdf5_attrs_to_dict(unsteady_results.attrs, prefix="Unsteady Results")
    results_attrs.update(unsteady_results_attrs)

    summary = unsteady_results['Summary']
    summary_attrs = hdf5_attrs_to_dict(summary.attrs, prefix="Results Summary")
    computation_time_total = summary_attrs['results_summary:computation_time_total']
    computation_time_total_minutes = parse_duration(computation_time_total).total_seconds() / 60
    results_summary = {
        "results_summary:computation_time_total": computation_time_total,
        "results_summary:computation_time_total_minutes": computation_time_total_minutes,
        "results_summary:run_time_window": summary_attrs.get("results_summary:run_time_window"),
        "results_summary:solution": summary_attrs.get("results_summary:solution"),
    }
    results_attrs.update(results_summary)

    volume_accounting = summary['Volume Accounting']
    volume_accounting_attrs = hdf5_attrs_to_dict(volume_accounting.attrs, prefix="Volume Accounting")
    results_attrs.update(volume_accounting_attrs)

    return results_attrs


def asset_extra_fields_intersection(item: pystac.Item) -> dict:
    extra_fields_to_intersect = []
    for key, asset in item.assets.items():
        # if key.endswith('.p01.hdf'):
        if asset.media_type == pystac.MediaType.HDF5 and asset.has_role('ras-output'):
            extra_fields_to_intersect.append(asset.extra_fields)
    intersection = extra_fields_to_intersect[0].copy()
    # print(intersection)
    for d in extra_fields_to_intersect[1:]:
        # print(d)
        # print(d.items() & intersection.items())
        # intersection = dict(d.items() & intersection.items())
        intersection = {k: v for k, v in intersection.items() if k in d and d[k] == v}
    return intersection


def drop_common_fields(extra_fields: dict, common: dict) -> dict:
    difference = set(extra_fields) - set(common)
    result = {k: extra_fields[k] for k in difference} 
    realization = extra_fields.get('cloud_wat:realization')
    if realization is not None:
        result['cloud_wat:realization'] = realization  # don't drop the realization number
    return result


def dedupe_asset_metadata(item: pystac.Item):
    for k, v in item.assets.items():
        # if k.endswith('.p01.hdf'):
        if v.media_type == pystac.MediaType.HDF5 and v.has_role('ras-output'):
            deduped_extra_fields = drop_common_fields(v.extra_fields, item.properties)
            item.assets[k].extra_fields = deduped_extra_fields


def get_2d_flow_area_perimeter(model_g01_key) -> Optional[shapely.Polygon]:
    h5f = open_s3_hdf5(model_g01_key)
    projection = h5f.attrs['Projection'].decode()
    d2_flow_area = get_first_group(h5f['Geometry']['2D Flow Areas'])
    if not d2_flow_area:
        return None
    perim = d2_flow_area['Perimeter']
    perim_coords = perim[:]
    perim_polygon = shapely.Polygon(perim_coords).simplify(0.001)
    return geom_to_4326(perim_polygon, projection)


def get_datetime_from_item_assets(item: pystac.Item) -> datetime:
    latest = datetime.fromtimestamp(0, tz=timezone.utc)
    for asset in item.assets.values():
        last_modified = asset.extra_fields.get('last_modified')
        if last_modified:
            dt = datetime.fromisoformat(last_modified)
            if dt > latest:
                latest = dt
    return latest


def get_temporal_extent_from_item_assets(item: pystac.Item) -> pystac.TemporalExtent:
    assets = item.assets.values()
    asset_datetimes = []
    for asset in assets:
        last_modified = asset.extra_fields.get('last_modified')
        if last_modified:
            asset_datetimes.append(datetime.fromisoformat(last_modified))
    dt_min = min(asset_datetimes)
    dt_max = max(asset_datetimes)
    return pystac.TemporalExtent(intervals=[dt_min, dt_max])


def get_temporal_extent_from_collections(collections: List[pystac.Collection]) -> pystac.TemporalExtent:
    temporal_extents = [c.extent.temporal.intervals for c in collections]
    logger.info(temporal_extents)
    starts = [t[0][0] for t in temporal_extents]
    stops = [t[0][1] for t in temporal_extents]
    dt_min = min(starts)
    dt_max = max(stops)
    return pystac.TemporalExtent(intervals=[dt_min, dt_max])


def main():
    t1 = datetime.now()
    stac_path = Path('./stac')
    if stac_path.exists():
        shutil.rmtree(stac_path)
    stac_path.mkdir(exist_ok=True)
    catalog = create_catalog()
    ras_models_parent_collection = create_ras_models_parent_collection()

    ras_model_bboxes = []

    ras_model_names = list_ras_model_names()
    for _, ras_model_key_base in enumerate(ras_model_names):
        logger.info(ras_model_key_base)
        ras_model_collection = create_ras_model_collection(ras_model_key_base)
        ras_model_bboxes.extend(ras_model_collection.extent.spatial.bboxes)
        ras_models_parent_collection.add_child(ras_model_collection)

        logger.info(f"Creating realization collection: {ras_model_key_base}")
        realization_collection = create_ras_model_realization_collection(ras_model_key_base, 1)
        realization_collection.extent = ras_model_collection.extent
        ras_model_collection.add_child(realization_collection)

        r = 1
        logger.info(f"Creating realization item: r={r} {ras_model_key_base}")
        item = create_realization_ras_results_item(ras_model_key_base, r)
        # item.properties = asset_extra_fields_intersection(item)
        logger.info("Setting item datetime based on assets")
        item.datetime = get_datetime_from_item_assets(item)
        logger.info("Adding item to realization collection")
        realization_collection.add_item(item)
        # dedupe_asset_metadata(item)

        logger.info("Settings temporal extent based on item assets")
        realization_collection.extent.temporal = get_temporal_extent_from_item_assets(item)

        logger.info("Creating depth grids collection")
        depth_grids_collection = create_depth_grids_collection(ras_model_key_base, 1)
        logger.info("Adding depth grids collection to realization collection")
        realization_collection.add_child(depth_grids_collection)
    
    spatial_extent = pystac.SpatialExtent(ras_model_bboxes)
    # temporal_extent = pystac.TemporalExtent(intervals=[datetime.now(), datetime.now()])
    temporal_extent = get_temporal_extent_from_collections(ras_models_parent_collection.get_children())
    ras_models_parent_collection.extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

    logger.info("Adding ras models collection to parent catalog")
    catalog.add_child(ras_models_parent_collection)
    logger.info("Saving catalog")
    catalog.normalize_and_save(root_href=ROOT_HREF, catalog_type=pystac.CatalogType.SELF_CONTAINED)
    logger.info("Done.")
    t2 = datetime.now()
    logger.info(f"Took: {(t2 - t1).total_seconds() / 60:0.2f} min")

if __name__ == "__main__":
    main()
