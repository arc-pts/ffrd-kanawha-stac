import logging
from datetime import datetime
import os
from pathlib import Path
import pystac
import shutil
import s3fs
from dotenv import load_dotenv

from build_static_catalog import (
    create_catalog,
    get_fake_extent,
    obj_key_to_s3_url,
    get_ras_file_roles,
    ras_geom_extents,
    get_temporal_extent_from_collections,
)
from ffrd_stac.rasmeta import RasHdf
from ffrd_stac.zipmeta import S3Zip
from ffrd_stac.utils import flatten_dict_list

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler()
file_handler = logging.FileHandler("stac.log")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)

load_dotenv()

WATERSHED_NAME = "trinity"
BUCKET_NAME = f"{WATERSHED_NAME}-pilot"
CATALOG_TYPE = "zips"

CATALOG_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")
ROOT_HREF = f"./stac/{WATERSHED_NAME}-{CATALOG_TYPE}-{CATALOG_TIMESTAMP}"

ZIP_COLLECTION_PREFIX = "common-files/01_DataPrep/ExistingModels/DFW CDC"
S3_ZIPFILE_NAME = "Final_Model_Inundation.zip"

# ZIP_COLLECTION_PREFIX="common-files/01_DataPrep/ExistingModels/_From_MIP"
# S3_ZIPFILE_NAME = "_Trinity_Hydraulic_Studies_Part1.zip"


S3_ZIPFILE_KEY = f"{ZIP_COLLECTION_PREFIX}/{S3_ZIPFILE_NAME}"
ZIP_COLLECTION_ID = f"{WATERSHED_NAME}-zip-collection"


def create_zip_collection(watershed_name: str = "Trinity"):
    collection = pystac.Collection(
        id=ZIP_COLLECTION_ID,
        title=S3_ZIPFILE_NAME,
        description=f"{S3_ZIPFILE_NAME} zip collection for the {watershed_name}",
        extent=get_fake_extent(),
    )
    return collection


def create_zipped_ras_model_collection(
    s3_zip_file_key: str, zip_collection_id: str, model_objs: dict
):
    prj_file = model_objs["ras_prj_file"]
    logger.info(f"Creating RAS model collection: {prj_file}")
    basename = os.path.basename(prj_file)
    collection = pystac.Collection(
        id=f"{zip_collection_id}-ras-{basename}",
        title=f"{basename}",
        description=f"HEC-RAS Model: {basename}",
        extent=get_fake_extent(),
    )
    collection.ext.add("proj")
    collection.ext.add("file")

    for hdf_file in model_objs["geom_hdfs"]:
        asset = pystac.Asset(
            href=obj_key_to_s3_url(f"{s3_zip_file_key}/{hdf_file}"),
            title=hdf_file,
        )
        asset.roles = get_ras_file_roles(hdf_file)
        ras_hdf_zipfile = RasHdf.hdf_from_zip(
            BUCKET_NAME, S3_ZIPFILE_KEY, hdf_file, "geom", mode="r"
        )
        geom_attrs = ras_hdf_zipfile.get_geom_attrs()
        asset.extra_fields = geom_attrs
        if "proj:wkt2" in geom_attrs.keys() and "geometry:extents" in geom_attrs.keys():
            geom_extents = ras_geom_extents(
                geom_attrs["geometry:extents"], geom_attrs["proj:wkt2"]
            )
            spatial_extent = pystac.SpatialExtent([geom_extents.bounds])
        temporal_extent = pystac.TemporalExtent(
            intervals=[datetime.now(), datetime.now()]
        )
        collection.extent = pystac.Extent(
            spatial=spatial_extent, temporal=temporal_extent
        )
        asset.media_type = pystac.MediaType.HDF5
        collection.add_asset(key=hdf_file, asset=asset)

    for hdf_file in model_objs["plan_hdfs"]:
        asset = pystac.Asset(
            href=obj_key_to_s3_url(f"{s3_zip_file_key}/{hdf_file}"),
            title=hdf_file,
        )
        ras_hdf_zipfile = RasHdf.hdf_from_zip(
            BUCKET_NAME, S3_ZIPFILE_KEY, hdf_file, "plan", mode="r"
        )
        plan_attrs = ras_hdf_zipfile.get_plan_attrs()
        asset.extra_fields = plan_attrs
        asset.media_type = pystac.MediaType.HDF5
        collection.add_asset(key=hdf_file, asset=asset)

    for hdf_file in model_objs["tmp_hdfs"]:
        asset = pystac.Asset(
            href=obj_key_to_s3_url(f"{s3_zip_file_key}/{hdf_file}"),
            title=hdf_file,
        )
        asset.media_type = pystac.MediaType.HDF5
        collection.add_asset(key=hdf_file, asset=asset)

    other_keys = [
        str(k)
        for k in model_objs.keys()
        if k not in ["geom_hdfs", "plan_hdfs", "tmp_hdfs", "ras_prj_file"]
    ]
    for key in other_keys:
        for f in model_objs[key]:
            asset = pystac.Asset(
                href=obj_key_to_s3_url(f"{s3_zip_file_key}/{f}"),
                title=f,
            )
            collection.add_asset(key=f, asset=asset)

    # asset.extra_fields.update(get_basic_object_metadata(obj))
    # asset.extra_fields = dict(sorted(asset.extra_fields.items()))
    # collection.add_asset(key=filename, asset=asset)
    return collection


def main():
    t1 = datetime.now()
    stac_path = Path("./stac")
    if stac_path.exists():
        shutil.rmtree(stac_path)
    stac_path.mkdir(exist_ok=True)

    fs = s3fs.S3FileSystem()

    zfile = S3Zip(BUCKET_NAME, S3_ZIPFILE_KEY, fs)

    ras_model_files = flatten_dict_list(zfile.ras_models)

    # TODO: add non_ras files to top level collection
    # TODO: add other file types to top level collection
    non_ras_files = [f for f in zfile.contents if f not in ras_model_files]

    catalog = create_catalog()
    zip_collection = create_zip_collection()

    ras_model_bboxes = []

    # ras_model_names = [m["ras_prj_file"] for m in zfile.ras_models]

    for ras_model in zfile.ras_models:
        
        ras_model_collection = create_zipped_ras_model_collection(
            S3_ZIPFILE_KEY, ZIP_COLLECTION_ID, ras_model
        )
        ras_model_bboxes.extend(ras_model_collection.extent.spatial.bboxes)
        zip_collection.add_child(ras_model_collection)
        spatial_extent = pystac.SpatialExtent(ras_model_bboxes)
        temporal_extent = get_temporal_extent_from_collections(
            zip_collection.get_children()
        )
        zip_collection.extent = pystac.Extent(
            spatial=spatial_extent, temporal=temporal_extent
        )

        logger.info("Adding ras models collection to parent catalog")
        catalog.add_child(zip_collection)
        logger.info("Saving catalog")
        catalog.normalize_and_save(
            root_href=ROOT_HREF, catalog_type=pystac.CatalogType.SELF_CONTAINED
        )
        logger.info("Done.")
        t2 = datetime.now()
        logger.info(f"Took: {(t2 - t1).total_seconds() / 60:0.2f} min")


if __name__ == "__main__":
    main()

# https://radiantearth.github.io/stac-browser/#/external/trinity-pilot.s3.amazonaws.com/stac/collection.json?.language=en