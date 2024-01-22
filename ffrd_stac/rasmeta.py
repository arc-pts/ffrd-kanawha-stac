from datetime import datetime, timedelta
import re
from typing import Any, Optional, Tuple
import warnings
import zipfile 
import os
import fsspec
import h5py
import numpy as np
import pyproj
import shapely
import shapely.ops
import boto3
from io import BytesIO
import logging


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


def get_first_hdf_group(parent_group: h5py.Group) -> Optional[h5py.Group]:
    for _, item in parent_group.items():
        if isinstance(item, h5py.Group):
            return item
    return None


def geom_to_4326(s: shapely.Geometry, proj_wkt: str) -> shapely.Geometry:
    source_crs = pyproj.CRS.from_wkt(proj_wkt)
    target_crs = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_proj(source_crs, target_crs, always_xy=True)
    return shapely.ops.transform(transformer.transform, s)

class RasHdf(h5py.File):
    def __init__(self, name, mode='r', **kwargs):
        super().__init__(name, mode, **kwargs)

    @classmethod
    def open_url(cls, url: str, mode: str = "r", **kwargs):
        s3f = fsspec.open(url, mode="rb")
        return cls(s3f.open(), mode, **kwargs)

    def get_attrs(self):
        attrs = hdf5_attrs_to_dict(self.attrs)
        projection = attrs.pop("projection", None)
        if projection is not None:
            attrs["proj:wkt2"] = projection
        return attrs

    @classmethod
    def hdf_from_zip(cls, bucket_name, zip_file_key, hdf_file_name, ras_hdf_type, mode='r', **kwargs):
        """
        acceptable parameters for ras_hdf_type are `plan` or `geom`
        """
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=zip_file_key)

        with zipfile.ZipFile(BytesIO(response['Body'].read())) as zip_ref:
            if hdf_file_name not in zip_ref.namelist():
                raise FileNotFoundError(f"{hdf_file_name} not found in the zip file.")

            with zip_ref.open(hdf_file_name) as hdf_file:
                hdf_content = hdf_file.read()

        if ras_hdf_type == "plan":
            return RasPlanHdf(BytesIO(hdf_content), mode, **kwargs)
        elif ras_hdf_type == "geom":    
            return RasGeomHdf(BytesIO(hdf_content), mode, **kwargs)
        else:
            return cls(BytesIO(hdf_content), mode, **kwargs)


class RasPlanHdf(RasHdf):

    def __init__(self, name, mode='r', **kwargs):
        super().__init__(name, mode, **kwargs)

    def get_plan_attrs(self, include_results: bool = False):
        attrs = self.get_attrs()

        plan_info = self.get('Plan Data/Plan Information')
        if plan_info is not None:
            plan_info_attrs = hdf5_attrs_to_dict(plan_info.attrs, prefix="Plan Information")
            attrs.update(plan_info_attrs)

        plan_params = self.get('Plan Data/Plan Parameters')
        if plan_params is not None:
            plan_params_attrs = hdf5_attrs_to_dict(plan_params.attrs, prefix="Plan Parameters")
            attrs.update(plan_params_attrs)

        precip = self.get('Event Conditions/Meteorology/Precipitation')
        if precip is not None:
            precip_attrs = hdf5_attrs_to_dict(precip.attrs, prefix="Meteorology")
            precip_attrs.pop("meteorology:projection", None)
            attrs.update(precip_attrs)

        if include_results:
            attrs.update(self.get_plan_results_attrs())

        return attrs

    def get_plan_results_attrs(self):
        attrs = {}

        unsteady_results = self.get('Results/Unsteady')
        if unsteady_results is not None:
            unsteady_results_attrs = hdf5_attrs_to_dict(unsteady_results.attrs, prefix="Unsteady Results")
            attrs.update(unsteady_results_attrs)
        
        summary = self.get('Results/Unsteady/Summary')
        if summary is not None:
            summary_attrs = hdf5_attrs_to_dict(summary.attrs, prefix="Results Summary")
            computation_time_total = summary_attrs.get('results_summary:computation_time_total')
            results_summary = {
                "results_summary:computation_time_total": computation_time_total,
                "results_summary:run_time_window": summary_attrs.get("results_summary:run_time_window"),
                "results_summary:solution": summary_attrs.get("results_summary:solution"),
            }
            if computation_time_total is not None:
                computation_time_total_minutes = parse_duration(computation_time_total).total_seconds() / 60
                results_summary['results_summary:computation_time_total_minutes'] = computation_time_total_minutes
            attrs.update(results_summary)

        volume_accounting = self.get('Results/Unsteady/Summary/Volume Accounting')
        if volume_accounting is not None:
            volume_accounting_attrs = hdf5_attrs_to_dict(volume_accounting.attrs, prefix="Volume Accounting")
            attrs.update(volume_accounting_attrs)

        return attrs


class RasGeomHdf(RasHdf):

    def __init__(self, name, mode='r', **kwargs):
        super().__init__(name, mode, **kwargs) 

    def get_geom_attrs(self):
        attrs = self.get_attrs()

        geometry = self.get('Geometry')
        if geometry is not None:
            geometry_attrs = hdf5_attrs_to_dict(geometry.attrs, prefix="Geometry")
            attrs.update(geometry_attrs)

        structures = self.get('Geometry/Structures')
        if structures is not None:
            structures_attrs = hdf5_attrs_to_dict(structures.attrs, prefix="Structures")
            attrs.update(structures_attrs)

        try:
            d2_flow_area = get_first_hdf_group(self.get('Geometry/2D Flow Areas'))
        except AttributeError:
            logging.warning("Unable to get 2D Flow Area; Geometry/2D Flow Areas group not found in HDF5 file.")   
            return attrs
        
        if d2_flow_area is not None:
            d2_flow_area_attrs = hdf5_attrs_to_dict(d2_flow_area.attrs, prefix="2D Flow Areas")
            cell_average_size = d2_flow_area_attrs.get('2d_flow_area:cell_average_size', None)
            if cell_average_size is not None:
                d2_flow_area_attrs["2d_flow_area:cell_average_length"] = cell_average_size ** 0.5
            attrs.update(d2_flow_area_attrs)

        return attrs

    def get_projection(self) -> Optional[str]:
        try:
            projection = self.attrs.get("Projection")
        except AttributeError:
            logging.warning("Unable to get projection; Projection attribute not found in HDF5 file.")   
            return None 
        if projection is not None:
            return projection.decode('utf-8')

    def get_2d_flow_area_perimeter(self, simplify: float = 0.001, wgs84: bool = True) -> Optional[shapely.Polygon]:
        try:
            d2_flow_area = get_first_hdf_group(self.get('Geometry/2D Flow Areas'))
        except AttributeError:
            logging.warning("Unable to get 2D Flow Area perimeter; Geometry/2D Flow Areas group not found in HDF5 file.")   
            return None
        
        if d2_flow_area is None:
            return None
        
        perim = d2_flow_area.get('Perimeter')
        if perim is None:
            return None
        
        perim_coords = perim[:]
        perim_polygon = shapely.Polygon(perim_coords).simplify(simplify)
        if wgs84:
            proj_wkt = self.get_projection()
            if proj_wkt is not None:
                return geom_to_4326(perim_polygon, proj_wkt)
            warnings.warn("Unable to convert 2D Flow Area perimeter to WGS 84 (EPSG:4326); projection not specified in HDF5 file.")
            
        return perim_polygon
