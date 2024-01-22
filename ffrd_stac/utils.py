import fsspec
from mypy_boto3_s3.service_resource import Bucket, Object
from ffrd_stac.rasmeta import RasGeomHdf, RasPlanHdf

import copy
import re
from typing import Any, List


def filter_objects(bucket: Bucket, pattern: str = None, prefix: str = None) -> List[Object]:
    compiled_pattern = re.compile(pattern) if pattern else None
    objects = []
    for obj in bucket.objects.filter(Prefix=prefix):
        if compiled_pattern:
            if re.match(compiled_pattern, obj.key):
                objects.append(obj)
        else:
            objects.append(obj)
    return objects


def list_ras_model_names(bucket: Bucket, prefix: str) -> list:
    plan_hdfs_pattern = r".*\.p01\.hdf$"
    ras_plan_hdfs = list(filter_objects(bucket, prefix=prefix, pattern=plan_hdfs_pattern))
    return [hdf.key[:-8].split("/")[-1] for hdf in ras_plan_hdfs]


def get_dict_values(dicts: List[dict], key: Any) -> List[dict]:
    results = []
    for d in dicts:
        if key in d:
            results.append(d[key])
    return results

def intersect_dicts(dicts: List[dict]) -> dict:
    results = copy.deepcopy(dicts[0])
    for d in dicts[1:]:
        for k, v in d.items():
            if k in results:
                if results[k] != v:
                    results[k] = None
    return results


def flatten_dict_list(dict_list):
    flat_list = []
    for d in dict_list:
        for value in d.values():
            if isinstance(value, list):
                flat_list.extend(value)
            else:
                flat_list.append(value)
    return flat_list