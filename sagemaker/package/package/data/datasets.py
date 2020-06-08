"""Used to split original dataset into three denormalized tables: credits,
people and contacts."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def clear_datasets(folder: Path):
    if folder.exists():
        assert folder.is_dir()
        shutil.rmtree(folder)
        folder.mkdir()


JSON_TO_NUMPY_TYPES = {
    "string": np.string_,
    "number": np.float_,
    "integer": np.int_,
    "boolean": np.bool_,
}


def read_csv_dataset(folder, schema):
    names = schema.item_titles
    types = {
        n: JSON_TO_NUMPY_TYPES[t] for n, t in schema.item_types_dict.items()
    }
    filepaths = Path(folder).glob("*")
    if type(filepaths) in set(["str", Path]):
        filepaths = [filepaths]
    dfs = []
    for filepath in filepaths:
        dfs.append(
            pd.read_csv(
                filepath, dtype=types, names=names, index_col=None, header=None
            )
        )
    df = pd.concat(dfs, axis=0, ignore_index=True)
    ndarray = df.to_numpy()
    return ndarray


def read_json_dataset(folder, schema):
    filepaths = Path(folder).glob("*")
    if type(filepaths) in set(["str", Path]):
        filepaths = [filepaths]
    records = []
    for filepath in filepaths:
        with open(filepath) as lines:
            for line in lines:
                if line:
                    record = json.loads(line)
                    record = schema.transform(record)
                    records.append(record)
    ndarray = np.array(records)
    return ndarray
