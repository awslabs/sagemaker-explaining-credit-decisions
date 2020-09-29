from pathlib import Path
import json
import numpy as np

from package.data import datasets, schemas


def test_read_label(tmp_path):
    label_folder = Path(tmp_path, 'label')
    label_folder.mkdir(exist_ok=True, parents=True)
    label_file = Path(label_folder, 'label.csv')
    with open(label_file, 'w') as openfile:
        openfile.write('\n'.join(['false', 'true']))

    label_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": [
            {
                "title": "credit__default",
                "type": "boolean"
            }
        ],
        "title": "Credit Application Outcome"
    }
    label_schema_file = Path(tmp_path, 'label.schema.json')
    with open(label_schema_file, 'w') as openfile:
        json.dump(label_schema, openfile)

    schema = schemas.from_json_schema(label_schema_file)
    loaded_data = datasets.read_csv_dataset(label_folder, schema)
    assert loaded_data.shape == (2, 1)
    assert isinstance(loaded_data[0][0], np.bool_)
    assert isinstance(loaded_data[1][0], np.bool_)
