import numpy as np
import pytest
from jsonschema.exceptions import ValidationError

from package.data import schemas


JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": [
        {
            "title": "contact_has_telephone",
            "type": "boolean",
            "description": "Customer has a registered telephone number."
        },
        {
            "title": "credit_amount",
            "type": "integer",
            "description": "Amount of money requested as part of credit application (in EUR)."  # noqa
        },
        {
            "title": "credit_purpose",
            "type": "string",
            "description": "Customer's reason for requiring credit."
        }
    ],
    "title": "Credit Application",
    "description": "An array of items used to describe a credit application."
}


def test_create_schema():
    schema = schemas.Schema(JSON_SCHEMA)
    assert isinstance(schema, schemas.Schema)


def test_validate_list():
    schema = schemas.Schema(JSON_SCHEMA)
    data = [True, 1, 'test']
    schema.validate(data)


def test_validate_list_wrong_types():
    schema = schemas.Schema(JSON_SCHEMA)
    data = ['test', 1, True]
    with pytest.raises(ValidationError):
        schema.validate(data)


def test_validate_list_too_long():
    schema = schemas.Schema(JSON_SCHEMA)
    data = [True, 1, 'test', 123.456]
    with pytest.raises(ValidationError):
        schema.validate(data)


def test_validate_list_too_short():
    schema = schemas.Schema(JSON_SCHEMA)
    data = [True, 1]
    with pytest.raises(ValidationError):
        schema.validate(data)


def test_validate_dict():
    schema = schemas.Schema(JSON_SCHEMA)
    data = {
        "contact_has_telephone": True,
        "credit_amount": 1,
        "credit_purpose": "test"
    }
    schema.validate(data)


def test_validate_dict_wrong_types():
    schema = schemas.Schema(JSON_SCHEMA)
    data = {
        "contact_has_telephone": "test",
        "credit_amount": 1,
        "credit_purpose": True
    }
    with pytest.raises(ValidationError):
        schema.validate(data)


def test_validate_dict_extra_param():
    schema = schemas.Schema(JSON_SCHEMA)
    data = {
        "contact_has_telephone": "test",
        "credit_amount": 1,
        "credit_purpose": True,
        "extra_param": 123.456
    }
    with pytest.raises(ValidationError):
        schema.validate(data)


def test_validate_dict_missing_param():
    schema = schemas.Schema(JSON_SCHEMA)
    data = {
        "contact_has_telephone": True,
        "credit_amount": 1
    }
    with pytest.raises(ValidationError):
        schema.validate(data)


def test_transform_list():
    schema = schemas.Schema(JSON_SCHEMA)
    data = [True, 1, 'test']
    data = schema.transform(data)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.object
    assert data.shape == (3,)


def test_transform_dict():
    schema = schemas.Schema(JSON_SCHEMA)
    data = {
        "credit_purpose": "test",
        "credit_amount": 1,
        "contact_has_telephone": True
    }
    data = schema.transform(data)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.object
    assert data.shape == (3,)
    assert data[0] == True  # noqa
    assert data[1] == 1
    assert data[2] == "test"
