import json
from pathlib import Path
from jsonschema import validate, Draft4Validator
import copy
import numpy as np


def from_json_schema(filepath):
    with open(filepath, "r") as openfile:
        json_schema = json.load(openfile)
    return Schema(json_schema)


class Schema:
    def __init__(self, json_schema):
        self._validate_schema(json_schema)
        self._schema = json_schema
        self._object_schema = self._to_object_schema(json_schema)

    def validate(self, instance):
        if isinstance(instance, list):
            validate(instance, self._schema)
        elif isinstance(instance, dict):
            validate(instance, self._object_schema)
        else:
            raise TypeError('instance should be list or dict.')

    @staticmethod
    def _validate_schema(schema):
        Draft4Validator.check_schema(schema)

    @property
    def title(self):
        return self._schema["title"]

    @title.setter
    def title(self, title):
        self._schema["title"] = title

    @property
    def description(self):
        return self._schema["description"]

    @description.setter
    def description(self, description):
        self._schema["description"] = description

    @property
    def item_descriptions_dict(self):
        item_descriptions_dict = {}
        for item in self._schema["items"]:
            if 'description' in item:
                item_descriptions_dict[item['title']] = item['description']
        return item_descriptions_dict

    @item_descriptions_dict.setter
    def item_descriptions_dict(self, item_descriptions_dict: dict):
        for item in self._schema["items"]:
            if item["title"] in item_descriptions_dict:
                item["description"] = item_descriptions_dict[item["title"]]
        self._validate_schema(self._schema)

    @property
    def items(self):
        return self._schema["items"]

    @property
    def item_titles(self):
        return [item["title"] for item in self._schema["items"]]

    @property
    def item_types(self):
        return [item["type"] for item in self._schema["items"]]

    @property
    def item_descriptions(self):
        item_descriptions = []
        for item in self._schema["items"]:
            if 'description' in item:
                item_descriptions.append(item['description'])
            else:
                item_descriptions.append(None)
        return item_descriptions

    @property
    def item_types_dict(self):
        return {item["title"]: item["type"] for item in self._schema["items"]}

    @staticmethod
    def _to_object_schema(schema):
        object_schema = copy.deepcopy(schema)
        properties = {}
        for item in schema["items"]:
            item_title = item["title"]
            properties[item_title] = {
                "type": item["type"]
            }
        object_schema = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "minProperties": len(schema["items"])
        }
        return object_schema

    def save(self, filepath):
        assert isinstance(filepath, Path)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as openfile:
            json.dump(self._schema, openfile, indent=4)

    def transform(self, instance):
        if isinstance(instance, dict):
            instance = [instance[title] for title in self.item_titles]
        return np.array(instance, dtype=np.object)
