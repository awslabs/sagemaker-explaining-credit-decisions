"""Used to split original dataset into three denormalized tables: credits,
people and contacts."""
import json
import random
import io
import csv
from pathlib import Path
import boto3
from faker import Faker


def download_dataset(s3_bucket, s3_key):
    s3_resource = boto3.resource("s3")
    obj = s3_resource.Object(  # pylint: disable=no-member
        s3_bucket, s3_key
    )
    string = obj.get()["Body"].read().decode("utf-8")
    return string


def parse_statlog(string):
    """
    Statlog files are separated by variable number of spaces.
    """
    lines = string.split("\n")
    lines = list(
        filter(lambda line: len(line.strip()) > 0, lines)
    )  # filter out empty lines # noqa: E501
    for line_idx, line in enumerate(lines):
        line = line.split(" ")
        line = list(
            filter(lambda e: len(e) > 0, line)
        )  # filter out empty elements # noqa: E501
        lines[line_idx] = line
    return lines


def escape_rows(rows):
    """
    Checks all rows for string elements, and replaces '\n' with '\\n'.
    """
    output = []
    for row in rows:
        output_row = []
        for element in row:
            if type(element) == str:
                escaped_element = element.replace("\n", "\\n")
                output_row.append(escaped_element)
            else:
                output_row.append(element)
        output.append(output_row)
    return output


def create_csv_str(header, rows):
    """
    Creates a string representing a csv table (with header row).
    All string columns are quoted and new line characters contained within
    them are escaped.
    """
    str_io = io.StringIO()
    writer = csv.writer(str_io, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(header)
    rows = escape_rows(rows)
    writer.writerows(rows)
    csv_str = str_io.getvalue()
    return csv_str


def create_json_lines_str(rows):
    return "\n".join([json.dumps(row) for row in rows])  # json lines format


class PeopleDataset:
    """
    Creates a dataset of people, where each row is a nested object.
    Serializes to a JSON Lines file.
    """

    def __init__(self, input_lines, person_ids):
        self._faker = Faker()
        output_rows = []
        for input_line, person_id in zip(input_lines, person_ids):
            output_row = {}
            output_row["person_id"] = person_id
            output_row["finance"] = self._finance(input_line)
            output_row["personal"] = self._personal(input_line)
            output_row["dependents"] = self._dependents(input_line)
            output_row["employment"] = self._employment(input_line)
            output_row["residence"] = self._residence(input_line)
            output_rows.append(output_row)
        self._rows = output_rows

    def _finance(self, input_line):
        output = {}
        checking_account_balance = self._checking_account_balance(input_line)
        savings_account_balance = self._savings_account_balance(input_line)
        if checking_account_balance or savings_account_balance:
            output["accounts"] = {}
            if checking_account_balance:
                output["accounts"]["checking"] = {
                    "balance": checking_account_balance
                }
            if savings_account_balance:
                output["accounts"]["savings"] = {
                    "balance": savings_account_balance
                }
        output["repayment_history"] = self._repayment_history(input_line)
        output["credits"] = {}
        output["credits"]["this_bank"] = self._credits_this_bank(input_line)
        output["credits"]["other_banks"] = self._credits_other_banks(
            input_line
        )  # noqa
        output["credits"]["other_stores"] = self._credits_other_stores(
            input_line
        )
        output["other_assets"] = self._other_assets(input_line)
        return output

    def _personal(self, input_line):
        output = {}
        output["age"] = self._age(input_line)
        output["gender"] = self._gender(input_line)
        output["relationship_status"] = self._relationship_status(input_line)
        output["name"] = self._name(gender=output["gender"])
        return output

    def _dependents(self, input_line):
        dependents = []
        num_dependents = int(input_line[17])
        for _ in range(num_dependents):
            dependent = {}
            dependent["gender"] = random.choice(["male", "female"])
            if dependent["gender"] == "male":
                dependent["name"] = self._faker.name_male()
            elif dependent["gender"] == "female":
                dependent["name"] = self._faker.name_female()
            dependents.append(dependent)
        return dependents

    def _employment(self, input_line):
        output = {}
        output["type"] = self._employment_type(input_line)
        output["title"] = self._faker.job()
        output["duration"] = self._employment_duration(input_line)
        output["permit"] = self._employment_permit(input_line)
        return output

    def _residence(self, input_line):
        output = {}
        output["type"] = self._residence_type(input_line)
        output["duration"] = self._residence_duration(input_line)
        return output

    def _residence_type(self, input_line):
        mapping = {"A151": "rent", "A152": "own", "A153": "free"}
        residence_type = mapping[input_line[14]]
        return residence_type

    def _residence_duration(self, input_line):
        residence_duration = int(input_line[10])
        return residence_duration

    def _employment_type(self, input_line):
        mapping = {
            "A171": "unemployed",
            "A172": "labourer",
            "A173": "professional",
            "A174": "management",
        }
        employment_type = mapping[input_line[16]]
        return employment_type

    def _employment_duration(self, input_line):
        mapping = {
            "A71": 0,
            "A72": 1,
            "A73": random.randint(2, 4),
            "A74": random.randint(5, 7),
            "A75": random.randint(8, 16),
        }
        employment_duration = mapping[input_line[6]]
        return employment_duration

    def _employment_permit(self, input_line):
        mapping = {"A201": "foreign", "A202": "domestic"}
        employment_permit = mapping[input_line[19]]
        return employment_permit

    def _age(self, input_line):
        age = int(input_line[12])
        return age

    def _gender(self, input_line):
        mapping = {
            "A91": "male",
            "A92": "female",
            "A93": "male",
            "A94": "male",
            "A95": "female",
        }
        gender = mapping[input_line[8]]
        return gender

    def _relationship_status(self, input_line):
        mapping = {
            "A91": "married",
            "A92": "married",
            "A93": "single",
            "A94": "married",
            "A95": "single",
        }
        relationship_status = mapping[input_line[8]]
        return relationship_status

    def _name(self, gender=None):
        if gender == "male":
            name = self._faker.name_male()
        elif gender == "female":
            name = self._faker.name_female()
        else:
            name = self._faker.name()
        return name

    def _checking_account_balance(self, input_line):
        mapping = {"A11": "negative", "A12": "low", "A13": "high", "A14": None}
        checking_account_balance = mapping[input_line[0]]
        return checking_account_balance

    def _savings_account_balance(self, input_line):
        mapping = {
            "A61": "low",
            "A62": "medium",
            "A63": "high",
            "A64": "very_high",
            "A65": None,
        }
        savings_account_balance = mapping[input_line[5]]
        return savings_account_balance

    def _repayment_history(self, input_line):
        mapping = {
            "A30": "good",
            "A31": "good",
            "A32": "good",
            "A33": "poor",
            "A34": "very_poor",
        }
        repayment_history = mapping[input_line[2]]
        return repayment_history

    def _credits_this_bank(self, input_line):
        credits_this_bank = int(input_line[15])
        return credits_this_bank

    def _credits_other_banks(self, input_line):
        mapping = {"A141": random.randint(1, 2), "A142": 0, "A143": 0}
        credits_other_banks = mapping[input_line[13]]
        return credits_other_banks

    def _credits_other_stores(self, input_line):
        mapping = {"A141": 0, "A142": random.randint(1, 2), "A143": 0}
        credits_other_stores = mapping[input_line[13]]
        return credits_other_stores

    def _other_assets(self, input_line):
        mapping = {
            "A121": "real_estate",
            "A122": "life_insurance",
            "A123": "car_or_other",
            "A124": "none",
        }
        other_assets = mapping[input_line[11]]
        return other_assets

    def as_json_lines_str(self):
        return create_json_lines_str(self._rows)

    def save(self, filepath):
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as openfile:
            openfile.write(self.as_json_lines_str())


class ContactsDataset:
    """
    Creates a dataset of contacts.
    Serializes to a CSV file.
    """

    def __init__(self, input_lines, person_ids):
        self._faker = Faker()
        output_rows = []
        for input_line, person_id in zip(input_lines, person_ids):
            has_telephone = self._has_telephone(input_line)
            if has_telephone:
                for _ in range(random.randint(1, 2)):
                    contact_id = self._faker.uuid4()[:8]
                    telephone = self._faker.phone_number()
                    output_row = [
                        contact_id,
                        person_id,
                        "telephone",
                        telephone,
                    ]
                    output_rows.append(output_row)
            for _ in range(random.randint(0, 2)):
                contact_id = self._faker.uuid4()[:8]
                email = self._faker.email()
                output_row = [contact_id, person_id, "email", email]
                output_rows.append(output_row)
            for _ in range(random.randint(0, 1)):
                contact_id = self._faker.uuid4()[:8]
                address = self._faker.address()
                output_row = [contact_id, person_id, "address", address]
                output_rows.append(output_row)
        self._rows = output_rows

    @property
    def _header(self):
        header = ["contact_id", "person_id", "type", "value"]
        return header

    def _has_telephone(self, input_line):
        mapping = {"A191": False, "A192": True}
        has_telephone = mapping[input_line[18]]
        return has_telephone

    def as_csv_str(self):
        csv_str = create_csv_str(self._header, self._rows)
        return csv_str

    def save(self, filepath):
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as openfile:
            openfile.write(self.as_csv_str())


class CreditsDataset:
    """
    Creates a dataset of credit applications.
    Serializes to a CSV file.
    """

    def __init__(self, input_lines, person_ids):
        self._faker = Faker()
        output_rows = []
        for input_line, person_id in zip(input_lines, person_ids):
            credit_id = self._faker.uuid4()[:8]
            amount = self._amount(input_line)
            duration = self._duration(input_line)
            purpose = self._purpose(input_line)
            installment_rate = self._installment_rate(input_line)
            guarantor = self._guarantor(input_line)
            coapplicant = self._coapplicant(input_line)
            default = self._default(input_line)
            output_row = [
                credit_id,
                person_id,
                amount,
                duration,
                purpose,
                installment_rate,
                guarantor,
                coapplicant,
                default,
            ]
            output_rows.append(output_row)
        self._rows = output_rows

    @property
    def _header(self):
        header = [
            "credit_id",
            "person_id",
            "amount",
            "duration",
            "purpose",
            "installment_rate",
            "guarantor",
            "coapplicant",
            "default",
        ]
        return header

    def _amount(self, input_line):
        amount = int(input_line[4])
        return amount

    def _purpose(self, input_line):
        mapping = {
            "A40": "new_car",
            "A41": "used_car",
            "A42": "furniture",
            "A43": "electronics",
            "A44": "appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation",
            "A48": "re-training",
            "A49": "business",
            "A410": "other",
        }
        purpose = mapping[input_line[3]]
        return purpose

    def _duration(self, input_line):
        duration = int(input_line[1])
        return duration

    def _installment_rate(self, input_line):
        installment_rate = int(input_line[7])
        return installment_rate

    def _guarantor(self, input_line):
        mapping = {"A101": 0, "A102": 0, "A103": 1}
        guarantor = mapping[input_line[9]]
        return guarantor

    def _coapplicant(self, input_line):
        mapping = {"A101": 0, "A102": 1, "A103": 0}
        coapplicant = mapping[input_line[9]]
        return coapplicant

    def _default(self, input_line):
        default = bool(int(input_line[20]) - 1)
        return default

    def as_csv_str(self):
        csv_str = create_csv_str(self._header, self._rows)
        return csv_str

    def save(self, filepath):
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as openfile:
            openfile.write(self.as_csv_str())


def generate_datasets(s3_bucket, s3_key, folder):
    # download and parse german dataset
    string = download_dataset(s3_bucket, s3_key)
    lines = parse_statlog(string)
    # create fake person_ids that will be used to join data back together
    faker = Faker()
    person_ids = [faker.uuid4()[:8] for l in lines]
    # create credits dataset
    credits_filepath = Path(folder, "credits", "part-r-00000")
    CreditsDataset(lines, person_ids).save(credits_filepath)
    # create people dataset
    people_filepath = Path(folder, "people", "part-r-00000")
    PeopleDataset(lines, person_ids).save(people_filepath)
    # create contacts dataset
    contacts_filepath = Path(folder, "contacts", "part-r-00000")
    ContactsDataset(lines, person_ids).save(contacts_filepath)
    return credits_filepath, people_filepath, contacts_filepath
