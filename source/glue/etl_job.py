"""Used as part of AWS Glue Job to transform credits data."""
# see https://github.com/awslabs/aws-glue-libs/tree/master/awsglue for details
from awsglue.context import GlueContext  # pylint: disable=import-error
from awsglue.job import Job  # pylint: disable=import-error
from awsglue.utils import getResolvedOptions  # pylint: disable=import-error
import boto3
from pyspark import SparkConf, SparkContext
import sys


def setup_contexts():
    spark_conf = (
        SparkConf()
        .set(
            "spark.hadoop.hive.metastore.client.factory.class",
            "com.amazonaws.glue.catalog.metastore"
            + ".AWSGlueDataCatalogHiveClientFactory",
        )
        .set("spark.sql.catalogImplementation", "hive")
    )
    sc = SparkContext(conf=spark_conf)
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    return glueContext, spark


def create_job(glueContext, args):
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)
    return job


def get_database_location(database_name, region):
    glue = boto3.client("glue", region_name=region)
    database = glue.get_database(Name=database_name)
    assert "Database" in database
    assert (
        "LocationUri" in database["Database"]
    ), "Must set LocationUri on database."
    location_uri = database["Database"]["LocationUri"]
    s3_bucket, s3_prefix = parse_database_location(location_uri)
    return s3_bucket, s3_prefix


def parse_database_location(location_uri):
    assert (
        location_uri[:5] == "s3://"
    ), "database location expected to be an s3 uri."
    s3_bucket = location_uri[5:].split("/")[0]
    s3_prefix = "/".join(location_uri[5:].split("/")[1:]).strip("/")
    return s3_bucket, s3_prefix


def load_credits():
    credits = glueContext.create_dynamic_frame.from_catalog(
        database=GLUE_DATABASE, table_name="credits"
    )
    return credits


def transform_credits(credits):
    credits = credits.toDF()  # convert from dynamic frame to data frame
    return credits


def load_people():
    people = glueContext.create_dynamic_frame.from_catalog(
        database=GLUE_DATABASE, table_name="people"
    )
    return people


def relationalize_people(people):
    # flatten and unnest attributes
    dfc = people.relationalize("people", GLUE_TEMP)
    # `dfc` is a 'dynamic frame collection' with two frames
    # `people` contains the flattened attributes
    people = dfc.select("people").toDF()
    people = people.toDF(
        *[c.replace(".", "__") for c in people.columns]
    )  # avoid issues with '.'
    dependents = dfc.select("people_dependents").toDF()
    return people, dependents


def fill_finance_accounts(people):
    people = people.na.fill({
        "finance__accounts__checking__balance": "no_account"})
    people = people.na.fill({
        "finance__accounts__savings__balance": "no_account"})
    return people


def join_dependents(people, dependents):
    # create a new feature called `num_dependents`
    people.createOrReplaceTempView("people")
    dependents.createOrReplaceTempView("dependents")
    num_dependents = spark.sql(
        """
        SELECT id, count(id) as num_dependents
        FROM dependents
        GROUP BY id
    """
    )
    # add this feature to `people` table
    num_dependents.createOrReplaceTempView("num_dependents")
    people = spark.sql(
        """
        SELECT *
        FROM people
        LEFT JOIN num_dependents ON people.dependents=num_dependents.id
    """
    )
    # clean up columns
    people = people.drop(*["dependents", "id"])
    people = people.withColumnRenamed(
        "num_dependents", "personal__num_dependents"
    )
    return people


def transform_people(people):
    people, dependents = relationalize_people(people)
    people = fill_finance_accounts(people)
    people = join_dependents(people, dependents)
    # `people_dependents` contains the previously nested array of dependents
    return people


def load_contacts():
    contacts = glueContext.create_dynamic_frame.from_catalog(
        database=GLUE_DATABASE, table_name="contacts"
    )
    return contacts


def transform_contacts(contacts):
    contacts.toDF().createOrReplaceTempView("contacts")
    contacts = spark.sql(
        """
        SELECT person_id, count(contact_id) as num_telephones
        FROM contacts
        WHERE type = 'telephone'
        GROUP BY person_id
    """
    )
    return contacts


def join_data(credits, people, contacts):
    # rename before join to avoid column name clash
    credits = credits.toDF(*["credit__" + n for n in credits.columns])
    contacts = contacts.toDF(*["contact__" + n for n in contacts.columns])
    # add temp views so can access from spark sql
    credits.createOrReplaceTempView("credits")
    people.createOrReplaceTempView("people")
    contacts.createOrReplaceTempView("contacts")
    # join all tables on `person_id`
    data = spark.sql(
        """
        SELECT *
        FROM credits
        JOIN people ON credits.credit__person_id = people.person_id
        LEFT JOIN contacts ON credits.credit__person_id =
                              contacts.contact__person_id
    """
    )
    return data


def transform_data(data):
    # clean up columns
    data = data.selectExpr(
        "*", "isnotnull(contact__num_telephones) as contact__has_telephone"
    )
    data = data.drop("contact__num_telephones")
    data = data.drop("credit__credit_id")
    data = data.drop(*["credit__person_id", "person_id", "contact__person_id"])
    data = data.drop(*["employment__title", "personal__name"])
    data = data.select(["`{}`".format(c) for c in sorted(data.columns)])
    return data


def split_train_test(data, train_split=0.8):
    data = data.cache()
    data_train, data_test = data.randomSplit([train_split, 1 - train_split])
    return data_train, data_test


def split_data_label(data_train, data_test):
    label_train = data_train.select("credit__default")
    data_train = data_train.drop("credit__default")
    label_test = data_test.select("credit__default")
    data_test = data_test.drop("credit__default")
    return data_train, label_train, data_test, label_test


def delete_table_data(table_name):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(S3_BUCKET)  # pylint: disable=no-member
    bucket.objects.filter(
        Prefix="{}/{}/".format(S3_PREFIX, table_name)
    ).delete()


def delete_table(table_name):
    spark.sql("DROP TABLE IF EXISTS `{}`".format(table_name))


def create_table(df, table_name):
    df.registerTempTable("df_tmp")
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS {}
        USING HIVE
        OPTIONS(
            fileFormat 'textfile',
            fieldDelim ',')
        AS SELECT * FROM df_tmp
    """.format(
            table_name
        )
    )


def save(df, table_name, overwrite=False):
    if overwrite:
        delete_table(table_name)
        delete_table_data(table_name)
    create_table(df, table_name)


def main():
    # set default database to avoid having to reference each time
    spark.sql("USE `{}`".format(GLUE_DATABASE))

    # load and transform each of the three tables
    credits = load_credits()
    credits = transform_credits(credits)
    people = load_people()
    people = transform_people(people)
    contacts = load_contacts()
    contacts = transform_contacts(contacts)

    # join all three tables into one
    data = join_data(credits, people, contacts)
    data = transform_data(data)

    # split data into train and test sets (and separate label)
    data = data.coalesce(1)
    data_train, data_test = split_train_test(data, train_split=0.8)
    data_train, label_train, data_test, label_test = split_data_label(
        data_train, data_test
    )

    # save all sets
    save(data_train, "data_train", overwrite=True)
    save(label_train, "label_train", overwrite=True)
    save(data_test, "data_test", overwrite=True)
    save(label_test, "label_test", overwrite=True)


if __name__ == "__main__":
    valid_params = ["JOB_NAME", "TempDir"]  # base params
    valid_params += ["GLUE_DATABASE", "GLUE_REGION"]  # custom params
    args = getResolvedOptions(sys.argv, valid_params)
    glueContext, spark = setup_contexts()
    # create and initialize job
    job = create_job(glueContext, args)
    # define global constants
    GLUE_DATABASE = args["GLUE_DATABASE"]
    GLUE_REGION = args["GLUE_REGION"]
    S3_BUCKET, S3_PREFIX = get_database_location(GLUE_DATABASE, GLUE_REGION)
    GLUE_TEMP = args["TempDir"]
    # define job in `main` and then commit.
    main()
    job.commit()
