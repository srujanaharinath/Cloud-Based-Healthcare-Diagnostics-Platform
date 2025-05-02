from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col, coalesce
from pyspark.sql.functions import count

spark = SparkSession.builder \
    .master("yarn") \
    .appName('Jupyter BigQuery Storage')\
    .getOrCreate()
	
table_full_name = "sharp-imprint-420622.Disease.training"
df = spark.read.format("bigquery").option("table",table_full_name).load()

df = df.drop("string_field_133")
indexer = StringIndexer(inputCol="prognosis", outputCol="label")
indexer_model = indexer.fit(df)
df_encoded = indexer_model.transform(df)
indexer_model.save("gs://big_data_diseases/models/encoder")
df_encoded = df_encoded.drop("prognosis")
assembler = VectorAssembler(inputCols=[col for col in df_encoded.columns if col != 'label'], outputCol="features")
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(df_encoded)
model.save("gs://big_data_diseases/models/rf_model")
spark.stop()