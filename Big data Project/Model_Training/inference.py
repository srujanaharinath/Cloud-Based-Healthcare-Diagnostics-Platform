from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col, coalesce
from pyspark.sql.functions import count, current_timestamp, to_utc_timestamp
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

spark = SparkSession.builder \
    .master("local") \
    .appName('inference') \
    .getOrCreate()
	
model_path = "gs://big_data_diseases/models/rf_model"
trained_model = PipelineModel.load(model_path)
loaded_model = StringIndexerModel.load("gs://big_data_diseases/models/encoder")
file_path = 'gs://big_data_diseases/incoming/symptoms.csv'
df_test = spark.read.csv(file_path, header=True, inferSchema=True)
	
x_test = df_test.drop("name","age","gender","patient_id")
x_test.show()

prediction = trained_model.transform(x_test)
curr_time = current_timestamp()
predicted_values = prediction.select("probability").collect()[0][0]
predicted_values = [float(round(value,4)) for value in predicted_values]
predicted_values.append(df_test.select("name").collect()[0][0])
predicted_values.append(df_test.select("age").collect()[0][0])
predicted_values.append(df_test.select("gender").collect()[0][0])
predicted_values.append(df_test.select("patient_id").collect()[0][0])

labels = loaded_model.labels
schema = StructType([StructField(name, FloatType(), True) for name in labels])
additional_fields = [
    StructField('name', StringType(), True),
    StructField('age', IntegerType(), True),
    StructField('gender', StringType(), True),
    StructField('patient_id', IntegerType(), True)
]
schema = StructType(schema.fields + additional_fields)
op_df = spark.createDataFrame([predicted_values],schema)
op_df = op_df.withColumn("load_timestamp", curr_time)
df_test = df_test.withColumn("load_timestamp", curr_time)

df_test.write.format("bigquery") \
.mode("append") \
.option("temporaryGcsBucket", "temp_bigquery_disease") \
.option("table", "sharp-imprint-420622.Disease.incomingdata") \
.save()	

op_df.write.format("bigquery") \
.mode("append") \
.option("temporaryGcsBucket", "temp_bigquery_disease") \
.option("table", "sharp-imprint-420622.Disease.inference") \
.save()

spark.stop()