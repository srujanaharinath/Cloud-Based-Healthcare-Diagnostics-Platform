import json, falcon, sys,os
import csv
import pandas as pd
import traceback
import io
import matplotlib.pyplot as plt

import random
from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
from falcon_cors import CORS

# Set up BigQuery client
credentials = service_account.Credentials.from_service_account_file(
	'sharp-imprint-420622-895e48b8584e.json'
)
client = bigquery.Client(credentials=credentials, project='sharp-imprint-420622')


class UploadGCPClass:
	def on_post(self, req, resp):	
		data = json.loads(req.stream.read())
		new_row = {}
		for row,value in data.items():
			convrted_key = format_string_reverse(row)
			if convrted_key == 'toxic_look_(typhos)':
				convrted_key = 'toxic_look__typhos'
			new_row[convrted_key] = value
			random_number = random.randint(10000, 99999)
		new_row['patient_id'] = int(random_number)	
		df = pd.DataFrame([new_row])
		df.to_csv('symptoms.csv', index=None)

		# Authenticate with GCP (Google Cloud Platform)
		storage_client = storage.Client.from_service_account_json('sharp-imprint-420622-895e48b8584e.json')

		# Get the bucket
		bucket_name = 'symptoms_incoming'
		bucket = storage_client.get_bucket(bucket_name)

		# Define the name for the file in the bucket
		folder_name = 'incoming/'
		filename = 'symptoms.csv'
		full_filename =  filename

		# Upload the file
		blob = bucket.blob(full_filename)
		blob.upload_from_filename('symptoms.csv', content_type='text/csv')
		resp.status = falcon.HTTP_201  # Created
		resp.body = 'File uploaded successfully'

	def on_get(self, req, resp):
		# Raise a 400 Bad Request status code
		raise falcon.HTTPBadRequest(description='This API supports only POST method')

class BigQueryDataResource:
	def on_get(self, req, resp):
		# Construct BigQuery query
		query = """
			SELECT *
			FROM `sharp-imprint-420622.Disease.inference`
		"""

		# Run the query
		query_job = client.query(query)
		results = query_job.result()

		# Convert results to JSON
		data = []
		for row in results:
			# Extract name, age, and gender from the row
			patient_data = {
				"name": row["name"],
				"age": row["age"],
				"gender": row["gender"],
				"patient_id": row["patient_id"]
			}
			
			# Extract all symptoms and their values from the row
			symptoms = {key: value for key, value in row.items() if key not in ["name", "age", "gender","load_timestamp","patient_id"]}
			
			# Sort symptoms by value in descending order and get the top 3
			top_symptoms = sorted(symptoms.items(), key=lambda item: item[1], reverse=True)[:3]
			
			# Assign the top symptoms to p1, p2, and p3 keys
			for i, (symptom, value) in enumerate(top_symptoms, start=1):
				patient_data[f"Probability{i}"] = f"{symptom} : {round(value * 100, 2)}%"
			
			# Append the patient data to the response
			data.append(patient_data)

		# Set response data
		resp.media = data
		resp.status = falcon.HTTP_200

	def on_post(self, req, resp):
		# Raise a 400 Bad Request status code
		raise falcon.HTTPBadRequest(description='This API supports only GET method')
	


class VisualizeDataResource:
	def on_post(self, req, resp):
		data = json.loads(req.stream.read())
		if 'patient_id' in data:
			if not os.path.exists(str(data['patient_id'])):
				# Create the folder if it doesn't exist
				os.makedirs(str(data['patient_id']))
			credentials = service_account.Credentials.from_service_account_file(
	'sharp-imprint-420622-895e48b8584e.json'
)
			client = bigquery.Client(credentials=credentials, project='sharp-imprint-420622')
			symptoms_data = client.query(f"""SELECT * FROM sharp-imprint-420622.Disease.incomingdata WHERE patient_id = {data['patient_id']} ORDER BY load_timestamp DESC
		LIMIT 1;""" ).result().to_dataframe()
			training_data = client.query(f"""SELECT * FROM sharp-imprint-420622.Disease.training""" ).result().to_dataframe()
			output_data = client.query(f"""SELECT * FROM sharp-imprint-420622.Disease.inference WHERE patient_id = {data['patient_id']} ORDER BY load_timestamp DESC
		LIMIT 1;""" ).result().to_dataframe()
			plot_model_predictions(output_data,data['patient_id'])
			generate_symptom_barchart_from_csv(output_data, training_data, symptoms_data,data['patient_id'])
			files = os.listdir(str(data['patient_id']))
			file_dict = {}
			counter = 1
			for index, file_name in enumerate(files, start=1):
				key = "img{}".format(index)
   				# Assign the filename as the value
				file_path = os.path.join(str(data['patient_id']), file_name)
				# Assign the complete file path as the value
				file_dict[key] = os.path.abspath(file_path)

			resp.media = file_dict
			resp.status = falcon.HTTP_200

			   
			
def format_string_reverse(input_string):
	words = input_string.lower().split()
	for i in range(len(words)):
		if i > 0:
			words[i] = words[i][0].lower() + words[i][1:]
	return '_'.join(words) 

def plot_model_predictions(data,patient_id):
	if 'name' in data.columns:
		data = data.drop(['name', 'age', 'gender','patient_id', 'load_timestamp'], axis=1)
		prob_data = data.iloc[0]
		top_3 = prob_data.nlargest(3)
		others = pd.Series(prob_data.drop(top_3.index).sum(), index=['Other'])
		pie_data = pd.concat([top_3, others])
		colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'] 
		fig, ax = plt.subplots()
		ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
		ax.axis('equal')
		plt.title("Top 3 Model Probability Predictions")
		# plt.show()
		plt.savefig(str(patient_id)+'/probability_pred.png')

def generate_symptom_barchart_from_csv(output_data, training_data, symptoms_data,patient_id):
	training_data = training_data.iloc[:, :-1]
	probabilities = output_data.drop(['name', 'age', 'gender', 'patient_id', 'load_timestamp'], axis=1).iloc[0]
	top_3_diseases = probabilities.nlargest(3).index.tolist()
	relevant_symptoms = training_data[training_data["prognosis"].isin(top_3_diseases)]
	symptom_columns = [col for col in symptoms_data.columns if col not in ['name', 'age', 'gender', 'patient_id', 'load_timestamp']]
	symptoms_series = symptoms_data.iloc[0][symptom_columns]

	for disease in top_3_diseases:
		disease_data = relevant_symptoms[relevant_symptoms['prognosis'] == disease]
		symptom_counts = disease_data[symptom_columns].sum()
		symptoms_to_plot = symptom_counts[symptoms_series == 0].nlargest(5)
		if not symptoms_to_plot.empty:
			symptoms_to_plot = symptoms_to_plot[symptoms_to_plot > 0]
			if symptoms_to_plot.empty:
				continue    
			fig, ax = plt.subplots()
			bars = symptoms_to_plot.plot(kind='bar', ax=ax, color="grey")
			ax.set_title(f'Top 5 Symptoms for {disease}')
			ax.set_ylabel('Count of Symptoms')
			ax.set_xlabel('Symptoms')
			plt.xticks(rotation=45)
			for bar in bars.patches:
				ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(bar.get_height())}', 
						ha='center', va='bottom')
			plt.tight_layout()
			
			# Save the figure
			plt.savefig(str(patient_id)+f'/barchat_{disease}.png')


api = falcon.API()
# Initialize CORS middleware
cors = CORS(allow_all_origins=True, allow_all_headers=True, allow_all_methods=True)  # Allow requests from any origin

# Add CORS middleware to the middleware stack
api.add_middleware(cors.middleware)

api.add_route("/uploadGCP",UploadGCPClass())
api.add_route('/getData', BigQueryDataResource())
api.add_route('/vizualizeData',VisualizeDataResource())
