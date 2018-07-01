import csv
from datetime import datetime

label_names = {}

def get_label_name(label_id):
	return label_names[label_id]

def load_label_names():
	global labels
	
	with open('vocabulary.csv', encoding='utf8') as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)
		for rows in reader:
			label_names[rows[0]] = rows[3]
	print("[SimpleVideoSearch][{}] Done loading label names from CSV".format(datetime.now()))
