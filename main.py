from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from numpy.linalg import norm
from numpy import dot
from sentence_transformers import SentenceTransformer
import os

script_dir = os.path.dirname(__file__)

model = SentenceTransformer('distiluse-base-multilingual-cased')

rel_path = "resources/Mapping SEO.xlsx"
abs_file_path = os.path.join(script_dir, rel_path)
df = pd.read_excel(abs_file_path)
df = df.loc[df["page"].str.contains("fiche")]
df = df.reset_index(drop=True)

rel_path = "resources/descriptions_encodings.p"
abs_file_path = os.path.join(script_dir, rel_path)
with open(abs_file_path, 'rb') as f:
    descriptions_encodings = pickle.load(f)

rel_path = "resources/titles_encodings.p"
abs_file_path = os.path.join(script_dir, rel_path)
with open(abs_file_path, 'rb') as f:
    titles_encodings = pickle.load(f)

def find_closest_sheets(question, descriptions_encodings, titles_encodings, url_list, nb_results=1) :

  q_enc = model.encode(question)

  sims = []

  nb_encodings = len(descriptions_encodings)

  for i in range(nb_encodings) :

    title_sim = dot(q_enc, titles_encodings[i])/(norm(q_enc)*norm(titles_encodings[i]))
    desc_sim = dot(q_enc, descriptions_encodings[i])/(norm(q_enc)*norm(descriptions_encodings[i]))

    sims.append(max(title_sim,desc_sim))

  closest_idxs = np.argsort(sims)[-nb_results:]

  results = [ (url_list.iloc[idx]["page"],sims[idx]) for idx in closest_idxs ]

  return np.flipud(results)

app = FastAPI()

@app.post("/question/")
async def search_engine(question: str) :
    return find_closest_sheets(question, descriptions_encodings, titles_encodings, df, 3)