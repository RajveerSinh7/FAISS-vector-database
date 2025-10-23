from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
demo_data = [
    {"name": "Naruto Uzumaki", "desc": "A ninja with a dream to become hokage."},
    {"name": "Sasuke Uchiha", "desc": "A vengeful ninja from the uchiha clan."},
    {"name": "Luffy", "desc": "A pirate with rubber body and a big dream."},
    {"name": "Goku", "desc": "Saiyaan warrior who protects earth."},
    {"name": "Light yagami", "desc": "A genius who gets a death note."}
]

model = SentenceTransformer('all-MiniLM-L6-v2')

descriptions = [char["desc"] for char in demo_data] #got the "desc" from data

vectors = model.encode(descriptions)

index = faiss.IndexFlatL2(384) #each vector-> 384 dimensions

index.add(np.array(vectors)) #knowledge source ready for matching

query = "who is the strongest ninja?"

query_vector = model.encode([query]) #convert to vector

D,I=index.search(np.array(query_vector),k=2) #D- distance btw both query and similar vector, I-nth desc

for idx in I[0]:
    print(f"your answer is {demo_data[idx]["name"]}")

print(D)
print(I)