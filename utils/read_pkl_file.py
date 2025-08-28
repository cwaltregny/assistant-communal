import pickle


with open('vector_store_enhanced/mappings.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data)

with open('vector_store_enhanced/chunks.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data)
