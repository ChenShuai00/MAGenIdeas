import os
import json
import numpy as np
import faiss
import ollama
import ast  # For parsing Python literals
from tqdm import tqdm

def create_paper_embeddings(paper_folder_path, output_index_path):
    """
    Vectorize papers and store them in a FAISS index
    
    Args:
        paper_folder_path: Path to folder containing paper text files
        output_index_path: Path to save the FAISS index
    """
    # Initialize Ollama client
    client = ollama.Client(host='http://127.0.0.1:11434')
    
    # Store embeddings and paper info
    paper_embeddings = []
    paper_info = []
    
    # Get all files in folder
    files = os.listdir(paper_folder_path)
    
    # Process files with progress bar
    for filename in tqdm(files, desc="Processing papers"):
        if not filename.endswith('.txt'):
            print(f"Skipping non-txt file: {filename}")
            continue
            
        file_path = os.path.join(paper_folder_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse Python dict format
                paper_data = ast.literal_eval(content)
            
            # Combine title and abstract for embedding
            text_to_embed = f"Title: {paper_data['title']}\nAbstract: {paper_data['abstract']}"
            
            # Get embedding vector
            embedding = client.embeddings(
                model="mxbai-embed-large",
                prompt=text_to_embed
            )
            
            # Store vector and paper info
            paper_embeddings.append(embedding['embedding'])
            paper_info.append(paper_data)
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            print(f"File path: {file_path}")
    
    # Check if any papers were processed
    if not paper_embeddings:
        print(f"Warning: No papers processed in folder {paper_folder_path}")
        return
    
    # Convert to numpy array
    embeddings_array = np.array(paper_embeddings).astype('float32')
    
    # Create FAISS index
    dimension = len(paper_embeddings[0])  # Vector dimension
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    
    # Add vectors to index
    index.add(embeddings_array)
    
    # Save index
    faiss.write_index(index, output_index_path)
    
    # Save paper info mapping with error handling
    try:
        with open(output_index_path + '_info.json', 'w', encoding='utf-8') as f:
            # ensure_ascii=False for Chinese support
            json_str = json.dumps(paper_info, ensure_ascii=False, indent=2)
            # Clean surrogate characters
            clean_str = json_str.encode('utf-16', 'surrogatepass').decode('utf-16')
            f.write(clean_str)
    except Exception as e:
        print(f"Error saving paper info: {str(e)}")
        # Fallback with more lenient encoding
        with open(output_index_path + '_info.json', 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(paper_info, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Successfully processed {len(paper_embeddings)} papers")
    print(f"Index saved to: {output_index_path}")
    print(f"Paper info saved to: {output_index_path}_info.json")

if __name__ == "__main__":
    acl2024_long_folder = "/home/shuaichen/code/virtual_scientists/dataset/data/acl2024_long"
    # Get all subfolder paths
    aim_paper_folders = []
    ref_paper_folders = []
    for root, dirs, files in os.walk(acl2024_long_folder):
        for dir in dirs:
            aim_paper_folders.append(os.path.join(root, dir, "aim_paper"))
            ref_paper_folders.append(os.path.join(root, dir, "ref_papers"))
        break

    # Process aim papers
    for aim_paper_folder in aim_paper_folders:
        create_paper_embeddings(aim_paper_folder, aim_paper_folder + "_index.index")
    
    # Process reference papers
    for ref_paper_folder in ref_paper_folders:
        target_path = os.path.join(*ref_paper_folder.split("/")[:-1])
        if os.path.exists(f"/{target_path}/ref_papers_index.index"):
            print(f"Skipping existing index: /{target_path}/ref_papers_index.index")
            continue
        create_paper_embeddings(ref_paper_folder, ref_paper_folder + "_index.index")
