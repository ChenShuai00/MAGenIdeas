from pymongo import MongoClient
import os
import json
from typing import Dict, List, Any
from pathlib import Path

def get_mongodb_connection() -> tuple:
    """
    Get MongoDB collections for papers and authors
    
    Returns:
        Tuple of (acl2024_long collection, author_info collection)
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client['scholar']
    return db['acl2024_long'], db['author_info']

def validate_work(work: Dict) -> bool:
    """
    Validate if a paper meets our criteria
    
    Args:
        work: Paper document from MongoDB
        
    Returns:
        bool: True if paper meets all criteria
    """
    return (work.get('citationCount', 0) > 10 and 
            work.get('abstract') is not None and
            work.get('authors') is not None and
            work.get('references') is not None and
            len(work.get('references', [])) > 20)

def process_author(author_info: Any, author_id: str, base_path: Path, author_idx: int) -> None:
    """
    Process and save author information
    
    Args:
        author_info: MongoDB author document
        author_id: Author ID
        base_path: Base directory path
        author_idx: Author index number
    """
    name = f"Scientist{author_idx}"
    affiliations = author_info.get('institutions', [])
    topics = author_info.get('topics', [])
    paperCount = author_info.get('paperCount', 0)
    citationCount = author_info.get('citationCount', 0)
    
    # Create author description
    author_text = (
        f"Your name is {name}, "
        f"you belong to following affiliations: {affiliations}, "
        f"you have researched on following topics: {topics}, "
        f"you have published {paperCount} papers, "
        f"you have {citationCount} citations."
    )
    
    # Save author info
    author_dir = base_path / f"authors/author{author_idx}"
    author_dir.mkdir(parents=True, exist_ok=True)
    
    with open(author_dir / f"author{author_idx}.txt", "w", encoding="utf-8") as f:
        f.write(author_text)
    
    # Save author papers
    papers_dir = author_dir / "papers"
    papers_dir.mkdir(exist_ok=True)
    
    for paper_idx, paper in enumerate(author_info.get('papers', [])):
        paper_data = {
            "id": paper.get('paperId'),
            "title": paper.get('title'),    
            "abstract": paper.get('abstract'),
        }
        with open(papers_dir / f"author_paper_{paper_idx}.txt", "w", encoding="utf-8") as f:
            json.dump(paper_data, f, indent=4)

def process_dataset(
    acl2024_long: Any, 
    author_info: Any, 
    base_path: str = "/home/shuaichen/code/virtual_scientists/dataset/data/acl2024_long"
) -> None:
    """
    Process dataset and save to organized folder structure
    
    Args:
        acl2024_long: MongoDB collection of papers
        author_info: MongoDB collection of authors
        base_path: Base directory path for output
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Find qualifying papers
    works = acl2024_long.find({
        "citationCount": {"$gt": 10},
        "abstract": {"$ne": None},
        "authors": {"$ne": None},
        "references": {"$ne": None},
        "$expr": {"$gt": [{"$size": "$references"}, 20]}
    })

    folder_num = 0
    for work in works:
        if not validate_work(work):
            continue
            
        # Process authors
        valid_authors = []
        for author in work['authors']:
            author_data = author_info.find_one({"authorId": author['authorId']})
            if (author_data and 
                author_data.get('papers') and 
                author_data.get('institutions') and 
                author_data.get('topics')):
                valid_authors.append(author_data)

        if not valid_authors:
            continue
            
        folder_num += 1
        folder_path = base_path / str(folder_num)
        
        # Create directory structure
        (folder_path / "aim_paper").mkdir(parents=True, exist_ok=True)
        (folder_path / "ref_papers").mkdir(exist_ok=True)
        (folder_path / "authors").mkdir(exist_ok=True)

        # Save main paper
        work_data = {
            "id": work['sem_paperId'],
            "title": work['title'],
            "abstract": work['abstract'],
        }
        with open(folder_path / "aim_paper/aim_paper.txt", "w", encoding="utf-8") as f:
            json.dump(work_data, f, indent=4)

        # Save references
        for ref_idx, ref_paper in enumerate(work['references']):
            ref_data = {    
                "id": ref_paper['paperId'],
                "title": ref_paper['title'],
                "abstract": ref_paper['abstract'],
            }
            with open(folder_path / f"ref_papers/ref_paper_{ref_idx}.txt", "w", encoding="utf-8") as f:
                json.dump(ref_data, f, indent=4)

        # Process authors
        for author_idx, author in enumerate(valid_authors):
            process_author(author, author['authorId'], folder_path, author_idx)

if __name__ == "__main__":
    acl2024_long, author_info = get_mongodb_connection()
    count = acl2024_long.count_documents({
        "citationCount": {"$gt": 10},
        "abstract": {"$ne": None},
        "references": {"$ne": None},
        "authors": {"$ne": None},
        "$expr": {"$gt": [{"$size": "$references"}, 20]}
    })
    print(f"Found {count} qualifying papers")
    process_dataset(acl2024_long, author_info)
