from acl_anthology import Anthology
import requests
from pymongo import MongoClient
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from pyalex import Works, Authors
import pyalex
from pyalex import config
from typing import Optional, Tuple, List, Dict, Any

def setup_openalex_config() -> None:
    """Configure OpenAlex API settings"""
    config.max_retries = 3
    config.retry_backoff_factor = 0.1
    config.retry_http_codes = [429, 500, 503]
    pyalex.config.email = "shuaichen@njust.edu.cn"

def get_mongodb_connection() -> Tuple[Any, Any]:
    """
    Get MongoDB collections for papers and authors
    
    Returns:
        Tuple: (papers_collection, author_collection)
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client['scholar']
    return db['acl2024_long'], db['author_info']

def get_author_info(title: str) -> Optional[Tuple[str, List[str], List[List[str]], List[List[str]]]]:
    """
    Get first author's institutions and research topics from OpenAlex
    
    Args:
        title: Paper title
        
    Returns:
        Tuple containing (alexId, author_ids, institutions, topics) if successful
        None if error occurs
    """
    try:
        works_result = Works().search_filter(title=title).get()
        if not works_result:
            print("No matching papers found")
            return None
        
        work = works_result[0]
        alexId = work.get('id')
        if work.get('title') == title:
            authorships = work.get('authorships', [])
            if not authorships:
                print("No author information found")
                return None
                
            all_institutions = []
            all_topics = []
            authors_ids = []
            
            for idx, authorship in enumerate(authorships, 1):
                print(f"Processing author [{idx}/{len(authorships)}]")
                alexAuthorId = authorship.get('author', {}).get('id')
                authors_ids.append(alexAuthorId)
                if alexAuthorId:
                    print(f"Fetching details for author (ID: {alexAuthorId})...")
                    author_info = Authors()[alexAuthorId]
                    institutions = [
                        inst.get('display_name') 
                        for inst in author_info.get('last_known_institutions', [])
                    ]
                    all_institutions.append(institutions)
                    
                    topics = [
                        topic.get('display_name') 
                        for topic in author_info.get('topics', [])
                    ]
                    all_topics.append(topics)

            if all_institutions or all_topics:
                return alexId, authors_ids, all_institutions, all_topics
                
            print("No institution or topic information found")
            return None
        else:
            print("Paper title mismatch")
            return None
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def fetch_author_info(author_ids: List[str]) -> Optional[List[Dict]]:
    """
    Batch fetch author info from Semantic Scholar API
    
    Args:
        author_ids: List of author IDs
        
    Returns:
        List of author info dicts if successful
        None if error occurs
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/author/batch"
        
        response = requests.post(
            url,
            params={"fields": "name,url,affiliations,citationCount,paperCount,hIndex,papers,papers.title,papers.abstract"},
            json={"ids": author_ids},
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching author info: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def fetch_paper_info(paper_title: str) -> Optional[Tuple[Dict, List[str]]]:
    """
    Fetch paper info from Semantic Scholar API
    
    Args:
        paper_title: Title of paper to search for
        
    Returns:
        Tuple containing (paper_info, author_ids) if successful
        None if error occurs
    """
    try:
        r = requests.get(
            'https://api.semanticscholar.org/graph/v1/paper/search/match',
            params={
                'query': paper_title,
                'fields': 'paperId,title,abstract,authors,publicationTypes,citationCount,references,references.title,references.abstract,references.citationCount,references.publicationTypes'
            },
            timeout=10
        )
        r.raise_for_status()
        
        if r.status_code == 200:
            paper_data = r.json()
            if paper_data.get('data') and len(paper_data['data']) > 0:
                paper_info = paper_data['data'][0]
                author_ids = [author['authorId'] for author in paper_info.get('authors', [])]
                
                # Skip papers without abstracts or with low citations
                if paper_info.get('abstract') is None or paper_info.get('citationCount') < 1:
                    print(f"Skipping paper - no abstract or low citations: {paper_title}")
                    return None
                
                return {
                    'sem_paperId': paper_info.get('paperId'),
                    'title': paper_info.get('title'),
                    'citationCount': paper_info.get('citationCount'),
                    'abstract': paper_info.get('abstract'),
                    'authors': paper_info.get('authors', []),
                    'references': [
                        ref for ref in paper_info.get('references', [])
                        if (ref.get('abstract') and 
                            ref.get('citationCount', 0) > 10 and 
                            any(pub_type in ['Conference', 'JournalArticle'] 
                                for pub_type in ref.get('publicationTypes', []) or []))
                    ]
                }, author_ids
            print(f"No matching paper found: {paper_title}")
            return None
        
        print(f"API request failed: {r.status_code}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching paper info: {str(e)}")
        raise

# Initialize Anthology and OpenAlex
anthology = Anthology(datadir="/home/shuaichen/acl-anthology/data")
setup_openalex_config()

# Get ACL 2024 long papers
volume = anthology.get("2024.acl-long")
papers_title = [paper.title for paper in volume.data.values()][1:]

# Get MongoDB collections
papers_collection, author_collection = get_mongodb_connection()

# Process papers with progress tracking
total_papers = len(papers_title)
for index, paper_title in enumerate(papers_title, 1):
    try:
        print(f"Processing [{index}/{total_papers}]: {paper_title}")
        
        # Get OpenAlex info
        alex_result = get_author_info(str(paper_title))
        alexId = alexAuthorIds = institutions = topics = None
        if alex_result:
            alexId, alexAuthorIds, institutions, topics = alex_result
        else:
            print(f"No matching paper found in OpenAlex: {paper_title}")

        # Get Semantic Scholar info
        paper_info = fetch_paper_info(paper_title)
        if not paper_info:
            continue
        paper_data, author_ids = paper_info
        
        time.sleep(1)  # Rate limiting

        # Get author details
        authors_info = fetch_author_info(author_ids)
        time.sleep(1)  # Rate limiting
        
        # Store author info
        for idx, author_info in enumerate(authors_info):
            update_data = {
                'alexAuthorId': alexAuthorIds[idx] if alexAuthorIds else None,
                'institutions': institutions[idx] if institutions else None,
                'topics': topics[idx] if topics else None,
                **author_info
            }
            author_collection.update_one(
                {'authorId': author_info['authorId']},
                {'$set': update_data},
                upsert=True
            )
        print("Author info stored successfully")

        # Store paper info
        papers_collection.update_one(
            {'sem_paperId': paper_data['sem_paperId']},
            {'$set': {'alexId': alexId, **paper_data}},
            upsert=True
        )
        print(f"✓ Paper stored: {paper_title}")
        
    except Exception as e:
        print(f"Error processing paper: {paper_title}")
        print(f"Error details: {str(e)}")
