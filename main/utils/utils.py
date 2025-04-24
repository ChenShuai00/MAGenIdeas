"""Utility functions for the scientific research platform.

This module contains various utility functions grouped into:
1. OpenAI API utilities
2. Paper search utilities  
3. Idea comparison and ranking
4. General utilities
"""

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_exponential
import json
from openai import OpenAI
import os
import re
from collections import defaultdict
import random
import tqdm

from .chat_model import deepseek_chat
from .scholar_search import semantic_scholar_search
from .prompt import Prompts
import time

# ==============================================
# OpenAI API Utilities
# ==============================================

def init_client():
        """Initialize OpenAI client with API key validation"""
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        if not client.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((IndexError, json.JSONDecodeError, KeyError)),
    retry_error_callback=lambda retry_state: ([], retry_state.outcome.result()[1])
)

def get_search_keywords(idea: str):
      
        # Create client and get response
        client = init_client()
        response, tokens = deepseek_chat(client, Prompts.search_keyword_sys_prompt, Prompts.search_keyword_prompt.format(idea))
        print(f'{"="*40}')
        print(f'{"*"*20} Generate Search Plans {"*"*20}\n')
        print(response)
        print(f'{"="*40}')
        # Parse JSON response
        json_blocks = re.findall(r'```json(.*?)```', response, re.DOTALL)[0]
        json_data = json.loads(json_blocks)
        search_keywords = json_data['Search_Keywords']
        if not search_keywords:  # If keyword list is empty, raise exception to trigger retry
            raise ValueError("未获取到关键词")
        return search_keywords, tokens

# ==============================================
# Paper Search Utilities
# ==============================================

def get_reference_paper(query: str, result_limit=3):
        if not query:
            return "", []
        
        reference_paper = []
        # Use complete query string directly
        query_str = ' '.join(query)
        query_str = query_str.replace('-', ' ').replace('"','').replace(',','')
        results = semantic_scholar_search(query_str, result_limit)
        if "data" in results:
            for paper in results["data"]:
                if not paper.get('abstract') or not paper.get('publicationTypes'):
                    continue
   
                valid_types = [t for t in paper['publicationTypes'] if t in ["JournalArticle", "Conference"]]
                if valid_types:
                    reference_paper.append({
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'types': valid_types  # Include all valid types
                    })
        else:
            reference_paper = []
        print(reference_paper)
        print(f'{"="*40}')
        return reference_paper


    

def get_ref_papers(idea: str):
    """Get reference papers for a research idea by first generating search keywords
    then retrieving relevant papers for each keyword set.
    
    Args:
        idea: Research idea to find papers for
        
    Returns:
        tuple: (formatted_references, list_of_papers)
    """
    search_keywords, _ = get_search_keywords(idea)

    titles_seen = set()
    paper_reference = ""
    search_papers = []
    unique_papers = []
    for search_keyword in search_keywords:
        field = search_keyword['Field']
        keywords = search_keyword['Keywords']
        print(f'{"="*40}')
        print(f'{"*"*20} Search papers in **{field}** {"*"*20}\n')
        print(f'search keywords: {keywords}')
        search_papers.extend(get_reference_paper(keywords))
    for id, paper in enumerate(search_papers):
        if paper['title'] not in titles_seen:
            titles_seen.add(paper['title'])
            unique_papers.append(paper)
            paper_reference += f"Paper {id+1}:\n"
            paper_reference += f"Title: {paper['title']}\n"
            paper_reference += f"Abstract: {paper['abstract']}\n"
    return paper_reference, unique_papers



# ==============================================
# Idea Comparison and Ranking
# ==============================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def better_idea(idea_1, idea_2, method, client, few_shot_demos=None):
        sys_prompt = "You are a reviewer specialized in Natural Language Processing. You are given two project summaries. One of them is accepted by a top AI conference (like ICLR or ACL) and the other one is rejected. Your task is to identify the one that has been accepted.\n"
        prompt = ""
        ## zero-shot methods
        if "zero_shot" in method:
            prompt += "The two project proposals are:\n\n" 
            prompt += "idea 1:\n" + idea_1+ "\n\n"
            prompt += "idea 2:\n" + idea_2 + "\n\n"
            # prompt += "\nYou can consider factors like novelty, soundness, excitement, and potential impact.\n"
            if method == "zero_shot":
                prompt += "Now decide which one is the accepted idea. Directly return a number 1 or 2 and nothing else.\n"
            elif method == "zero_shot_cot":
                prompt += "Now decide which one is the accepted idea. Think step by step by writing a meta-review to compare the strengths and weaknesses of both ideas and explain why one idea is better than the other. After the meta-review, start a new line and directly return a number 1 or 2 to indicate the accepted idea and end the response.\n"
        ## few-shot methods
        elif "self-review" in method:
            prompt += "\n\nThe two project summaries given to you are:\n\n" 
            prompt += "paper 1:\n" + idea_1 + "\n\n"
            prompt += "paper 2:\n" + idea_2 + "\n\n"
            # prompt += "\nYou should consider factors like novelty, soundness, excitement,and potential impact.\n"

            if method == "self-review":
                prompt += "Now decide which one is the accepted idea. Follow the above examples: return a number 1 or 2 and nothing else.\n"
            elif method == "self-review_cot":
                prompt += """Now decide which one is the accepted idea. give a meta-review to each paper.
                ```json{
                        Decision: <DECISION>
                        ReviewForPaper1: <ReviewForPaper1>
                        ReviewForPaper2: <ReviewForPaper2>
                        } ```

                In <DECISION>, return a number 1 or 2 and nothing else 
                In <ReviewForPaper1>, write the review for paper 1,reasons for acceptance or rejection.
                In <ReviewForPaper2>, write the review for paper 2,reasons for acceptance or rejection.
                This JSON will be automatically parsed, so ensure the format is precise.\n"""

        response, cost = deepseek_chat(client, sys_prompt, prompt)
        json_blocks = re.findall(r'```json(.*?)```', response, re.DOTALL)[0]
        json_dict = json.loads(json_blocks)
        return prompt, json_dict, cost


def tournament_ranking(idea_lst, filename_lst, openai_client, ranking_score_dir, max_round=5):
    """Run tournament-style ranking of research ideas using pairwise comparisons.
    
    Args:
        idea_lst: List of research ideas to rank
        filename_lst: Corresponding filenames for the ideas
        openai_client: OpenAI client instance
        ranking_score_dir: Directory to save ranking results
        max_round: Maximum number of ranking rounds
        
    Returns:
        tuple: (top_ideas, final_scores, total_cost, cache_exists)
    """
    scores = defaultdict(lambda: 1)
    idea_review = defaultdict(str)  # Use defaultdict to avoid KeyError
    all_costs = 0
    os.makedirs(ranking_score_dir, exist_ok=True)
    if len(os.listdir(ranking_score_dir)) > 0:
        return None, None, None, True
        
    def single_round(ideas, current_round=0, all_costs=0):
        if current_round == 0:
            random.shuffle(ideas)

        match_pairs = []
        sorted_ideas = sorted(ideas, key=lambda idea: scores[idea[:200]], reverse=True)

        for i in range(0, len(sorted_ideas), 2):
            if i + 1 < len(sorted_ideas):
                match_pairs.append((sorted_ideas[i], sorted_ideas[i+1]))
            else:
                # If there is an odd number of ideas, the last one automatically wins this round
                scores[sorted_ideas[i][:200]] += 1

        for idea1, idea2 in tqdm.tqdm(match_pairs):
            prompt, result, cost = better_idea(idea1, idea2, "self-review_cot", openai_client)
            if result['Decision'] == 1:
                scores[idea1[:200]] += 1
                idea_review[idea2[:200]] = result.get('ReviewForPaper2', '')  
            else:
                scores[idea2[:200]] += 1
                idea_review[idea1[:200]] = result.get('ReviewForPaper1', '')  

            all_costs += cost

        return all_costs
        
    # Conduct the tournament rounds until only one idea remains
    current_round = 0
    score_predictions = {}
    bad_review = {}
    while current_round < max_round:
        print("Current round: ", current_round + 1)
        all_costs = single_round(idea_lst[:], current_round=current_round, all_costs=all_costs)
        current_round += 1

        # Convert scores to a list matching the order of the original idea list
        final_scores = [scores[idea[:200]] for idea in idea_lst]
        final_reviews = [idea_review[idea[:200]] for idea in idea_lst]
        for i in range(len(filename_lst)):
            score_predictions[filename_lst[i]] = final_scores[i]
            bad_review[filename_lst[i]] = final_reviews[i]
        os.makedirs(ranking_score_dir, exist_ok=True)
        # Save all scores
        cache_file = os.path.join(ranking_score_dir, "round_{}.json".format(current_round))
        bad_review_file = os.path.join(ranking_score_dir, "bad_review_{}.json".format(current_round))
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))
        with open(cache_file, "w") as f:
            json.dump(score_predictions, f, indent=4)
        with open(bad_review_file, "w") as f:
            json.dump(bad_review, f, indent=4)
                
    top_ideas = {}
    # Filter ideas with score >= 5
    sorted_ideas_with_idx = [(idx, (fname, score)) for idx, (fname, score) in enumerate(zip(filename_lst, final_scores)) if score >= 5]
    
    # If no ideas with score >=5, select the highest scoring one
    if not sorted_ideas_with_idx:
        # Get highest score
        max_score = max(final_scores)
        # Select only the highest scoring idea
        sorted_ideas_with_idx = [(idx, (fname, score)) 
                               for idx, (fname, score) 
                               in enumerate(zip(filename_lst, final_scores))
                               if score == max_score]
    
    # Sort by score descending
    sorted_ideas_with_idx = sorted(sorted_ideas_with_idx, 
                                 key=lambda x: x[1][1], 
                                 reverse=True)
    for idx, (idea_name, score) in sorted_ideas_with_idx:
        top_ideas[idea_name] = {
            "idea": idea_lst[idx],
            "ai_ranking_score": score,
        }
    top_ideas_file = os.path.join(ranking_score_dir, "top_ideas.json")
    with open(top_ideas_file, "w") as f:
        json.dump(top_ideas, f, indent=4)
    return top_ideas, final_scores, all_costs, False


# ==============================================
# General Utilities
# ==============================================

def ensure_string(value):
        """Ensure value is string type"""
        return str(value) if isinstance(value, dict) else value

def save_json(path, data):
    """Save data to JSON file with consistent formatting.
    
    Args:
        path: File path to save to
        data: Data to save (must be JSON-serializable)
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def clean_json_string(json_str: str) -> str:
    """Remove invalid control characters from JSON string.
    
    Args:
        json_str: Potentially dirty JSON string
        
    Returns:
        Cleaned JSON string
    """
    if not json_str:
        return json_str
    return ''.join(char for char in json_str 
                  if ord(char) >= 32 or char in '\n\r\t')
