from openai import OpenAI
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils.prompt import Prompts

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_research_ideas(client: OpenAI, prompt: str, max_retries: int = 3) -> Tuple[Optional[str], Optional[Any]]:
    """
    Generate research ideas with retry mechanism and better error handling.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert researcher in AI."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content, response.usage
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached, giving up")
                return None, None

def load_paper_data(file_path: Path) -> Dict[str, str]:
    """Load paper data with parallel processing support"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            paper_json = json.load(file)  # Use json.load instead of json.loads(file.read())
            return {
                "title": paper_json.get("title", ""),
                "abstract": paper_json.get("abstract", "")
            }
    except Exception as e:
        logger.error(f"Error loading paper data {file_path}: {str(e)}")
        return {"title": "", "abstract": ""}

def extract_ideas_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse all IDEA JSON data from text with better error handling.
    """
    if not text:
        return []
    
    try:
        json_blocks = re.findall(r'IDEA: ```json\n(.*?)\n```', text, re.DOTALL)
        ideas = []
        for block in json_blocks:
            try:
                ideas.append(json.loads(block))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON block: {block[:100]}... Error: {str(e)}")
        return ideas
    except Exception as e:
        logger.error(f"Error extracting ideas: {str(e)}")
        return []

def load_papers_parallel(ref_papers_dir: Path) -> List[Dict[str, str]]:
    """Load reference papers in parallel"""
    with ThreadPoolExecutor() as executor:
        return list(executor.map(
            load_paper_data,
            [f for f in ref_papers_dir.iterdir() if f.is_file()]
        ))

def save_ideas_batch(output_dir: Path, ideas: List[Dict[str, Any]]) -> None:
    """Save all ideas in a batch operation"""
    for idx, idea in enumerate(ideas):
        output_file = output_dir / f"idea_{idx}.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(idea, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved idea to {output_file}")
        except Exception as e:
            logger.error(f"Error saving idea {idx}: {str(e)}")

def main(paper_id: str, ideas_num: int ) -> None:
    """
    paper_id: str: The ID of the paper to process.
    ideas_num: int: The number of ideas to generate.
    """
    base_path = Path("/home/shuaichen/code/virtual_scientists/dataset/data/acl2024_long")
    output_dir = base_path / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    
    # Load target paper
    target_paper = load_paper_data(base_path / f"{paper_id}/aim_paper/aim_paper.txt")
    
    # Load reference papers in parallel
    ref_papers_dir = base_path / f"{paper_id}/ref_papers"
    reference_papers = load_papers_parallel(ref_papers_dir)
    
    # Generate prompt and ideas
    scientific_discovery_prompt = Prompts.scientific_discovery_prompt.format(
        Prompts.scientific_discovery_theory,
        target_paper,
        reference_papers
    )
    
    ideas_lists = []
    for _ in range(ideas_num):
        response, _ = generate_research_ideas(client, scientific_discovery_prompt)
        if response:
            ideas_lists.extend(extract_ideas_from_text(response))
    
    # Save all ideas
    if ideas_lists:
        save_ideas_batch(output_dir, ideas_lists)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", type=str, help="ID of the paper to process")
    parser.add_argument("--ideas_num", type=int, default=3, help="Number of ideas to generate")
    args = parser.parse_args()
    main(args.paper_id, args.ideas_num)
