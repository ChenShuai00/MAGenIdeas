from datetime import datetime

import logging

import json
import os
import sys
import json
sys.path.append('../agentscope-main/src')

from agentscope.message import Msg
from utils.prompt import Prompts
from utils.utils import clean_json_string,get_ref_papers
from utils.scientist_utils import (
    format_msg,
    extract_between_json_tags,
    extract_metrics,
) 

class Team:
    def __init__(self, team_name, log_dir, info_dir):
        # attrs
        self.team_name = team_name
        self.teammate = []
        # init log file dir
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.info_file = f"{info_dir}/{current_time}_{self.team_name}_dialogue.json"
        self.log_file = f"{log_dir}/{current_time}_{self.team_name}_dialogue.log"

        # Check if log file exists and delete it
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.logger = logging.getLogger(self.team_name)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        self.logger.addHandler(fh)

    def generate_idea_one_by_one(self, platform, idea: str, bad_review: str, 
                               seed_output_dir: str, seed_file_prefix: str) -> tuple:
        """Generate ideas from team members one by one with evaluation.
        
        Args:
            platform: Reference to platform instance
            idea: Seed idea to build upon
            bad_review: Critical feedback on previous iteration (optional)
            seed_output_dir: Directory to save generated ideas
            seed_file_prefix: Prefix for output filenames
            
        Returns:
            tuple: (last_save_path, list_of_ideas) 
        """
        seed_idea = idea
        # Create output directory
        os.makedirs(seed_output_dir, exist_ok=True)
        if len(os.listdir(seed_output_dir)) > 0:
            print(f"Output directory {seed_output_dir} is not empty")
            return None, None
            
        # Get related papers and prepare team
        print(f'{"="*40}')
        print(f'Strat searching relevant papers for idea generation')
        print(f'{"="*40}')
        paper_reference, unique_papers = get_ref_papers(idea)
        print(f'{"="*40}')
        print(f'Finsh searching relevant papers for idea generation')
        print(f'{"="*40}')
        teammate = platform.id_to_agent(self.teammate)
        
        
        
        best_score = 0
        best_idea = None
        idea_list = []
        last_save_path = None
        
        for agent_id, agent in enumerate(teammate):
            # Build appropriate prompt based on whether we have bad reviews
            prompt_parts = [
                Prompts.prompt_task,
                Prompts.prompt_existing_idea.format(seed_idea),
                Prompts.prompt_reference.format(paper_reference)
            ]
            
            if bad_review:
                prompt_parts.insert(2, Prompts.prompt_bad_review.format(bad_review))
                
            prompt_parts.append(
                Prompts.prompt_response.format(
                    Prompts.prompt_excitement,
                    Prompts.prompt_excitement_rationale,
                    Prompts.prompt_feasibility, 
                    Prompts.prompt_feasibility_rationale,
                    Prompts.prompt_novelty,
                    Prompts.prompt_novelty_rationale
                )
            )
            
            idea_prompt = ''.join(prompt_parts)
            agent_prompt = format_msg(
                Msg(name="user", role="user", content=idea_prompt)
            )
            
            # Get agent's response
            reply = agent.prompt_reply(
                agent_prompt, 
                add_memory=False, 
                use_memory=False, 
                use_RAG=False
            )
            self.log_dialogue('user', idea_prompt)
            self.log_dialogue(agent.name, reply.content)
            
            # Extract and evaluate the idea
            idea_str = extract_between_json_tags(reply.content, num=1)
            metrics = extract_metrics(idea_str, ['Excitement', 'Feasibility', 'Novelty'])
            
            # Calculate total score and validate metrics
            total_score = 0
            is_valid = all(metrics[key] is not None for key in metrics)
            if is_valid:
                total_score = sum(metrics.values())
                
                try:
                    # Clean and parse the JSON idea
                    cleaned_idea = clean_json_string(idea_str)
                    idea_dict = json.loads(cleaned_idea)
                    idea_dict.update({
                        'ref_papers': unique_papers,
                        'total_score': total_score,
                        'meets_threshold': total_score >= 23  # Quality threshold
                    })
                    
                    # Track best idea
                    if total_score > best_score:
                        best_score = total_score
                        best_idea = idea_dict
                    
                    # Save this idea
                    last_save_path = os.path.join(
                        seed_output_dir, 
                        f"{seed_file_prefix}_agent_{agent_id}_idea.json"
                    )
                    with open(last_save_path, 'w', encoding='utf-8') as f:
                        json.dump(idea_dict, f, indent=4, ensure_ascii=False)
                    idea_list.append(idea_dict)
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parse error for agent {agent_id}: {e}")
                    continue
        
        # Fallback: Save best idea if no others met threshold
        if not idea_list and best_idea:
            last_save_path = os.path.join(
                seed_output_dir,
                f"{seed_file_prefix}_best_idea.json"
            )
            with open(last_save_path, 'w', encoding='utf-8') as f:
                json.dump(best_idea, f, indent=4, ensure_ascii=False)
            idea_list.append(best_idea)

        return last_save_path, idea_list

    def log_dialogue(self, name, content):
        self.logger.info(f'{name}:{content}')
        self.logger.info(f'{"="*40}')
