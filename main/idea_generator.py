# Standard library imports
import sys
import os
import json
import argparse
import multiprocessing as mp
from functools import partial

# Third-party imports
import nltk

# Local imports
from sci_team import Team
from utils.utils import tournament_ranking, init_client, ensure_string, save_json

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Agentscope setup
sys.path.append('/home/shuaichen/code/virtual_scientists/agentscope-main/src')
import agentscope
from agentscope.agents import SciAgent
from agentscope.message import Msg


class IdeaGenerator:

    def __init__(self,
                 idea_id: int = 0,
                 model_configuration: str = '/home/shuaichen/code/virtual_scientists/main/configs/model_configs.json',
                 data_folder: str = '/home/shuaichen/code/virtual_scientists/dataset/data/acl2024_long/',
                 paper_idx: int = 1,
                 author_info_folder: str = 'authors',
                 agent_model_config_name: str = 'litellm_chat-deepseek-chat',
                 log_dir: str = '/home/shuaichen/code/virtual_scientists/main/logs',
                 group_max_discuss_iteration: int = 3,
                 ):
        """Initialize the Platform with configuration settings
        
        Args:
            idea_id: ID of the idea being processed
            model_configuration: Path to model config file
            data_folder: Path to dataset directory
            paper_idx: Index of paper being processed
            author_info_folder: Name of folder containing author info
            agent_model_config_name: Name of agent model configuration
            log_dir: Directory for log files
            info_dir: Directory for info files
            group_max_discuss_iteration: Max discussion iterations
            recent_n_team_mem_for_retrieve: Number of recent memories to retrieve
            cite_number: Number of citations to consider
        """
        # Path configurations
        self.data_folder = data_folder
        self.results_dir = "/home/shuaichen/code/virtual_scientists/results"
        self.log_dir = log_dir

        # Experiment parameters
        self.paper_idx = paper_idx
        self.idea_id = idea_id

        # Team discussion settings
        self.group_max_discuss_iteration = group_max_discuss_iteration

        # Initialize model configuration
        agentscope.init(model_configs=model_configuration)

        # Initialize agent pool
        self.agent_pool = []
        for agent_id in range(len(os.listdir(f"{data_folder}/{self.paper_idx}/{author_info_folder}"))):
            self.agent_pool.append(self.init_agent(
                str(agent_id), 
                agent_model_config_name, 
                f'{data_folder}/{self.paper_idx}/{author_info_folder}/author{agent_id}/author{agent_id}.txt'
            ))

        # Create agent name to agent mapping
        self.id2agent = {agent.name: agent for agent in self.agent_pool}

        # Initialize host message template
        self.HostMsg = partial(Msg, name="user", role="user", echo=True)

        # Create and initialize team
        self.team = Team(
            team_name=f"1,{len(self.agent_pool)}",
            log_dir=self.log_dir,
        )
        self.team.teammate.extend(agent.name for agent in self.agent_pool)

    def init_agent(self, agent_id, agent_model_config_name, information_path):
        """Initialize a scientist agent with given parameters
        
        Args:
            agent_id: Unique identifier for the agent
            agent_model_config_name: Name of model configuration to use
            information_path: Path to file containing agent's background/prompt
            
        Returns:
            Initialized SciAgent instance
        """
        # load author info
        with open(information_path, 'r') as file:
            prompt = file.read()
        agent = SciAgent(
            name=f'Scientist{agent_id}',
            model_config_name=agent_model_config_name,
            sys_prompt=prompt,
            recent_n_mem_for_retrieve=2,
        )
        return agent
    
   
    def id_to_agent(self, teammate):
        """Convert list of agent IDs to agent objects
        
        Args:
            teammate: List of agent ID strings
            
        Returns:
            List of corresponding SciAgent objects
        """
        agent_list = []
        for agent_id in teammate:
            agent_list.append(self.id2agent[agent_id])
        return agent_list

    def agent_to_id(self, team_list):
        """Convert list of agent objects to their IDs
        
        Args:
            team_list: List of SciAgent objects
            
        Returns: 
            List of agent ID strings
        """
        agent_list = []
        for agent_id in team_list:
            agent_list.append(agent_id.name)
        return agent_list


    

    def _create_turn_dirs(self, turn):
        """Create directory structure for a discussion turn"""
        turn_dir = os.path.join(self.results_dir, str(self.paper_idx), f"idea_{self.idea_id}", f"turn_{turn}")
        dirs = {
            "seed_ideas": os.path.join(turn_dir, "seed_ideas"),
            "generated_ideas": os.path.join(turn_dir, "generated_ideas"),
            "ranking": os.path.join(turn_dir, "ranking"),
            "score": os.path.join(turn_dir, "score")
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        return turn_dir, dirs

    def _load_initial_seed(self, seed_idea_dir):
        """Load initial seed idea from data folder"""
        seed_idea_file = f"{self.data_folder}/{str(self.paper_idx)}/idea_{self.idea_id}.txt"
        with open(seed_idea_file, "r") as f:
            seed_data = json.load(f)
        seed_path = os.path.join(seed_idea_dir, "seed_0.json")
        save_json(seed_path, {
            "title": seed_data["Title"],
            "idea": seed_data["Idea"]
        })
        return seed_path

    def _load_previous_top_ideas(self, turn, seed_idea_dir):
        """Load top ideas from previous turn as new seeds"""
        seed_folder_list = os.listdir(f"{self.results_dir}/{self.paper_idx}/idea_{self.idea_id}/turn_{turn-1}/ranking")
        for seed_folder in seed_folder_list:
            prev_top_ideas = os.path.join(
                self.results_dir, str(self.paper_idx), f"idea_{self.idea_id}", 
                f"turn_{turn-1}", "ranking", seed_folder, "top_ideas.json"
            )
            with open(prev_top_ideas, "r") as f:
                top_ideas = json.load(f)
                top_ideas_paths = list(top_ideas.keys())
            
            for idx, top_idea_path in enumerate(top_ideas_paths):
                self._create_seed_from_previous_idea(
                    seed_idea_dir, seed_folder, idx, top_idea_path, turn
                )

    def _create_seed_from_previous_idea(self, seed_idea_dir, seed_folder, idx, top_idea_path, turn):
        """Create new seed from previous top idea"""
        with open(top_idea_path, "r") as f:
            top_idea = json.load(f)
        
        # Ensure string type for all fields
        title_text = ensure_string(top_idea["Title"])
        idea_text = ensure_string(top_idea["Idea"])
        experiment_text = ensure_string(top_idea["Experiment"])
        bad_review_text = self._collect_bad_reviews(turn, seed_folder, top_idea_path)
        
        seed_path = os.path.join(seed_idea_dir, f"{seed_folder}_seed_{idx}.json")
        save_json(seed_path, {
            "title": title_text,
            "idea": f"{idea_text}\n{experiment_text}",
            "bad_review": bad_review_text
        })

    def _collect_bad_reviews(self, turn, seed_folder, top_idea_path):
        """Collect bad reviews for a top idea from previous turn"""
        bad_review_text = []
        for round in range(5):
            bad_review_file = os.path.join(
                self.results_dir, str(self.paper_idx), f"idea_{self.idea_id}", 
                f"turn_{turn-1}", "score", seed_folder, f"bad_review_{round+1}.json"
            )
        with open(bad_review_file, "r") as f:
            bad_review = json.load(f)
            if bad_review.get(top_idea_path):
                bad_review_text.append(f"{len(bad_review_text)+1}: {bad_review[top_idea_path]}")
        
        return "\n".join(bad_review_text) if bad_review_text else None

   

    def _generate_new_ideas(self, seed_idea_dir, generated_idea_dir):
        """Generate new ideas from seeds"""
        all_generated_ideas_files = []
        for seed_file in os.listdir(seed_idea_dir):
            seed_path = os.path.join(seed_idea_dir, seed_file)
            seed_data = self._load_seed_data(seed_path)
            seed_file_prefix = seed_file[:-5]
            
            seed_output_dir = os.path.join(generated_idea_dir, f"from_{seed_file_prefix}")
            os.makedirs(seed_output_dir, exist_ok=True)
            
            generated_ideas_files = self._process_single_seed(
                seed_data, seed_output_dir, seed_file_prefix
            )
            all_generated_ideas_files.append(generated_ideas_files)
        return all_generated_ideas_files

    def _load_seed_data(self, seed_path):
        """Load seed data with validation"""
        with open(seed_path, "r") as f:
            seed_data = json.load(f)
            if "bad_review" not in seed_data or seed_data.get("bad_review") == "None":
                seed_data["bad_review"] = ""
        return seed_data

    def _process_single_seed(self, seed_data, seed_output_dir, seed_file_prefix):
        """Process a single seed to generate new ideas"""
        self.team.generate_idea_one_by_one(
            self, 
            seed_data["idea"], 
            seed_data["bad_review"],
            seed_output_dir,
            seed_file_prefix
        )
        return [
            os.path.join(seed_output_dir, idea_file)
            for idea_file in os.listdir(seed_output_dir)
        ]

    def _evaluate_ideas(self, idea_file_path_list, client, score_dir, ranking_dir):
        """Evaluate and rank generated ideas"""
        idea_list = []
        seed_folder = os.path.basename(os.path.dirname(idea_file_path_list[0]))
        
        for idea_file_path in idea_file_path_list:
            with open(idea_file_path, "r") as file:
                idea_json = json.loads(file.read())
                title = ensure_string(idea_json["Title"])
                idea_text = ensure_string(idea_json["Idea"])
                experiment = ensure_string(idea_json["Experiment"])
                idea_list.append(f"Title: {title}\nIdea: {idea_text}\nExperiment: {experiment}")
        
        top_ideas, final_scores, all_costs, status = tournament_ranking(
            idea_list, idea_file_path_list, client, f"{score_dir}/{seed_folder}", max_round=5
        )
        if status:
            return
        
        os.makedirs(f"{ranking_dir}/{seed_folder}", exist_ok=True)
        save_json(
            os.path.join(ranking_dir, seed_folder, "scores.json"),
            {"scores": final_scores, "costs": all_costs}
        )
        save_json(
            os.path.join(ranking_dir, seed_folder, "top_ideas.json"),
            top_ideas
        )

    def running(self):
        """Run the platform's main loop"""
        client = init_client()

        for turn in range(self.group_max_discuss_iteration):
            turn_dir, dirs = self._create_turn_dirs(turn)
            
            # Load seed ideas
            if turn == 0:
                self._load_initial_seed(dirs["seed_ideas"])
            else:
                self._load_previous_top_ideas(turn, dirs["seed_ideas"])

            # Generate and evaluate new ideas
            all_generated_ideas_files = self._generate_new_ideas(
                dirs["seed_ideas"], dirs["generated_ideas"]
            )
            for idea_file_path_list in all_generated_ideas_files:
                self._evaluate_ideas(
                    idea_file_path_list, 
                    client, 
                    dirs["score"], 
                    dirs["ranking"]
                )


def run(args):
        idea_id, paper_idx = args
        platform = IdeaGenerator(idea_id=idea_id, paper_idx=paper_idx)
        platform.running()
        

if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    
    parser = argparse.ArgumentParser(description='Run scientific platform')
    parser.add_argument('--idea_ids', type=int, nargs='+', default=[0], help='List of idea IDs to process')
    parser.add_argument('--paper_idxs', type=int, nargs='+', required=True, help='List of paper indexes to process')
    parser.add_argument('--num_processes', type=int, default=15, help='Number of parallel processes')
    
    args = parser.parse_args()
    
    # Process all idea_ids for each paper_idx sequentially
    for paper_idx in args.paper_idxs:
        # Create parameter list of (idea_id, paper_idx) tuples
        params = [(idea_id, paper_idx) for idea_id in args.idea_ids]
        
        # Create process pool to handle all ideas for current paper_idx
        with mp.Pool(processes=min(args.num_processes, len(args.idea_ids))) as pool:
            pool.map(run, params)
            
        print(f"Finished processing all ideas for paper_idx {paper_idx}")
    
    # example 
    # python idea_generator.py --idea_ids 0 1 --paper_idx 127
