import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from openai import OpenAI
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
import tqdm
from collections import defaultdict, Counter
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type,wait_exponential
import random
import pandas as pd
from matplotlib import font_manager

class Evaluation:
    def __init__(self, base_path, turn=2):
        self.base_path = base_path
        self.turn = turn
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.embeding_model_path = "/home/shuaichen/all-MiniLM-L6-v2"
        self.evaluation_dir = None
        self.abstract_dir = None
        self.ranking_score_dir = None
        self.image_dir = None
    
    def get_embedding(self, sentences_list):
        """获取文本嵌入向量"""
        try:
            model = SentenceTransformer(self.embeding_model_path)
            return model.encode(sentences_list)
        except Exception as e:
            print(f"生成嵌入向量时出错: {str(e)}")
            return None

    def deepseek_chat(self, sys_prompt, prompt):
        """调用 DeepSeek API 进行对话"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content, response.usage.total_tokens
        except Exception as e:
            print(f"API 调用出错: {str(e)}")
        return None, 0
    
    def get_top_ideas(self, base_path):
        """获取所有顶级想法的路径"""
        top_idea_paths = []
        for idea_id in range(len(os.listdir(base_path))):
            top_ideas_dir = os.path.join(base_path, f"idea_{idea_id}", f"turn_{self.turn}", "ranking")
            if not os.path.exists(top_ideas_dir):
                continue
            else:
                for seed_folder in os.listdir(top_ideas_dir):
                    seed_path = os.path.join(top_ideas_dir, seed_folder, "top_ideas.json")
                    try:
                        with open(seed_path, "r") as f:
                            top_ideas = json.load(f)
                            top_idea_paths.extend(list(top_ideas.keys()))
                    except Exception as e:
                        print(f"读取文件 {seed_path} 时出错: {str(e)}")
        return top_idea_paths

    def generate_abstract(self, top_idea_path, abstract_dir, id):
        """为单个想法生成摘要"""
        # 获取原始文件名并在 abstract_dir 中创建对应的摘要文件路径
        abstract_filename = f"abstract_{id}.json"
        # 拼接abstract_dir和abstract_filename
        abstract_path = os.path.join(abstract_dir, abstract_filename)
        # 如果摘要文件已存在，直接读取返回
        if os.path.exists(abstract_path):
            try:
                with open(abstract_path, 'r') as f:
                    abstract_content = json.load(f)
                    title_abstract = abstract_content['Title'] + "\n" + abstract_content['Abstract']
                    return title_abstract
            except Exception as e:
                print(f"读取已存在摘要时出错 {abstract_path}: {str(e)}")

        try:
            with open(top_idea_path, 'r') as file:
                top_idea = json.load(file)
                title = top_idea['Title']
                idea = top_idea['Idea']
                experiment = top_idea['Experiment']

                sys_prompt = """You are an ambitious scientist who will generate a summary based on given research idea and experimental steps."""

                prompt = f"""
                        Requirements: The content of the abstract should cover: research questions and objectives, research methods, expected research results, and conclusions.Do not exceed 300 words.
                        Here is the research idea: '''{title}\nIdea: {idea}'''
                        Here is the experimental steps: '''{experiment}'''
                        Please respond in the following format: 
                        Thought: <THOUGHT> 
                        Abstract: ```json<JSON>```
                        In <THOUGHT>, please briefly describe your thinking.
                        In <JSON>, provide the abstract with the following fields: 
                        - "Title": A title for the abstract.
                        - "Abstract": abstract.
                        Be cautious and realistic on your ratings.
                        """

                response, _ = self.deepseek_chat(sys_prompt, prompt)
                if response:
                    json_blocks = re.findall(r'```json(.*?)```', response, re.DOTALL)[0]
                    json_data = json.loads(json_blocks)
                    # 创建JSON格式的内容
                    abstract_content = {
                        "Title": json_data['Title'],
                        "Abstract": json_data['Abstract'],
                        "Source": top_idea_path
                    }

                    # 保存生成的摘要为JSON格式
                    try:
                        with open(abstract_path, 'w') as f:
                            json.dump(abstract_content, f, indent=4)
                    except Exception as e:
                        print(f"保存摘要时出错 {abstract_path}: {str(e)}")

                    # 返回标题和摘要的组合
                    return f"{json_data['Title']}\n{json_data['Abstract']}"
        except Exception as e:
            print(f"生成摘要时出错 {top_idea_path}: {str(e)}")
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10))
    def better_idea(self,abstract_1, abstract_2, method):
        sys_prompt = "You are a reviewer specialized in Natural Language Processing. You are given two abstracts. One of them is accepted by a top AI conference (like ICLR or ACL) and the other one is rejected. Your task is to identify the one that has been accepted.\n"
        prompt = ""
        if "zero_shot" in method:
            prompt += "The two project proposals are:\n\n" 
            prompt += "abstract 1:\n" + abstract_1+ "\n\n"
            prompt += "abstract 2:\n" + abstract_2 + "\n\n"
            # prompt += "\nYou can consider factors like novelty, soundness, excitement, and potential impact.\n"
            if method == "zero_shot":
                prompt += "Now decide which one is the accepted idea. Directly return a number 1 or 2 and nothing else.\n"
            elif method == "zero_shot_cot":
                prompt += "Now decide which one is the accepted idea. Think step by step by writing a meta-review to compare the strengths and weaknesses of both ideas and explain why one idea is better than the other. After the meta-review, start a new line and directly return a number 1 or 2 to indicate the accepted idea and end the response.\n"

        response, cost = self.deepseek_chat(sys_prompt, prompt) 
        return response, cost


    def tournament_ranking(self, abstract_lst, filename_lst, ranking_score_dir, max_round=5):
        scores = defaultdict(lambda: 1)
        all_costs = 0
        os.makedirs(ranking_score_dir, exist_ok=True)
        if len(os.listdir(ranking_score_dir)) > 0:
            return None, None, None, True
        
        def single_round(abstract_lst, current_round=0, all_costs=0):
            if current_round == 0:
                random.shuffle(abstract_lst)

            match_pairs = []
            sorted_abstracts = sorted(abstract_lst, key=lambda abstract: scores[abstract[:200]], reverse=True)
            for i in range(0, len(sorted_abstracts), 2):
                if i + 1 < len(sorted_abstracts):
                    match_pairs.append((sorted_abstracts[i], sorted_abstracts[i+1]))
                else:
                    # If there is an odd number of ideas, the last one automatically wins this round
                    scores[sorted_abstracts[i][:200]] += 1

            for abstract1, abstract2 in tqdm.tqdm(match_pairs):
                result, cost = self.better_idea(abstract1, abstract2, "zero_shot")
                if result.strip() == '1':
                    scores[abstract1[:200]] += 1
                else:
                    scores[abstract2[:200]] += 1
                all_costs += cost
            return all_costs
        
        # Conduct the tournament rounds until only one idea remains
        current_round = 0
        score_predictions = {}
        while current_round < max_round:
            print("Current round: ", current_round + 1)
            all_costs = single_round(abstract_lst[:], current_round=current_round, all_costs=all_costs)
            current_round += 1

            # Convert scores to a list matching the order of the original idea list
            final_scores = [scores[abstract[:200]] for abstract in abstract_lst]
            for i in range(len(filename_lst)):
                score_predictions[filename_lst[i]] = final_scores[i]
            os.makedirs(ranking_score_dir, exist_ok=True)
            # Save all scores
            cache_file = os.path.join(ranking_score_dir, "round_{}.json".format(current_round))
            if not os.path.exists(os.path.dirname(cache_file)):
                os.makedirs(os.path.dirname(cache_file))
            with open(cache_file, "w") as f:
                json.dump(score_predictions, f, indent=4)
                        
            top_ideas = {}
            sorted_ideas_with_idx = [(idx, (fname, score)) for idx, (fname, score) 
                                   in enumerate(zip(filename_lst, final_scores)) ]
            
            
            # 按分数降序排序
            sorted_ideas_with_idx = sorted(sorted_ideas_with_idx, 
                                         key=lambda x: x[1][1], 
                                         reverse=True)
            for idx, (idea_name, score) in sorted_ideas_with_idx:
                top_ideas[idea_name] = {
                    "idea": abstract_lst[idx],
                    "ai_ranking_score": score,
                }
            abstract_ranking_file = os.path.join(ranking_score_dir, "abstract_ranking_score.json")
            with open(abstract_ranking_file, "w") as f:
                json.dump(top_ideas, f, indent=4)
        return top_ideas, final_scores, all_costs, False

    def setup_directories(self, paper_id):
        """设置评估所需的目录结构"""
        self.evaluation_dir = f"/home/shuaichen/code/virtual_scientists/evaluation/{paper_id}/turn_{self.turn}"
        self.abstract_dir = f"{self.evaluation_dir}/abstract"
        self.ranking_score_dir = f"{self.evaluation_dir}/ranking_score"
        self.image_dir = f"{self.evaluation_dir}/image"
        
        for directory in [self.evaluation_dir, self.abstract_dir, 
                         self.image_dir, self.ranking_score_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def process_abstracts(self, top_idea_paths):
        """处理并生成摘要"""
        abstract_list = [None] * len(top_idea_paths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {
                executor.submit(self.generate_abstract, path, self.abstract_dir, id): id 
                for id, path in enumerate(top_idea_paths)
            }
            
            for future in concurrent.futures.as_completed(future_to_id):
                id = future_to_id[future]
                try:
                    abstract_list[id] = future.result()
                except Exception as e:
                    print(f"处理ID {id} 时发生错误: {str(e)}")
        
        return [a for a in abstract_list if a is not None]
    
    def get_abstract_contents(self):
        """获取摘要内容列表"""
        abstract_files = [os.path.join(self.abstract_dir, f) for f in os.listdir(self.abstract_dir)]
        abstract_content_list = []
        
        for abstract_file in abstract_files:
            with open(abstract_file, "r") as f:
                abstract_content = json.load(f)
                abstract_content_list.append(abstract_content['Title'] + "\n" + abstract_content['Abstract'])
        
        return abstract_content_list
    
    def get_reference_papers(self, top_idea_paths):
        """获取参考文献列表"""
        ref_papers_list = []
        for idea_path in top_idea_paths:
            with open(idea_path, "r") as f:
                idea_content = json.load(f)
                ref_papers = [ref["title"]+ref["abstract"] for ref in idea_content['ref_papers'][:10]]
                ref_papers_list.append(ref_papers)
        return ref_papers_list
    
    def evaluate_ideas(self, paper_id):
        """评估单个论文ID的想法"""
        self.setup_directories(paper_id)
        
        # 获取想法路径
        if paper_id in [1001, 1002, 1003, 1004, 1005]:
            top_idea_paths = [os.path.join(self.base_path, path) for path in os.listdir(self.base_path)]
        else:
            top_idea_paths = self.get_top_ideas(self.base_path)
            
        # 处理摘要
        abstract_list = self.process_abstracts(top_idea_paths)
        abstract_content_list = self.get_abstract_contents()
        ref_papers_list = self.get_reference_papers(top_idea_paths)
        
        # 计算排名分数
        if len(os.listdir(self.ranking_score_dir)) == 0:
            self.tournament_ranking(abstract_content_list, top_idea_paths, self.ranking_score_dir)
        
        # 计算各项指标
        scores_ratio = self._calculate_scores_ratio()
        ref_similarity = calculate_ref_abstract_similarity(abstract_list, ref_papers_list)
        embeddings = self.get_embedding(abstract_list)
        metrics = calculate_similarity_metrics(embeddings)
        
        return {
            'idea_num': len(abstract_list),
            'similarity_ratio': metrics['ratio'],
            'unique_ratio': metrics['unique_ratio'],
            'novelty_ratio': ref_similarity['novel_ideas_ratio'],
            'high_score_ratio': scores_ratio
        }
    
    def _calculate_scores_ratio(self):
        """计算高分比例"""
        with open(os.path.join(self.ranking_score_dir, "abstract_ranking_score.json"), "r") as f:
            abstract_ranking_score = json.load(f)
            scores = [content['ai_ranking_score'] for content in abstract_ranking_score.values()]
            high_scores = sum(1 for score in scores if score >= 5)
            return high_scores / len(scores) if scores else 0

def calculate_similarity_metrics(embeddings, threshold=0.8):
    """计算相似度指标"""
    similarity_matrix = cosine_similarity(embeddings)
    upper_triangle_mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
    upper_triangle_values = similarity_matrix[upper_triangle_mask]
    
    similar_pairs = np.sum(upper_triangle_values >= threshold)
    total_pairs = len(upper_triangle_values)
    similarity_ratio = similar_pairs / total_pairs if total_pairs > 0 else 0
    
    # 计算独特想法
    # 对每个想法，检查它是否与其他任何想法的相似度都低于阈值
    n_ideas = len(embeddings)
    unique_ideas = []
    for i in range(n_ideas):
        # 获取当前想法与所有其他想法的相似度
        similarities = similarity_matrix[i]
        # 排除自身的相似度（总是1.0）
        other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
        # 如果所有相似度都低于阈值，则认为是独特想法
        if np.all(other_similarities < threshold):
            unique_ideas.append(i)
    
    unique_ratio = len(unique_ideas) / n_ideas if n_ideas > 0 else 0
    
    return {
        'ratio': similarity_ratio,
        'unique_ideas': unique_ideas,
        'unique_ratio': unique_ratio
    }

def calculate_ref_abstract_similarity(abstract_list, ref_papers_list, novelty_threshold=0.5):
    """计算每个摘要与其参考文献的相似度
    
    Args:
        abstract_list: 摘要列表
        ref_papers_list: 每个摘要对应的参考文献列表的列表
        novelty_threshold: 新颖性阈值，超过此值视为不够新颖
    
    Returns:
        dict: 包含相似度统计和新颖性分析的字典
    """
    # 初始化 SentenceTransformer 模型
    model = SentenceTransformer('/home/shuaichen/all-MiniLM-L6-v2')
    
    # 存储所有相似度的列表
    all_avg_similarities = []
    all_max_similarities = []
    
    # 存储新颖性分析结果
    novelty_analysis = []
    
    # 对每个摘要和其参考文献计算相似度
    for abstract, ref_papers in zip(abstract_list, ref_papers_list):
        if not ref_papers:
            continue
            
        abstract_embedding = model.encode([abstract])
        ref_embeddings = model.encode(ref_papers)
        similarities = cosine_similarity(abstract_embedding, ref_embeddings)[0]
        
        # 判断新颖性
        is_novel = all(sim < novelty_threshold for sim in similarities)
        max_similarity = np.max(similarities)
        avg_similarity = np.mean(similarities)
        
        novelty_analysis.append({
            'is_novel': is_novel,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity
        })
        
        all_avg_similarities.append(avg_similarity)
        all_max_similarities.append(max_similarity)
    
    # 扩展结果字典
    results = {
        'mean_avg_similarity': np.mean(all_avg_similarities),
        'mean_max_similarity': np.mean(all_max_similarities),
        'std_avg_similarity': np.std(all_avg_similarities),
        'std_max_similarity': np.std(all_max_similarities),
        'novelty_analysis': novelty_analysis,
        'novel_ideas_ratio': sum(1 for x in novelty_analysis if x['is_novel']) / len(novelty_analysis) if novelty_analysis else 0
    }
    
    return results

def plot_team_metrics(team_metrics, selected_metrics=None):
    """绘制各个团队的指标趋势图"""
    # 添加中文字体路径
    font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    font_prop = font_manager.FontProperties(fname=font_path)
    
    plt.figure(figsize=(15, 10))
    all_metrics = ['unique_ratio', 'novelty_ratio', 'high_score_ratio']
    all_labels = ['独特性比率', '新颖性比率', '高分比例']
    
    if selected_metrics:
        metrics = [m for m in all_metrics if m in selected_metrics]
        labels = [l for m, l in zip(all_metrics, all_labels) if m in selected_metrics]
    else:
        metrics = all_metrics
        labels = all_labels
    
    # 增加线条宽度和标记大小
    for metric, label in zip(metrics, labels):
        values = [team_metrics[team][metric] for team in sorted(team_metrics.keys()) if team != 1000]
        plt.plot(range(2, len(values) + 2), values, marker='o', label=label, linewidth=2.5, markersize=10)
    
    plt.xticks(range(2, len(values) + 2), fontsize=14)
    plt.yticks(fontsize=14)
    
    # 增加标签和标题的字体大小
    plt.xlabel('团队规模', fontproperties=font_prop, fontsize=16)
    plt.ylabel('指标值', fontproperties=font_prop, fontsize=16)
    plt.title('各团队评估指标对比', fontproperties=font_prop, fontsize=18)
    plt.legend(prop=font_prop, fontsize=14)
    plt.grid(True)
    
    # 保存图片时增加DPI以提高清晰度
    plt.savefig('/home/shuaichen/code/virtual_scientists/image/team_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    team2id = {
        8:[1,15,42,44,50],
        7:[2,3,12,18,21],
        6:[11,13,33,36,51],
        5:[4,7,28,29,41],
        4:[5,16,17,22,23],
        3:[25,26,37,45,64],
        2:[30,34,53,54,81],
        1:[47,471,472,89,131],
        # 1000:[1001,1002,1003,1004,1005],
    }
    
    # 存储所有团队的指标
    team_metrics = {}
    total_ideas = 0
    
    # 准备DataFrame的数据
    excel_data = []
    
    for team_id in team2id:
        results = []
        for paper_id in team2id[team_id]:
            base_path = f"/home/shuaichen/code/virtual_scientists/results/{paper_id}"
            evaluation = Evaluation(base_path, turn=0)
            result = evaluation.evaluate_ideas(paper_id)
            results.append(result)
            total_ideas += result['idea_num']
            
        # 计算平均值
        avg_metrics = {
            key: np.mean([r[key] for r in results]) 
            for key in ['idea_num', 'similarity_ratio', 'unique_ratio', 'novelty_ratio', 'high_score_ratio']
        }
        
        # 存储团队指标
        team_metrics[team_id] = avg_metrics
        
        # 添加到Excel数据
        excel_data.append({
            '团队ID': team_id,
            '平均想法数量': avg_metrics['idea_num'],
            '总想法数量': avg_metrics['idea_num'] * 5,
            '平均相似度比率': avg_metrics['similarity_ratio'],
            '平均独特性比率': avg_metrics['unique_ratio'],
            '平均新颖性比率': avg_metrics['novelty_ratio'],
            '高分想法比例': avg_metrics['high_score_ratio']
        })

        # 打印结果
        print(f"\n团队 {team_id} 的评估结果：")
        print(f"平均想法数量: {avg_metrics['idea_num']:.2f}")
        print(f"共生成{avg_metrics['idea_num']*5}个想法")
        print(f"平均相似度比率: {avg_metrics['similarity_ratio']:.2f}")
        print(f"平均独特性比率: {avg_metrics['unique_ratio']:.2f}")
        print(f"平均新颖性比率: {avg_metrics['novelty_ratio']:.2f}")
        print(f"平均得分≥5的想法比例: {avg_metrics['high_score_ratio']:.2%}")
    
    print(f"\n总共生成{total_ideas}个想法")
    
    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(excel_data)
    excel_path = '/home/shuaichen/code/virtual_scientists/evaluation/team_metrics_0.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"\nExcel文件已保存至: {excel_path}")
               