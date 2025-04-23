# -*- coding: utf-8 -*-
"""Used to record prompts, will be replaced by configuration"""

class Prompts:
    
    prompt_existing_idea = "Here is the seed idea : '''{}'''\n"

    prompt_bad_review = "Here are the bad reviews of the seed idea, please improve according to the reviews: '''{}'''\n"

    prompt_task = """
    You are an ambitious scientist who is looking to propose a new idea that will contribute significantly to the field.
    Improve the seed idea or come up with the next impactful and creative idea for publishing a paper that will contribute significantly to the field by integrating your own knowledge and insights with the information provided.
    """+"\n"
    
    
    prompt_reference = """You may refer to the following listed references to design a new idea or concept. 
    These references can serve as inspiration, but you are not allowed to directly copy or replicate their content. 
    Ensure that your design is original and addresses a specific problem or meets a unique need. 
    References: {}\n
    """

    prompt_novelty = """
    Whether the idea is creative and different from existing works on the topic, and brings fresh insights. You are encouraged to search for related works online. You should consider all papers that appeared online prior to 2024 as existing work when judging the novelty. A rating from 1 to 10.Here are the grading rules:
    1. Not novel at all - there are many existing ideas that are the same 
    2. 
    3. Mostly not novel - you can find very similar ideas 
    4. 
    5. Somewhat novel - there are differences from existing ideas but not enough to turn into a new paper 
    6. Reasonably novel - there are some notable differences from existing ideas and probably enough to turn into a new paper 
    7. 
    8. Clearly novel - major differences from all existing ideas 
    9. 
    10. Very novel - very different from all existing ideas in a very interesting and clever way
    """
    prompt_novelty_rationale = """
    Short justification for your score. (Your rationale should be at least 2-3 sentences.)
    """

    prompt_feasibility = """
    How feasible it is to implement and execute this idea as a research project? Specifically, how feasible the idea is for a typical CS PhD student to execute within 1-2 months of time. You can assume that we have rich API resources, but only limited hardware resources. A rating from 1 to 10.Here are the grading rules:
    1. Impossible: the idea doesn’t make sense or the proposed experiments are flawed and cannot be implemented 
    2. 
    3. Very challenging: there are flaws in the proposed method or experiments, or the experiments require compute/human resources beyond any academic lab 
    4. 
    5. Moderately feasible: It can probably be executed within the given time frame but would require careful planning, efficient use of APIs or some advanced computational strategies to overcome the limited GPU resources, and would require some modifications to the original proposal to make it work 
    6. Feasible: Can be executed within the given constraints with some reasonable planning 
    7. 
    8. Highly Feasible: Straightforward to implement the idea and run all the experiments 
    9. 
    10. Easy: The whole proposed project can be quickly executed within a few days without requiring advanced technical skills
    """
    prompt_feasibility_rationale = """Short justification for your score. If you give a low score, you should specify what parts are difficult to execute and why. (Your rationale should be at least 2-3 sentences.)
    """

    prompt_excitement = """How exciting and impactful this idea would be if executed as a full project. Would the idea change the field and be very influential. A rating from 1 to 10.Here are the grading rules:
    1. Poor: You cannot identify the contributions of this idea, or it’s not interesting at all and you would fight to have it rejected at any major AI conference 
    2. 
    3. Mediocre: this idea makes marginal contributions and is very incremental 
    4. 
    5. Leaning negative: it has interesting bits but overall not exciting enough 
    6. Learning positive: exciting enough to be accepted at a major AI conference, but still has some weaknesses or somewhat incremental 
    7. 
    8. Exciting: would deepen the community’s understanding or make major progress in this research direction 
    9. 
    10. Transformative: would change the research field profoundly and worth a best paper award at major AI conferences
    """
    prompt_excitement_rationale = """Short justification for your score. (Your rationale should be at least 2-3 sentences.)"""


    prompt_response = ("""
    "Please respond in the following format: 

    Thought: <THOUGHT> 

    ```json<JSON>```

    In <THOUGHT>, briefly discuss your intuitions and motivations for the idea. Justify how this idea differs from existing ones, highlighting its unique aspects.

    In <JSON>, provide the new idea with the following fields and provide as many details as possible: 
    - "Idea": A detailed description of the idea, outlining its significance and potential impact.
    - "Title": A title for the idea, will be used for the paper writing. 
    - "Experiment": An outline of the implementation process. Describe your high-level design plan, including necessary design steps and the ideal outcomes of the experiments.
    - “Excitement": {}
    - "Excitement Rationale": {}
    - "Feasibility": {}
    - "Feasibility Rationale": {}
    - "Novelty": {}
    - "Novelty Rationale": {}
    
    Be cautious and realistic on your ratings. This JSON will be automatically parsed, so ensure the format is precise, and the content should be longer than 600 words. You only need to output one idea.
    """)

    scientific_discovery_theory = """
        1. Define New Scientific: Problems Theoretical Basis: Kuhn’s paradigm theory, Laudan’s problem-solving model, Nichols’s problem-generation theory. Method: Identify anomalies in existing theories; explore theoretical boundaries and scope of application; integrate interdisciplinary knowledge and discover new problems; re-examine neglected historical problems. 
        2. Propose New Hypotheses: Theoretical Basis: Pierce’s hypothetical deduction method, Weber’s theory of accidental discovery, Simon’s scientific discovery as problem solving. Method: Analogical reasoning; thought experiment; intuition and creative leaps; reductio ad absurdum thinking. 
        3. Exploring the Limitations and Shortcomings of Current Methods: Theoretical Basis: Popper’s falsificationism, Lakatos’s research program methodology, Feyerabend’s methodological anarchism. Method: Critically analyze existing methods; find deviations between theoretical predictions and experimental results; explore the performance of methods under extreme conditions; interdisciplinary comparative methodology. 
        4. Design and Improve Existing Methods: Theoretical Basis: Laudan’s methodological improvement model, Ziemann’s creative extension theory, Hacking’s experimental system theory. Method: Integrate new technologies and tools; improve experimental design and control; improve measurement accuracy and resolution; develop new data analysis methods. 
        5. Abstract and Summarize the General Laws Behind Multiple Related Studies: Theoretical Basis: Whewell’s conceptual synthesis theory, Carnap’s inductive logic, Glaser and Strauss’s grounded theory. Method: Comparative analysis of multiple case studies; identify common patterns and structures; construct conceptual frameworks and theoretical models; formal and mathematical descriptions.
        6. Construct and Modify Theoretical Models: Theoretical Basis: Quine’s holism, Lakoff’s conceptual metaphor theory, Kitcher’s unified theory of science. Method: Form a balance between reductionism and emergence; develop an interdisciplinary theoretical framework; mathematical modeling and computer simulation; theoretical simplification and unification. 
        7. Designing Critical Experiments: Theoretical Basis: Duhem-Quine thesis, Bayesian experimental design theory, Mayo’s experimental reasoning theory. Method: Designing experiments that can distinguish competing theories; exploring extreme conditions and boundary cases; developing new observation and measurement techniques; designing natural experiments and quasi-experiments. 
        8. Explaining and Integrating Anomalous Findings: Theoretical Basis: Hansen’s theory of anomalous findings, Sutton’s model of scientific serendipity, Kuhn’s theory of crises and revolutions. Method: Revisiting basic assumptions; developing auxiliary hypotheses; exploring new explanatory frameworks; integrating multidisciplinary perspectives. 
        9. Evaluating and Selecting Competing Theories: Theoretical Basis: Reichenbach’s confirmation theory, Sober’s theory selection criteria, Laudan’s problem-solving progress assessment. Method: Comparing theories for explanatory power and predictive power; evaluating the simplicity and elegance of theories; considering the heuristics and research agenda of theories; weighing the empirical adequacy and conceptual coherence of theories. 
        10. Scientific Paradigm Shift: Theoretical Basis: Kuhn’s theory of scientific revolutions, Toulmin’s model of conceptual evolution, Hall’s dynamic system theory. Method: Identify accumulated anomalies and crises; develop new conceptual frameworks; reinterpret and organize known facts; establish new research traditions and practices.
        """
    
    scientific_discovery_prompt = """
        Follow the steps below to generate a idea:
        1. Understanding of the target paper and related papers is essential: 
            - The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research idea. 
            - The referenced papers are studies that the target paper has cited, indicating their direct relevance and connection to the primary research topic you are focusing on, and providing additional context and ideas that are essential for understanding and expanding upon the target paper. 
        2. Understanding of the science discovery theories is essential: You need to select appropriate theories and combine the information provided by the current paper to come up with creative, influential, and feasible ideas. 
        3. Here are 10 general laws and methodologies of scientific discovery from the perspective of the philosophy of science. 
        {} You can choose one or more of these methodologies and propose new a scientific research idea for the target paper: 

        Requirements:
        1. Output a idea worth exploring.
        2. You should aim for new research ideas that can potentially win best paper awards at top conferences like ACL, NeurIPS, ICLR, and CVPR.
        3. Skip the research theories that may not well match the target paper; the theory and method you use should make sense and be reasonable for the target paper.
        4. Thinking is your thinking process. Please explain which theory you used in your thinking process.
        5. Please output your thought process.
        6. Please think step by step.

        Input:
        target_paper: {}
        referenced_pa​​pers: {}

        Please respond in the following format: 

        Thought: <THOUGHT> 

        IDEA: ```json<JSON>```

        In <THOUGHT>, output your thinking process here, explain why you chose these theories to discover new ideas and why it should have a chance to win the best paper awards at top conferences. 
        In <JSON>, provide the new idea with the following fields and provide as many details as possible:
        - "Title": A title for the idea, will be used for the paper writing. 
        - "Idea": A detailed description of the idea, outlining its significance and potential impact.
        - "Thinking": A detailed description of the thinking process,indicate from which scientific discovery theory
        - "Rationale": A detailed description of the rationale for the idea.
        This JSON will be automatically parsed, so ensure the format is precise.
        """
    
    search_keyword_sys_prompt = "You are now a senior scholar in the field of artificial intelligence, and you need to develop a search strategy based on a given research idea."
    
    search_keyword_prompt = """
        According to a given research idea, analyze which fields of papers should be retrieved in order to collect comprehensive information and new knowledge to provide a basis for further research and finding new ideas. Please provide the thinking process and search keywords.
        Here is the research idea: {}

        Please respond in the following format: 

        Thought: <THOUGHT> 
        Search Plans: ```json<JSON>```

        In <THOUGHT>, explain why do you develop such a search strategy?. 
        In <JSON>, Provide a search plan and provide as much detail as possible:
        - "Search Plan": The search plan for the given research idea.
        - "Search Fields": The list of fields that should be retrieved.
        - "Search_Keywords": The list of keywords for each field (maximum 3 keywords per field). The format is 'Search_Keywords':[{{"Field":"field_name","Keywords":["keyword1","keyword2","keyword3"]}}]
        Be cautious and realistic on your ratings. This JSON will be automatically parsed, so ensure the format is precise and each field has no more than 3 keywords. """
