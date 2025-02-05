# **Chain-of-Thought**, **Tree of Thoughts**, and **Graph of Thoughts**

## **Description**
- *Chain-of-Thought* is a prompting technique that generates a series of intermediate steps for solving reasoning tasks.
- *Tree of Thoughts* is a prompting technique that extends the *Chain-of-Thought* prompting. Instead of using a linear reasoning path, it uses a tree structure to explore multiple different reasoning paths.
- *Graph of Thoughts* is a prompting technique extended from the *Chain-of-Thought* prompting and *Tree of Thoughts* prompting. By modeling the reasoning process as a grpah structure, a more complex network of thoughts can be generated.

## Example Prompts from Relevant Papers
### Example 1: Few-shot CoT
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)
    - Few-shot examplers for chain-of-thought prompting in math word problems
        ```
        Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
        Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
        A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
        Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
        A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
        Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
        A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
        Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
        A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
        Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
        A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
        Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
        A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
        Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
        A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
        Q: {question}
        ```

### Example 2: Zero-shot CoT
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916)
    ```
    Q: {question}
    A: Let's think step by step.
    ```

### Example 3: ToT
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601)
    - Tree-of-thoughts prompting in creative writing
        - cot_prompt
            ```
            Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}

            Make a plan then write. Your output should be of the following format:

            Plan:
            Your plan here.

            Passage:
            Your passage here.
            ```
        - vote_prompt
            ```
            Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
            ```
        - compare_prompt
            ```
            Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".
            ```
        - score_prompt
            ```
            Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.
            ```
            
### Example 4: GoT
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687)
    - Graph-of-thoughts prompting in document merging
        - merge prompt
            ```
            Merge the following 4 NDA documents <Doc1> - <Doc4> into a single NDA, maximizing retained information and minimizing redundancy. Output only the created NDA between the tags <Merged> and </Merged>, without any additional text.
            Here are NDAs <Doc1> - <Doc4>:
            <Doc1> {doc1} </Doc1>
            <Doc2> {doc2} </Doc2>
            <Doc3> {doc3} </Doc3>
            <Doc4> {doc4} </Doc4>
            ```
        - score prompt: 
            ```
            The following NDA <S> merges NDAs <Doc1> - <Doc4>.
            Please score the merged NDA <S> in terms of how much redundant information is contained, independent of the original NDAs, as well as how much information is retained from the original NDAs.
            A score of 10 for redundancy implies that absolutely no information is redundant, while a score of 0 implies that at least half of the information is redundant (so everything is at least mentioned twice).
            A score of 10 for retained information implies that all information from the original NDAs is retained, while a score of 0 implies that no information is retained.
            You may provide reasoning for your scoring, but the final score for redundancy should be between the tags <Redundancy> and </Redundancy>, and the final score for retained information should be between the tags <Retained> and </Retained>, without any additional text within any of those tags.
            Here are NDAs <Doc1> - <Doc4>:
            <Doc1> {doc1} </Doc1>
            <Doc2> {doc2} </Doc2>
            <Doc3> {doc3} </Doc3>
            <Doc4> {doc4} </Doc4>
            Here is the merged NDA <S>:
            <S> {s} </S>
        - aggregate prompt
            ```
            The following NDAs <S1> - <S{num_ndas_summaries}> each merge the initial NDAs <Doc1> - <Doc4>.
            Combine the merged NDAs <S1> - <S{num_ndas_summaries}> into a new one, maximizing their advantages and overall information retention, while minimizing redundancy.
            Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.
            Here are the original NDAs <Doc1> - <Doc4>:
            <Doc1> {doc1} </Doc1>
            <Doc2> {doc2} </Doc2>
            <Doc3> {doc3} </Doc3>
            <Doc4> {doc4} </Doc4>
            Here are the merged NDAs <S1> - <S{num_ndas_summaries}>:
            <S1> {s1} </S1>
            ...
            <S{num_ndas_summaries}> {s{num_ndas_summaries}} </S{num_ndas_summaries}>
            ```
        - improve prompt
            ```
            The following NDA <S> merges initial NDAs <Doc1> - <Doc4>.
            Please improve the merged NDA <S> by adding more information and removing redundancy. Output only the improved
            NDA, placed between the tags <Merged> and </Merged>, without any additional text.
            Here are NDAs <Doc1> - <Doc4>:
            <Doc1> {doc1} </Doc1>
            <Doc2> {doc2} </Doc2>
            <Doc3> {doc3} </Doc3>
            <Doc4> {doc4} </Doc4>
            Here is the merged NDA <S>:
            <S> {s} </S>
            ```

## Applications
### Example 1: CoT applied to a supervisor prompt in hierarchical multi-agent system
- SYSTEM: You are a supervisor.
  The goal of your team is to write a {report} on the theme requested by the user.
  You are responsible for managing collaboration between your {members}.

  1. First, break down the task into logical steps. Enclose your thought for why each step is necessary within <think></think> tags.
  2. For each step, allocate a suitable worker from your {members}.
  3. Your {members} will respond to their assigned tasks. Evaluate their responses. Are they accurate and relevant?
  4. If any part is incomplete or insufficient, provide feedback and request revisions. Each chapter must have more than 5 sentences.
  5. When the task is fully completed, respond with "FINISH".
  
  Ensure that the final {report} maintains a professional tone and meets high-quality standards.

### Example 2: ToT applied in math problem solving
```
Solve the math problem by exploring three different approaches.  
- Start by identifying three possible methods.  
- Solve step by step for each method.  
- If a method leads to an incorrect answer, go back and try a different approach.  
- Compare the final results and choose the best solution.  
```

### Example 3: GoT applied in developing plotlines for a story
```
Develop a story with interconnected plotlines using a **Graph of Thoughts (GoT)** approach.  
Structure the story as a network of nodes (key events, characters, themes) and edges (relationships, dependencies, influences).  

#### **Graph Structure:**
- **Nodes:** Represent major story elements (e.g., character arcs, conflicts, resolutions).
- **Edges:** Define how nodes influence and interact with each other.
- **Bidirectional Links:** Allow feedback loops for evolving plot coherence.

#### **Story Graph Definition:**
1. **Main Character’s Backstory (Node 1)**
   - Define key past events that shape motivations.
   - Connect to multiple future events through character choices.

2. **Inciting Incident (Edge 1)**
   - A pivotal event that propels the protagonist into action.
   - Forms an edge connecting past (Node 1) to the main plot trajectory.

3. **Secondary Character’s Subplot (Node 2)**
   - Develop a secondary character with independent goals.
   - Create multiple edges linking their story to both the protagonist and external conflicts.

4. **Intersection with Main Plot (Edge 2)**
   - Identify a key moment where the subplot critically alters the protagonist’s journey.
   - Allow reciprocal influence: the main plot may also change the secondary character’s arc.

5. **Thematic Reinforcement Nodes**
   - Define themes (e.g., redemption, ambition, betrayal).
   - Connect to character decisions and key events.

6. **Climax and Resolution (Node 3)**
   - Ensure multiple edges link back to character development arcs.
   - Introduce resolution pathways that emerge from prior interconnected nodes.

#### **Dynamic Refinement Using GoT:**
- Introduce **feedback loops** between nodes to refine coherence.
- Re-evaluate relationships: can new edges strengthen story depth?
- Allow non-linear evolution—if an event alters character motivation, dynamically update all affected nodes.
```

## **Related Usage**
- [17-LangGraph/03-Use-Cases/08-Hierarchical-Multi-Agent-Teams] (https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/tree/main/17-LangGraph/03-Use-Cases)

## **Reference**
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601)
- [Github: Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm)
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687)