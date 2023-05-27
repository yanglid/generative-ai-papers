# Paper Title: A Survey of Large Language Models

Paper link - https://arxiv.org/pdf/2303.18223.pdf
<!--
note link - https://github.com/chuckhelios/generative-ai-papers/blob/main/survey/A%20Survey%20of%20Large%20Language%20Models.md
Category - Survey
-->

## Summary
1. **Introduction**: The paper introduces the concept of Large Language Models (LLMs), their evolution, and their impact on various fields. It highlights the importance of LLMs in natural language understanding and generation tasks.
2. **Pre-training LLMs**: The paper discusses the process of pre-training LLMs, including data collection, model architecture, and optimization. It emphasizes the importance of diverse and high-quality data for pre-training and the role of transformer-based architectures in the success of LLMs.
3. **Adaptation Tuning**: The paper explores various methods for fine-tuning LLMs, such as prompt engineering, few-shot learning, and reinforcement learning from human feedback. These methods help in adapting the pre-trained LLMs to specific tasks or domains.
4. **Utilization of LLMs**: The paper discusses different ways to utilize LLMs, including zero-shot, few-shot, and many-shot learning. It also explores the concept of prompt engineering to elicit desired responses from the model.
5. **Evaluation of LLMs**: The paper delves into the evaluation of LLMs, discussing various evaluation tasks and settings. It also discusses the challenges in evaluating LLMs due to their complex nature and the diversity of tasks they can perform.
6. **Advanced Abilities of LLMs**: The paper explores three advanced abilities of LLMs: human alignment, interaction with the external environment, and tool manipulation. These abilities are crucial for the practical performance of LLMs.
7. **Public Benchmarks and Empirical Analysis**: The paper introduces several comprehensive benchmarks for evaluating LLMs, including MMLU, BIG-bench, and HELM. It also discusses comprehensive analyses conducted to investigate the strengths and limitations of LLMs.
8. **Conclusion and Future Directions**: The paper concludes by highlighting the key concepts, findings, and techniques for understanding and utilizing LLMs. It emphasizes the importance of pre-training, adaptation tuning, utilization, and evaluation in the success of LLMs.

In summary, the paper provides a comprehensive survey of Large Language Models, discussing their pre-training, fine-tuning, utilization, and evaluation. It highlights the advanced abilities of LLMs and the importance of comprehensive benchmarks for their evaluation. The paper concludes with a discussion on the future directions in the field of LLMs.

## Diagram
```mermaid
graph TB
  A["Data Collection"]
  B["Pre-training"]
  C["Adaptation Tuning"]
  D["Utilization"]
  E["Evaluation"]
  F["Advanced Abilities"]
  G["Public Benchmarks"]
  
  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
  F --> G
  
  subgraph B1["Pre-training"]
    B1A["Model Architecture"]
    B1B["Optimization"]
  end
  
  subgraph C1["Adaptation Tuning"]
    C1A["Prompt Engineering"]
    C1B["Few-shot Learning"]
    C1C["RL from Human Feedback"]
  end
  
  subgraph D1["Utilization"]
    D1A["Zero-shot Learning"]
    D1B["Few-shot Learning"]
    D1C["Many-shot Learning"]
    D1D["Prompt Engineering"]
  end
  
  subgraph E1["Evaluation"]
    E1A["Evaluation Tasks"]
    E1B["Evaluation Settings"]
  end
  
  subgraph F1["Advanced Abilities"]
    F1A["Human Alignment"]
    F1B["Interaction with External Environment"]
    F1C["Tool Manipulation"]
  end
  
  subgraph G1["Public Benchmarks"]
    G1A["MMLU"]
    G1B["BIG-bench"]
    G1C["HELM"]
    G1D["Comprehensive Analyses"]
  end
  
  B --> B1
  C --> C1
  D --> D1
  E --> E1
  F --> F1
  G --> G1
mermaid```

## Detail Notes

**Introduction**

The introduction provides an overview of the evolution of language modeling, from statistical language models (SLMs) to neural language models (NLMs), pre-trained language models (PLMs), and finally, large language models (LLMs). The authors highlight the significant impact LLMs are having on the AI community, with applications ranging from natural language processing (NLP) to information retrieval (IR) and computer vision (CV). They also note the challenges in understanding the underlying principles of LLMs, including why emergent abilities occur in LLMs and not smaller PLMs, the difficulty in training LLMs due to the high computational cost, and the challenge of aligning LLMs with human values or preferences.

**Statistical Language Models (SLM)**

SLMs are based on statistical learning methods that rose in the 1990s. They build the word prediction model based on the Markov assumption, predicting the next word based on the most recent context. However, they often suffer from the curse of dimensionality, making it difficult to accurately estimate high-order language models.

**Neural Language Models (NLM)**

NLMs use neural networks to model the probability of word sequences. They introduced the concept of distributed representation of words and built the word prediction function conditioned on the aggregated context features. This led to the use of language models for representation learning, beyond word sequence modeling.

**Pre-trained Language Models (PLM)**

PLMs were developed by pretraining Transformer models over large-scale corpora, showing strong capabilities in solving various NLP tasks. They often require fine-tuning for adapting to different downstream tasks. The "pre-training and fine-tuning" learning paradigm was set by these models.

**Large Language Models (LLM)**

LLMs are PLMs of significant size (e.g., containing tens or hundreds of billions of parameters). They display different behaviors from smaller PLMs and show surprising abilities in solving a series of complex tasks. For example, GPT-3 can solve few-shot tasks through in-context learning, whereas GPT-2 cannot do well.

The authors note that despite the progress and impact, the underlying principles of LLMs are still not well explored. They discuss the need for more attention on the research and development of LLMs and provide a literature review of the recent advances in LLMs from four major aspects, including pre-training, adaptation tuning, utilization, and capability evaluation.

**Background for LLMs**

LLMs refer to Transformer language models that contain hundreds of billions (or more) of parameters, trained on massive text data. They exhibit strong capacities to understand natural language and solve complex tasks via text generation. The authors introduce the basic background for LLMs, including scaling laws, emergent abilities, and key techniques.

- **Scaling Laws for LLMs**: The authors introduce two representative scaling laws for Transformer language models. The KM scaling law proposed by Kaplan et al. models the power-law relationship of model performance with respect to three major factors: model size (N), dataset size (D), and the amount of training compute (C). The Chinchilla scaling law proposed by Hoffmann et al. provides an alternative form for scaling laws to instruct the compute-optimal training for LLMs.
- **Emergent Abilities of LLMs**: Emergent abilities are defined as "the abilities that are not present in small models but arise in large models". The authors introduce three typical emergent abilities for LLMs: In-context learning, Instruction following, and Step-by-step reasoning.
- **Key Techniques for LLMs**: The authors list several important techniques that have led to the success of LLMs. These include Scaling, Training, and Ability eliciting.

**Alignment Tuning**

LLMs are trained to capture the data characteristics of pre-training corpora, which includes both high-quality and low-quality data. As a result, they may generate content that is toxic, biased, or even harmful. To align LLMs with human values, the authors discuss the approach of InstructGPT, which uses reinforcement learning with human feedback. This approach incorporates human judgment in the training loop with carefully designed labeling strategies. For example, ChatGPT, which is developed on a similar technique to InstructGPT, shows a strong alignment capacity in producing high-quality, harmless responses.

**Tools Manipulation**

LLMs are trained as text generators over massive plain text corpora, thus they perform less well on tasks that are not best expressed in the form of text, such as numerical computation. To tackle these issues, the authors discuss a technique that employs external tools to compensate for the deficiencies of LLMs. For example, LLMs can utilize a calculator for accurate computation and employ search engines to retrieve unknown information. More recently, ChatGPT has enabled the mechanism of using external plugins (existing or newly created apps), which can broadly expand the scope of capacities for LLMs.

**Technical Evolution of GPT-series Models**

The authors provide a detailed timeline of the evolution of the GPT-series models, from the early explorations with GPT-1 and GPT-2, to the capacity leap with GPT-3, and finally the capacity enhancement techniques applied to GPT-3. The authors discuss how training on code data and alignment with human preference have been used to further improve the GPT-3 model.

- **GPT-1**: The first GPT model was developed in 2018, based on a generative, decoder-only Transformer architecture. It established the underlying principle to model natural language text, i.e., predicting the next word.
- **GPT-2**: Following a similar architecture of GPT-1, GPT-2 increased the parameter scale to 1.5B and was trained with a large webpage dataset WebText. It sought to perform tasks via unsupervised language modeling, without explicit fine-tuning using labeled data.
- **GPT-3**: GPT-3 was released in 2020, which scaled the model parameters to an ever larger size of 175B. It introduced the concept of in-context learning (ICL), which utilizes LLMs in a few-shot or zero-shot way.
- **Training on code data**: To enhance the reasoning ability of GPT-3, Codex was introduced, which was a GPT model fine-tuned on a large corpus of GitHub code. It demonstrated that Codex can solve very difficult programming problems, and also lead to a significant performance improvement in solving math problems.
- **Human alignment**: The research of human alignment can be dated back to the year 2017 for OpenAI. A blog article entitled "learning from human preferences" was posted on the OpenAI blog describing a work that applied reinforcement learning (RL) to learn from the preference comparisons annotated by humans.

**Alignment Criteria**: The paper discusses three representative alignment criteria for LLMs: helpfulness, honesty, and harmlessness.

- **Helpfulness**: The LLM should assist users in solving their tasks or answering questions in a concise and efficient manner. It should also demonstrate the capability of eliciting additional relevant information through pertinent inquiries and exhibit suitable levels of sensitivity, perceptiveness, and prudence.
- **Honesty**: A LLM aligned to be honest should present accurate content to users instead of fabricating information. It should also convey appropriate degrees of uncertainty in its output, in order to avoid any form of deception or misrepresentation of information.
- **Harmlessness**: The language produced by the model should not be offensive or discriminatory. The model should be capable of detecting covert endeavors aimed at soliciting requests for malicious purposes.

The paper also discusses the collection of human feedback during the pre-training stage. High-quality human feedback is extremely important for aligning LLMs with human preferences and values. The paper discusses how to select a team of human labelers for feedback data collection and the different approaches to collecting feedback and preference data from human labelers.

**Reinforcement Learning from Human Feedback (RLHF)**: RLHF is a technique that has been widely used in recent powerful LLMs such as ChatGPT. RLHF employs reinforcement learning (RL) algorithms to adapt LLMs to human feedback by learning a reward model. The RLHF system mainly comprises three key components: a pre-trained LM to be aligned, a reward model learning from human feedback, and a RL algorithm training the LM.

**Efficient Tuning**: The paper discusses the approaches of instruction tuning and alignment tuning to adapt LLMs according to specific goals. Since LLMs consist of a huge amount of model parameters, it would be costly to perform the full-parameter tuning. The paper reviews several representative parameter-efficient fine-tuning methods for Transformer language models and summarizes existing work on parameter-efficient fine-tuned LLMs.

**In-Context Learning**

Several studies have shown that the effectiveness of In-Context Learning (ICL) is highly affected by the design of demonstrations. When ICL was introduced in the GPT-3's paper, it was originally defined to be a combination of the task description and demonstration examples, wherein either component is dispensable. Following this definition, when a Large Language Model (LLM) is required to solve an unseen task by using only task descriptions, it can be also considered to perform ICL for task solving, whereas the ICL ability can be enhanced by instruction tuning.

The demonstration design of ICL is introduced from three major aspects, i.e., demonstration selection, format, and order. The performance of ICL tends to have a large variance with different demonstration examples, so it is important to select a subset of examples that can effectively leverage the ICL capability of LLMs. There are two main demonstration selection approaches, namely heuristic and LLM-based approaches.

After selecting task examples, the next step is to integrate and format them into a natural language prompt for LLMs. A straightforward method is to instantiate a pre-defined template with the corresponding input-output pairs. To construct more informative templates, recent studies consider adding task descriptions or enhancing the reasoning capability of LLMs with chain-of-thought prompts.

LLMs are shown to sometimes suffer from the recency bias, i.e., they are prone to repeat answers that are near the end of demonstrations. Thus, it is important to arrange demonstrations (i.e., task examples) in a reasonable order. Early work proposes several heuristic methods to quickly find a good order. For example, demonstrations can be directly organized according to their similarity to the query in the embedding space: the more similar, the closer to the end.

After pre-training, LLMs can exhibit intriguing ICL capability without being updated. In what follows, we discuss two key questions about the ICL ability of LLMs, i.e., "how does pre-training affect the ICL ability" and "how do LLMs perform ICL during inference".

Chain-of-Thought (CoT) is an improved prompting strategy to boost the performance of LLMs on complex reasoning tasks, such as arithmetic reasoning, commonsense reasoning, and symbolic reasoning. Instead of simply constructing the prompts with input-output pairs as in ICL, CoT incorporates intermediate reasoning steps that can lead to the final output into the prompts.

Typically, CoT can be used with ICL in two major settings, namely the few-shot and zero-shot settings. Few-shot CoT is a special case of ICL, which augments each demonstration input, output as input, CoT, output by incorporating the CoT reasoning steps. To apply this strategy, we next discuss two key issues, i.e., how to design appropriate CoT prompts and how to utilize the generated CoTs for deriving the final answer.

Zero-shot CoT, different from few-shot CoT, does not include human-annotated task demonstrations in the prompts. Instead, it directly generates reasoning steps and then employs the generated CoTs to derive the answers.

**When does CoT work for LLMs?**

CoT is an emergent ability and it only has a positive effect on sufficiently large models (typically containing 10B or more parameters). It's mainly effective in improving tasks that require step-by-step reasoning, such as arithmetic reasoning, commonsense reasoning, and symbolic reasoning. However, for tasks that do not rely on complex reasoning, it might perform worse than standard prompting. Interestingly, the performance gain brought by CoT prompting could be significant only when standard prompting yields poor results.

**Why LLMs Can Perform CoT Reasoning?**

The source of CoT ability is widely hypothesized to be attributed to training on code since models trained on it show a strong reasoning ability. Code data is well organized with algorithmic logic and programming flow, which may be useful to improve the reasoning performance of LLMs. However, this hypothesis still lacks publicly reported evidence of ablation experiments (with and without training on code). Besides, instruction tuning seems not to be the key reason to obtain the CoT ability, since it has been empirically shown that instruction tuning on non-CoT data does not improve the performance on held-out CoT benchmarks.

The major distinction between CoT prompting and standard prompting is the incorporation of reasoning paths prior to the final answer. A recent study identifies three key components in CoT prompting, namely symbols, patterns, and text. It is shown that the latter two parts (i.e., patterns and text) are essential to the model performance, and removing either one would lead to a significant performance drop. However, the correctness of symbols and patterns does not seem critical. Further, there exists a symbiotic relationship between text and patterns: the text helps LLMs to generate useful patterns, and patterns aid LLMs to understand tasks and generate texts that help solve them.

In summary, CoT prompting provides a general yet flexible approach to eliciting the reasoning ability of LLMs. There are also some preliminary attempts that extend this technique to solve multimodal tasks and multilingual tasks. In addition to directly utilizing LLMs with ICL and CoT, some recent studies explore how to specialize the ability of LLMs towards specific tasks, which is called model specialization.

The next section of the paper is about Capacity Evaluation. It examines the effectiveness and superiority of LLMs, a surge of tasks and benchmarks have been leveraged for conducting empirical evaluation and analysis. The authors first introduce three types of basic evaluation tasks of LLMs for language generation and understanding, then present several advanced tasks of LLMs with more complicated settings or goals, and finally discuss existing benchmarks and empirical analyses.

**Basic Evaluation Tasks**

The authors mainly focus on three types of evaluation tasks for LLMs, i.e., language generation, knowledge utilization, and complex reasoning.

**Language Generation**

Existing tasks about language generation can be roughly categorized into language modeling, conditional text generation, and code synthesis tasks. Code synthesis is not a typical NLP task, but it's included for discussion because it can be directly solved by a number of LLMs (trained on code data) in a similar generation approach as natural language text.

**Knowledge Utilization**

Knowledge utilization is an important ability of intelligent systems to accomplish knowledge-intensive tasks (e.g., commonsense question answering and fact completion) based on supporting factual evidence. It requires LLMs to properly utilize the rich factual knowledge from the pretraining corpus or retrieve external data when necessary. In particular, question answering (QA) and knowledge completion have been two commonly used tasks for evaluating this ability.

**Closed-Book QA**

Closed-book QA tasks test the acquired factual knowledge of LLMs from the pre-training corpus, where LLMs should answer the question only based on the given context without using external resources. For evaluating this ability, there are several datasets that can be leveraged, including Natural Questions, Web Questions, and TriviaQA, where the accuracy metric is widely adopted. Empirical results have revealed that LLMs can perform well in this setting and even match the performance of state-of-the-art open-domain QA systems.

**Open-Book QA**

Unlike closed-book QA, in open-book QA tasks, LLMs can extract useful evidence from the external knowledge base or document collections, and then answer the question based on the extracted evidence. Typical open-book QA datasets have overlap with closed-book QA datasets, but they incorporate external data sources, e.g., Wikipedia. The metrics of accuracy and F1 score are widely used in open-book QA tasks for evaluation.

**Knowledge Completion**

In knowledge completion tasks, LLMs might be (to some extent) considered as a knowledge base, which can be leveraged to complete or predict the missing parts of knowledge units (e.g., knowledge triples). Such tasks can probe and evaluate how much and what kind of knowledge LLMs have learned from the pre-training data.

**Complex Reasoning**

Complex reasoning refers to the ability of understanding and utilizing supporting evidence or logic to derive conclusions or make decisions. According to the type of involved logic and evidence in the reasoning process, we consider dividing existing evaluation tasks into three major categories, namely knowledge reasoning, symbolic reasoning, and mathematical reasoning.

**Knowledge Reasoning**

The knowledge reasoning tasks rely on logical relations and evidence about factual knowledge to answer the given question. Existing work mainly uses specific datasets to evaluate the reasoning capacity of the corresponding type of knowledge, e.g., CSQA for commonsense knowledge reasoning and ScienceQA for science knowledge reasoning. In addition to the accuracy of the predicted results, existing work has also evaluated the quality of the generated reasoning process, via automatic metrics (e.g., BLEU) or human evaluation. Typically, these tasks require LLMs to perform step-by-step reasoning based on factual knowledge, until reaching the answer to the given question. To elicit the step-by-step reasoning ability, chain-of-thought (CoT) prompting strategy has been proposed for enhancing the complex reasoning capacity of LLMs.

**Human Alignment**

Human alignment is a key ability for the broad use of LLMs in real-world applications. It is desired that LLMs could well conform to human values and needs. To evaluate this ability, existing studies consider multiple criteria for human alignment, such as helpfulness, honesty, and safety. For helpfulness and honesty, adversarial question answering tasks (e.g., TruthfulQA) can be utilized to examine LLM's ability in detecting possible falsehood in the text. Furthermore, harmlessness can be also evaluated by several existing benchmarks, e.g., CrowS-Pairs and Winogender. Despite the automatic evaluation with the above datasets, human evaluation is still a more direct way to effectively test the human alignment ability of LLMs. OpenAI invites many experts in domains related to AI risks to evaluate and improve the behaviors of GPT-4 when encountering risky contents. Besides, for other aspects of human alignment (e.g., truthfulness), several studies propose to use specific instructions and devise annotation rules to guide the annotation process. Empirical studies have revealed that these strategies can greatly improve the human alignment ability of LLMs. For instance, after alignment tuning on data collected through interactions with experts, the incorrect behavior rate of GPT-4 can be largely reduced when it deals with sensitive or disallowed prompts. In addition, high-quality pre-training data can reduce the effort required for alignment. For instance, Galactica is potentially more harmless due to the less biased contents in the scientific corpus.

**Interaction with External Environment**

Besides standard evaluation tasks, LLMs have the ability to receive feedback from the external environment and perform actions according to the behavior instruction, e.g., generating action plans in natural language to manipulate agents. Such an ability is also emergent in LLMs that can generate detailed and highly realistic action plans, while smaller models (e.g., GPT-2) tend to generate shorter or meaningless plans. To test this ability, several embodied AI benchmarks can be used for evaluation, such as Virtual-Home, ALFRED, and BEHAVIOR. These benchmarks build a 3D simulator for household tasks such as cleaning and cooking, in which the agent can execute natural language actions generated by LLMs. Based on the generated action plans from LLMs, existing work either adopts the regular metrics (e.g., executability and correctness of the generated action plans) in the benchmark or directly conducts real-world experiments and measures the success rate, to evaluate such ability. Existing work has shown the effectiveness of LLMs in interacting with the external environment and generating accurate action plans. Recently, several improved methods have been proposed to enhance the interaction ability of LLMs, e.g., designing code-like prompts and providing real-world grounding.

**Tool Manipulation**

When solving complex problems, LLMs can turn to external tools if they determine it is necessary. By encapsulating available tools with API calls, existing work has involved a variety of external tools, e.g., search engine, calculator, and compiler, to enhance the performance of LLMs on several specific tasks. Recently, OpenAI has supported the use of plugins in ChatGPT, which can equip LLMs with broader capacities beyond language modeling. For example, the web browser plugin enables ChatGPT to access fresh information. Further, incorporating third-party plugins is particularly key for creating a prosperous ecosystem of applications based on LLMs.

**Evaluation Benchmarks**

Recently, several comprehensive benchmarks have been released for the evaluation of LLMs. These include MMLU, BIG-bench, and HELM.

- MMLU is a versatile benchmark for large-scale evaluation of multi-task knowledge understanding, covering a wide range of knowledge domains from mathematics and computer science to humanities and social sciences. The difficulties of these tasks vary from basic to advanced. GPT-4 has achieved a remarkable record in MMLU, significantly better than the previous state-of-the-art models.
- BIG-bench is a collaborative benchmark intended to probe existing LLMs from various aspects. It comprises 204 tasks that encompass a broad range of topics, including linguistics, childhood development, mathematics, commonsense reasoning, biology, physics, social bias, software development, and so on. By scaling the model size, LLMs can even outperform the average human performance under the few-shot setting on 65% of tasks in BIG-bench.
- HELM is a comprehensive benchmark that currently implements a core set of 16 scenarios and 7 categories of metrics. It is built on top of many prior studies, conducting a holistic evaluation of language models. Instruction tuning can consistently boost the performance of LLMs in terms of accuracy, robustness, and fairness in HELM.

**Comprehensive Analyses on LLMs' Capacities**

In addition to constructing large-scale evaluation benchmarks, a surge of studies have conducted comprehensive analyses to investigate the strengths and limitations of LLMs. These analyses are divided into two major aspects: generalist (general-purpose capacity) and specialist (domain-specific capacity).

- Generalist: Existing work has systematically evaluated the general capacities of LLMs, to explore their competences in a variety of different tasks or applications. These studies mainly focus on the newly emerged LLMs (e.g., ChatGPT and GPT-4) that have not been well investigated before. They evaluate the mastery level of LLMs in solving general tasks and their robustness against noises or perturbations.
- Specialist: As LLMs have been pre-trained on large-scale mixture-of-source corpora, they can capture rich knowledge from the pre-training data. Thus, LLMs are also employed as domain experts or specialists for specific areas. Recent studies have widely explored the use of LLMs for solving domain-specific tasks and evaluated the adaptation capacity of LLMs. They briefly discuss three representative domains receiving considerable attention from the research community, namely healthcare, education, and law.

The paper concludes by stating that they have reviewed the recent progress of large language models (LLMs), and introduced the key concepts, findings, and techniques for understanding and utilizing LLMs. They focus on the large-sized models (i.e., having a size larger than 10B) while excluding the contents of early pretrained language models (e.g., BERT and GPT-2) that have been well covered in the existing literature. In particular, their survey has discussed four important aspects of LLMs, i.e., pre-training, adaptation tuning, utilization, and evaluation. For each aspect, they highlight the techniques or findings that are key to the success of LLMs. Besides, they also summarize the available resources for developing LLMs and discuss important implementation guidelines for reproducing LLMs.