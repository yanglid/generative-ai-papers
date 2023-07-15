# Generative AI Paper Reading Collection :books:

This repository is a curated list of research papers on Generative AI that I've found particularly insightful or influential. The aim is to help those interested in this field to navigate the vast amount of literature available. 

Each entry includes the paper title, a brief summary of the paper's contributions, and a link to my personal reading notes.

## Table of Contents

- [Foundattion](#foundation)
- [Survey](#survey)
- [Fine Tuning](#fine-tune)

-----

<a name="#foundation"></a>
## Foundation

- **[Attention Is All You Need](https://1drv.ms/b/s!AlzIBZwjyb1_gbJYLQK3BzDa36Lvgw?e=laPgAo)** | [Reading Notes](./fundation/Attention%20is%20All%20You%20Need.md)
  <details>
  <summary>Summary</summary>
  
  1. **Abstract and Introduction:** The paper introduces the "Transformer", a novel model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer is proposed as a solution to the problem of long-range dependencies in sequence transduction tasks, which is a limitation of sequence-to-sequence models based on RNNs and CNNs.
  2. **Background:** The authors provide a brief overview of sequence transduction, recurrent neural networks, and the attention mechanism, which are the foundational concepts for their work.
  3. **Model Architecture:** The Transformer model consists of an encoder and decoder, each composed of a stack of identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. Residual connections and layer normalization are employed around each of the two sub-layers.
  4. **Attention:** The authors describe the scaled dot-product attention and multi-head attention mechanisms used in their model. The attention function is used to compute a weighted sum of values based on the dot product of the query and key.
  5. **Position-wise Feed-Forward Networks:** Each of the layers in the encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.
  6. **Embeddings and Softmax:** The model uses learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model. The same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation.
  7. **Positional Encoding:** Since the model doesn't contain recurrence or convolution, positional encodings are added to the input embeddings to indicate the position of the words in the sequence.
  8. **Why Self-Attention:** The authors discuss the advantages of the self-attention mechanism, such as its ability to handle long-range dependencies, its parallelizability, and its capacity to model various types of dependencies.
  9. **Training:** The authors describe the training process, including the use of residual dropout, label smoothing, and a custom learning rate scheduler.
  10. **Results:** The Transformer model achieves state-of-the-art performance on the WMT 2014 English-to-German and English-to-French translation tasks. The authors also conduct an ablation study to understand the importance of different components of the model.
</details>


- **[Title 2](paper-link-2)** | [Reading Notes](notes-link-2)
  <details>
  <summary>Summary</summary>

  Detailed summary of the paper and its contributions.

  </details>


<a name="#survey"></a>
## Survey
- **[A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)** | [Reading Notes](./survey/A%20Survey%20of%20Large%20Language%20Models.md)
  <details>
  <summary>Summary</summary>

  This paper provides a comprehensive survey of Large Language Models (LLMs), discussing their pre-training, fine-tuning, utilization, and evaluation. Key points include:

  1. **Introduction**: The paper introduces the concept of LLMs, their evolution, and their impact on various fields.
  2. **Pre-training LLMs**: Discusses the process of pre-training LLMs, including data collection, model architecture, and optimization.
  3. **Adaptation Tuning**: Explores various methods for fine-tuning LLMs, such as prompt engineering, few-shot learning, and reinforcement learning from human feedback.
  4. **Utilization of LLMs**: Discusses different ways to utilize LLMs, including zero-shot, few-shot, and many-shot learning.
  5. **Evaluation of LLMs**: Delves into the evaluation of LLMs, discussing various evaluation tasks and settings.
  6. **Advanced Abilities of LLMs**: Explores three advanced abilities of LLMs: human alignment, interaction with the external environment, and tool manipulation.
  7. **Public Benchmarks and Empirical Analysis**: Introduces several comprehensive benchmarks for evaluating LLMs, including MMLU, BIG-bench, and HELM.
  8. **Conclusion and Future Directions**: Concludes by highlighting the key concepts, findings, and techniques for understanding and utilizing LLMs.

  </details>

- **[Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/pdf/2306.08302v1.pdf)** | [Reading Notes](./survey/Unifying%20Large%20Language%20Models%20and%20Knowledge%20Graphs%3A%20A%20Roadmap.md)
  <details>
  <summary>Summary</summary>

  1. **Introduction**: The paper discusses the integration of Large Language Models (LLMs) and Knowledge Graphs (KGs) to leverage the strengths of both. LLMs excel in understanding and generating human-like text, while KGs provide structured and factual knowledge.
  2. **KGs for LLMs**: The authors discuss how KGs can be used to enhance LLMs. This can be done through pre-training, where KGs are used to generate training data for LLMs, or through KG-enhanced inference, where KGs are used during the inference stage to guide the LLM's responses.
  3. **LLMs for KGs**: The authors discuss how LLMs can be used to enhance KGs. This includes using LLMs to generate embeddings for KGs, for joint text and KG embedding, for KG completion, and for KG construction. The authors also discuss the use of LLMs for KG-to-text generation and KG question answering.
  4. **Synergized LLMs + KGs**: The authors discuss the synergy of LLMs and KGs, which combines the merits of both to enhance performance in various downstream applications. This includes knowledge representation and reasoning.
  5. **Future Directions**: The authors discuss several future directions for this research area, including using KGs for hallucination detection in LLMs, editing knowledge in LLMs, injecting knowledge into black-box LLMs, using multi-modal LLMs for KGs, developing LLMs that understand KG structure, and synergizing LLMs and KGs for bidirectional reasoning.
  6. **Conclusion**: The authors conclude that unifying LLMs and KGs is an active research direction that has attracted increasing attention. They hope that their overview of the recent research in this field can provide a comprehensive understanding and advance future research.
  
  </details>



<a name="#fine-tune"></a>
## Fine Tuning Large Language Models

- **[LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206.pdf)** | [Reading Notes](./large_language_models_fine_tuning/LIMA%3A%20Less%20Is%20More%20for%20Alignment.md)
  <details>
  <summary>Summary</summary>

  - **Abstract & Introduction**: The authors propose LIMA, a 65B parameter language model fine-tuned on only 1,000 carefully curated prompts and responses, suggesting that most knowledge in large language models is learned during pretraining.
  - **Alignment Data & Training LIMA**: They collect a dataset of 1,000 prompts and responses for fine-tuning LIMA, introducing a special end-of-turn token (EOT) to differentiate between each speaker.
  - **Human Evaluation & Experiment Setup**: LIMA is evaluated against state-of-the-art language models, outperforming OpenAI's RLHF-based DaVinci003 and a 65B-parameter reproduction of Alpaca trained on 52,000 examples.
  - **Results & Analysis**: Despite training on 52 times more data, Alpaca 65B tends to produce less preferable outputs than LIMA. Bard produces better responses than LIMA 42% of the time.
  - **Multi-turn Dialogue**: LIMA responses are surprisingly coherent for a zero-shot chatbot, but in 6 out of 10 conversations, LIMA fails to follow the prompt within 3 interactions.
  - **Discussion**: The authors show that fine-tuning a strong pretrained language model on 1,000 carefully curated examples can produce remarkable, competitive results. However, there are limitations to this approach, including the mental effort in constructing such examples and the robustness of LIMA.

  </details>


- **[Title 2](paper-link-2)** | [Reading Notes](notes-link-2)
  <details>
  <summary>Summary</summary>

  Detailed summary of the paper and its contributions.

  </details>




## Contributing

Contributions are welcome! If you would like to add a paper, please submit a pull request. Ensure that your submission includes the paper title, a brief summary, and a link to your reading notes.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
