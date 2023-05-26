# Generative AI Paper Reading Collection :books:

This repository is a curated list of research papers on Generative AI that I've found particularly insightful or influential. The aim is to help those interested in this field to navigate the vast amount of literature available. 

Each entry includes the paper title, a brief summary of the paper's contributions, and a link to my personal reading notes.

## Table of Contents

- [Fine Tuning Large Language Model](#fine-tune)


<a name="#fine-tune"></a>
## Fine Tuning Large Language Models

- **[LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206.pdf)** | [Reading Notes](https://github.com/chuckhelios/generative-ai-papers/blob/main/large_language_models_fine_tuning/LIMA%3A%20Less%20Is%20More%20for%20Alignment.md)
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
