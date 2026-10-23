# Third-party resources

This project does not claim ownership of upstream models, native libraries, datasets or news articles. Source licensing and model/data licensing are separate matters; no new blanket license for this repository is declared here.

- **Qwen3-14B**: official [model license](https://huggingface.co/Qwen/Qwen3-14B/blob/main/LICENSE), Apache-2.0. The adapter is a fine-tuned derivative for this project; the base is downloaded separately from the official pinned Qwen GGUF repository. Preserve applicable upstream license and notices when distributing model derivatives.
- **llama.cpp**: [official project](https://github.com/ggml-org/llama.cpp) and its [license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE). The application downloads official native release assets separately; this source repository does not bundle those binaries.
- **XSum**: Narayan, Cohen and Lapata (EMNLP 2018), [official dataset repository](https://github.com/EdinburghNLP/XSum). Source news text originates from BBC articles. Dataset and article rights must not be treated as a license for this project's original code.
- **Demonstrations**: the three built-in UI sample articles and the README screenshot contain XSum/BBC source material. They are selected validation demonstrations, not original NeGm news reports or unbiased benchmark examples. Full training and evaluation article collections are excluded from source Git.
- **Python dependencies**: dependency licenses remain with their respective maintainers; see the separately pinned requirement files.

Citation: Shashi Narayan, Shay B. Cohen and Mirella Lapata. 2018. Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization. Proceedings of EMNLP 2018.
