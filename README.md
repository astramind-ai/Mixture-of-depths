# Unofficial implementation for the paper "Mixture-of-Depths"


## Introduction
This is an unofficial implementation for the paper [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

## Currently supported models

| Model  | Supported? |
| ------------- | ------------- |
| Mistral  |  ✅ |
| Mixtral  |  ✅ |
| LLama  |  ✅ |
| LLama2  |  ✅ |
| LLama3  |  ✅ |
| Gemma  |  ✅ |
| BLOOMZ  |  ✅ |
| BLOOM  |  ✅ |
| DeepSeek  |  ✅ |
| Phi (1.5 & 2)  |  ✅ |
| Qwen2  |  ✅ |
| StarCoder2  |  ✅ |
| Qwen2-MoE  |  ❓ |
| Solar  |  ❓ |
| Baichuan  |  ❌ |
| ChatGLM3  |  ❌ |
| InternLM  |  ❌ |
| Olmo  |  ❌ |
| XVERSE  |  ❌ |
| Yi  |  ❌ |
| Yuan  |  ❌ |

## 💾 Installation
```bash
pip install mixture-of-depth
```
Both **Linux**, **Windows** and **MacOS** are supported.
## 🏁 Quick Start

### High-level API (tranformers-compatible)
```python
from transformers import AutoModelForCausalLM
from MoD import apply_mod_to_hf

# Initialize your model from an available hf model
model= AutoModelForCausalLM.from_pretrained("some-repo/some-model")
# Convert the model to include the mixture of depths layers
model = apply_mod_to_hf(model)
# train the model
# ...
# save the model
model.save_pretrained('some_local_directory')
```
### Loading the converted Model
To utilize the converted model, you will need to load the model from the AutoClass. Below is an example demonstrating how to load the model from a local directory:
```python
from MoD import AutoMoDModelForCausalLM

# Replace 'path_to_your_model' with the actual path to your model's directory
model = AutoMoDModelForCausalLM.from_pretrained('path_to_your_model')
```
## 🫱🏼‍🫲🏽 Contributing
We welcome contributions from the community, whether it's adding new features, improving documentation, or reporting bugs. Please refer to our contribution guidelines before making a pull request.

## 📜 License
This repo is open-sourced under the Apache-2.0 license.

## Citation
If you use our code in your research, please cite it using the following Bibtex entry:

```bibtex
@article{MoD2024,
  title={Unofficial implementation for the paper "Mixture-of-Depths"},
  author={AstraMind AI},
  journal={https://github.com/astramind-ai/Mixture-of-depths},
  year={2024}
}
```
## Support
For questions, issues, or support, please open an issue on our GitHub repository.
