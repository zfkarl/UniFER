# UniFER: Rethinking Facial Expression Recognition in the Era of Multimodal Large Language Models

![VQA](https://img.shields.io/badge/Task-VQA-red) 
![Facial Expression Recognition](https://img.shields.io/badge/Task-FER-red) 
![Emotion Reasoning](https://img.shields.io/badge/Task-Emotion_Reasoning-red) 
![UniFER-7B](https://img.shields.io/badge/Model-UniFER--7B-green) 

<p align="center">
    <img src="./figs/logo.png" width="100%" height="100%">
</p>


🌟 Official repository for the paper "Rethinking Facial Expression Recognition in the Era of Multimodal Large Language Models"

[📖 Paper] [[🤗 Dataset](https://huggingface.co/datasets/Karl28/UniFER)] [[🤗 Model](https://huggingface.co/Karl28/UniFER-7B)]

## 👀 About UniFER

Multimodal Large Language Models (MLLMs) have revolutionized numerous research fields, including computer vision and affective computing. As a pivotal challenge in this interdisciplinary domain, facial expression recognition (FER) has evolved from separate, domain-specific models to more unified approaches. One promising avenue to unify FER tasks is converting conventional FER datasets into visual question-answering (VQA) formats, enabling the direct application of powerful generalist MLLMs for inference. However,
despite the success of cutting-edge MLLMs in various tasks, their performance on FER tasks remains largely unexplored. To address this gap, we provide FERBench, a systematic benchmark that incorporates 20 state-of-the-art MLLMs across four widely used FER datasets. Our results reveal that, while MLLMs exhibit good classification performance, they still face significant limitations in reasoning and interpretability.

<p align="center">
    <img src="figs/ferbench.png" width="90%"> <br>
</p>

To this end, we introduce post-training strategies aimed at enhancing the facial expression reasoning capabilities of MLLMs. Specifically, we curate two high-quality and large-scale datasets: UniFER-CoT-230K for cold-start initialization and UniFER-RLVR-360K for reinforcement learning with verifiable rewards (RLVR), respectively. Building upon them, we develop a unified and interpretable FER foundation model termed UniFER-7B, which outperforms many open-sourced and closed-source generalist MLLMs (e.g., Gemini-2.5-Pro and Qwen2.5-VL-72B).

<p align="center">
    <img src="figs/unifer_framework.png" width="90%"> <br>
</p>

## 🔥 Datasets

Our curated datasets include four widely-used FER datasets: RAFDB, FERPlus, AffectNet, and SFEW2.0. Please download the images of these datasets from their websites first.

### Installation

Clone the repository:

```
git clone https://github.com/zfkarl/UniFER.git
cd UniFER
```

Create a conda environment:

```
conda create -n r1-v python=3.11
conda activate r1-v
```

Please follow the official instructions [here](https://github.com/StarsfieldAI/R1-V) to install both PyTorch and additional dependencies.

### FERBench

The proposed four subsets of FERBench are stored in the following json files:
```bash
eval_rafdb/data/rafdb_qa.json
eval_ferplus/data/ferplus_qa.json
eval_affectnet/data/affectnet_qa.json
eval_sfew_2.0/data/sfew_2.0_qa.json
```

### UniFER-CoT-230K

Download our [dataset](https://huggingface.co/datasets/Karl28/UniFER), and put the json file `UniFER_CoT_230K.json` in:
```bash
data/UniFER_CoT_230K.json
```

### UniFER-RLVR-360K

Download our [dataset](https://huggingface.co/datasets/Karl28/UniFER), and put the json file `UniFER_RLVR_360K.json` in:
```bash
data/UniFER_RLVR_360K.json
```

## 🚀 Training

### Stage 1: Cold Start SFT

```bash
cd train_unifer/src/scripts
bash run_sft_fer.sh
```

### Stage 2: RLVR GRPO Training

```bash
cd train_unifer/src/scripts
bash run_grpo_vllm.sh
```

## 💫 Evaluation

After the above two-stage post-training, we can subsequently employ the derived model UniFER-7B for inference and evaluate its performance. You may change the directory name `Qwen2.5-VL-7B-FER-GRPO-VLLM-8GPU` to `UniFER-7B` for inference. Also, you can directly download our provided [checkpoints](https://huggingface.co/Karl28/UniFER-7B) for inference.

### Inference and Evaluation

On RAFDB:
```bash
cd eval_rafdb/code
python infer_unifer.py 
python eval_unifer.py
```

On FERPlus:
```bash
cd eval_ferplus/code
python infer_unifer.py 
python eval_unifer.py
```

On AffectNet:
```bash
cd eval_affectnet/code
python infer_unifer.py 
python eval_unifer.py
```

On SFEW2.0:
```bash
cd eval_sfew_2.0/code
python infer_unifer.py 
python eval_unifer.py
```

Overall Performance:
```bash
cd eval_total/code
python eval_unifer.py
```


<!--
## :white_check_mark: Citation

If you find **UniFER** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{zhang2024mathverse,
  title={MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?},
  author={Zhang, Renrui and Jiang, Dongzhi and Zhang, Yichi and Lin, Haokun and Guo, Ziyu and Qiu, Pengshuo and Zhou, Aojun and Lu, Pan and Chang, Kai-Wei and Gao, Peng and others},
  journal={arXiv preprint arXiv:2403.14624},
  year={2024}
}
```
-->

🔥 Please contact `fzhang@link.cuhk.edu.hk` if you would like to contribute to the leaderboard or have any problems.