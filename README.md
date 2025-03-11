# Generative Context Distillation

This repository is the official implementation of [Generative Context Distillation](https://arxiv.org/abs/2411.15927).

<img src="assets/main.png">

## Overview
Generative Context Distillation (GCD) is a lightweight LLM prompt internalization method, 
enabling efficient/effective inference without explicit prompts by joint training with Prompt Generation Loss.

## Components for Prompt Internalization
- Pseudo User Input
  - `src/make_pseudo-input.py`
- Pseudo Conversational Outputs
  - `src/make_pseudo-conv.py`
- Reason
  - `src/make_pseudo-output.py`
  - `src/make_reason.py`

## Training
Please refer to scripts directory for each task setup.
```sh
cd scripts/{task}
bash finetune_meta-cog_joint_loss.sh
```

## Citation
```
@misc{shin2025generativepromptinternalization,
      title={Generative Prompt Internalization}, 
      author={Haebin Shin and Lei Ji and Yeyun Gong and Sungdong Kim and Eunbi Choi and Minjoon Seo},
      year={2025},
      eprint={2411.15927},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.15927}, 
}
```
