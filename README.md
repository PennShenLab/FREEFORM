## FREEFORM: Free-flow Reasoning and Ensembling for Enhanced Feature Output and Robust Modeling

This repository holds the official code for the paper [Knowledge-Driven Feature Selection and Engineering for Genotype Data with Large Language Models](http://arxiv.org/abs/2410.01795).

![alt text](https://github.com/PennShenLab/FREEFORM/raw/main/figure1.png?raw=true)

### 🎯 Abstract
Predicting phenotypes with complex genetic bases based on a small, interpretable set of variant features remains a challenging task. Conventionally, data-driven approaches are utilized for this task, yet the high dimensional nature of genotype data makes the analysis and prediction difficult. Motivated by the extensive knowledge encoded in pre-trained LLMs and their success in processing complex biomedical concepts, we set to examine the ability of LLMs in feature selection and engineering for tabular genotype data, with a novel knowledge-driven framework. We develop FREEFORM, Free-flow Reasoning and Ensembling for Enhanced Feature Output and Robust Modeling, designed with chain-of-thought and ensembling principles, to select and engineer features with the intrinsic knowledge of LLMs. Evaluated on two distinct genotype-phenotype datasets, genetic ancestry and hereditary hearing loss, we find this framework outperforms several data-driven methods, particularly on low-shot regimes. FREEFORM is available as open-source framework at GitHub.

### 📝 Requiremnets
The algorithm is implemented in Python. 
To install the related packages, use
```bash
pip install -r requirements.txt
```

### 🔨 Usage
```cmd
python 
```

### 🤝 Acknowledgements
This work was supported in part by the NIH grants U01 AG066833, U01 AG068057, R01 AG071470, U19 AG074879, and S10 OD023495.

### 📭 Maintainers
- Joseph Lee (jojolee@seas.upenn.edu)
- Shu Yang (shu.yang@pennmedicine.upenn.edu)

### 📚 Citation

```
@article{FreeForm,
      title={Knowledge-Driven Feature Selection and Engineering for Genotype Data with Large Language Models}, 
      author={Joseph Lee and Shu Yang and Jae Young Baik and Xiaoxi Liu and Zhen Tan and Dawei Li and Zixuan Wen and Bojian Hou and Duy Duong-Tran and Tianlong Chen and Li Shen},
      year={2024},
      journal={arXiv preprint arXiv:2410.01795},
}
```
