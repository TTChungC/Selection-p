# Selection-p: Self-Supervised Task-Agnostic Prompt Compression for Faithfulness and Transferability

[![arXiv](https://img.shields.io/badge/arXiv-2410.11786-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2410.11786) [![Web](https://img.shields.io/badge/Web-Selection_p-blue.svg?style=plastic)](https://ttchungc.github.io/projects/selection-p/)

This is the official release accompanying our Findings of EMNLP 2024 paper:

> Selection-p: Self-Supervised Task-Agnostic Prompt Compression for Faithfulness and Transferability <br>
> [Tsz Ting Chung](https://ttchungc.github.io/), [Leyang Cui](https://nealcly.github.io/), [Lemao Liu](https://lemaoliu.github.io/), [Xinting Huang](https://timhuang1.github.io/), [Shuming Shi](https://shumingshi.github.io/), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung) 

## Example Usage
```
python eval.py
--compression_model_path ttchungc/selection-p \
--target_model_path MODEL_PATH \
--dataset wsc
```

## Citation

```bibtex
@article{chung2024selection,
  title={Selection-p: Self-Supervised Task-Agnostic Prompt Compression for Faithfulness and Transferability},
  author={Chung, Tsz Ting and Cui, Leyang and Liu, Lemao and Huang, Xinting and Shi, Shuming and Yeung, Dit-Yan},
  journal={arXiv preprint arXiv:2410.11786},
  year={2024}
}
```


