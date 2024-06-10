# Fairness-Aware Structured Pruning in Transformers ([AAAI 2024](https://arxiv.org/pdf/2312.15398.pdf))
We show that certain attention heads are responsible for bias and pruning them improves fairness.

<div style="text-align: center">
<img src="FASP.png" width="400">
<p style="text-align: center;">  </p>
</div>


## How it works
The figure below illustrates how FASP is applied to a model with $6$ layers and $12$ heads per layer, e.g. DistilGPT-2. We identify and exclude the heads that significantly impact performance from the pruning process (black squares). Subsequently, the remaining heads are prioritized for removal
based on their contribution to bias, ensuring that the heads
contributing the most to bias are pruned first (red squares).
<div style="text-align: center">
<img src="FASP_figure.png" width="400">
<p style="text-align: center;">  </p>
</div>

## Running the experiments
Get started with the Colab tutorial, `FASP_AAAI24_reproducibility.ipynb`, which guides you through the process of downloading the models, understanding the preprocessing steps, and creating the scripts required to run the experiments. 

## Citation
```
@inproceedings{zayed2024fairness,
  title={Fairness-aware structured pruning in transformers},
  author={Zayed, Abdelrahman and Mordido, Gon{\c{c}}alo and Shabanian, Samira and Baldini, Ioana and Chandar, Sarath},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={20},
  pages={22484--22492},
  year={2024}
}


