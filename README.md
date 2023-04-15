# Superhuman-Eval

The project for [**Humanly Certifying Superhuman Classifiers**](https://openreview.net/forum?id=X5ZMzRYqUjB) (ICLR 2023, spotlight).

### Installation
**Requirement**: PyTorch, Numpy, etc.
**Download**:
```
git clone https://github.com/xuqiongkai/Superhuman-Eval.git
```

### Evaluation
**Step 0:** The evaluation requires a dataset with $N$ samples, each with multiple annotations and a predicted label by a classifier.
**Step 1 [Calculate Upper Bound $\mathcal U_N$]:** the accuracy between annotators, basically inter-annotator agreement.
**Step 2 [Calculate Lower Bound $\mathcal L_N$]:** the 'traditional' accuracy, by matching classifier's outputs and aggregated human annotations (voting is recommanded).
**Step 3 [Calculate the Confidence Score]:** run our script.
```
python fsa_example.py
```

### Citation

```bibtex
@inproceedings{
  xu2023humanly,
  title={Humanly Certifying Superhuman Classifiers},
  author={Qiongkai Xu and Christian Walder and Chenchen Xu},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=X5ZMzRYqUjB}
}

```
[We](qiongkai.xu@unimelb.edu.au) would be pleased to learn about your research if you have analysed and observed any *'superhuman'* classification models.