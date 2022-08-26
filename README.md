# OTMISC: Our Topic Modeling Is Super Cool

<div align="center">
  <img src="./docs/Images/logo.jpg" width="25%">
  <p>An advanced topic modeling tool that can do many things!</p>
</div>

_________________________________________________________

## Introduction

This project is developed by Computer Science and Mathematics master students at TUM (Technical University of Munich)
for the course "Master's Practical Course - Machine Learning for Natural Language Processing Applications"
in SS22 (Summer Semester 2022).
Since this project is still in its infancy, we suggest those who want to use this project to be careful.

* **Project Advisors:**
    * PhD Candidate (M.Sc.) Miriam Anschütz
    * PhD Candidate (M.Sc.) Ahmed Mosharafa
* **Project Scope:**
    * Evaluating different Topic Modeling algorithms on short/long text dataset.
    * Drawing observations on the applicability of certain algorithms’ clusters to different types of datasets.
    * Having an outcome including metric-based evaluation, as well as, human based evaluation to the algorithms.

## Contributors

| Contributor                                                                                         | GitHub Account                                       | Email Address                                                 | LinkedIn Account                                                                    | Other Links                                            |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------|--------------------------------------------------------|
| <center><img src="./docs/Images/berk_sudan_profile_photo.jpg" height="200"/><br>Berk Sudan</center> | [github.com/berksudan](https://github.com/berksudan) | [berk.sudan@protonmail.com](mailto:berk.sudan@protonmail.com) | [🔗](https://linkedin.com/in/berksudan)                                             | [medium.com/@berksudan](https://medium.com/@berksudan) |
| <center><br>Ferdinand Kapl</center>                                                                 | -                                                    | -                                                             | -                                                                                   | -                                                      |
| <center><img src="./docs/Images/yuyin_lang_profile_photo.jpg" height="200"/><br>Yuyin Lang</center> | [github.com/YuyinLang](https://github.com/YuyinLang) | [yuyin.lang@gmail.com](mailto:yuyin.lang@gmail.com)           | [🔗](https://www.linkedin.com/in/yuyin-lang-%E9%83%8E%E7%BE%BD%E5%AF%85-27aa7722a/) | -                                                      |

## Repository structure

- `docs` includes documents for this work, such as task description, final paper, presentations, and literature
  research.
- `data` includes all the datasets used in this work
- `notebooks` includes all the demo notebooks (for different algorithms) and one bulk run notebook
- `src` includes py files that consist of the pipeline of this work

## Project Report and Presentations

- Final Project Report: [pdf](./docs/Report/Final%20Report.pdf),
  [LaTeX](./docs/Report/LaTeX%20Files%20of%20Final%20Report).
- Presentations:
    - Final Presentation: [pdf](./docs/Presentations/Final%20Presentation%20%5B2022.07.26%5D.pdf),
      [odp](./docs/Presentations/Final%20Presentation%20%5B2022.07.26%5D.odp),
      [pptx](./docs/Presentations/Final%20Presentation%20%5B2022.07.26%5D.pptx).
    - Midterm Presentation: [pdf](./docs/Presentations/Midterm%20Presentation%20%5B2022.06.22%5D.pdf),
      [odp](./docs/Presentations/Midterm%20Presentation%20%5B2022.06.22%5D.odp),
      [pptx](./docs/Presentations/Midterm%20Presentation%20%5B2022.06.22%5D.pptx).
    - Intermediate Presentation - #2: [pdf](./docs/Presentations/Intermediate%20Presentation%20%5B2022.07.14%5D.pdf),
      [odp](./docs/Presentations/Intermediate%20Presentation%20%5B2022.07.14%5D.odp),
      [pptx](./docs/Presentations/Intermediate%20Presentation%20%5B2022.07.14%5D.pptx).
    - Intermediate Presentation - #1: [pdf](./docs/Presentations/Intermediate%20Presentation%20%5B2022.06.09%5D.pdf),
      [odp](./docs/Presentations/Intermediate%20Presentation%20%5B2022.06.09%5D.odp),
      [pptx](./docs/Presentations/Intermediate%20Presentation%20%5B2022.06.09%5D.pptx).

## Datasets

* Explored the provided datasets to unveil the inherent characteristics.
* Obtained an overview of the statistical characteristics of the datasets.

### Available Datasets

| Resource Name             | Is Suitable? | Type                                | Contains Tweet Text? | Topic Count | Total Instances | Topic Distribution                                                                                                      |
|---------------------------|--------------|-------------------------------------|----------------------|-------------|-----------------|-------------------------------------------------------------------------------------------------------------------------|
| 20 News (By Date)         | Yes          | Long Text Dataset                   | No                   | 20          | 853627          | (42K - 45K - 52K - 33K - 30K - 53K - 33K - 35K - 33K - 37K - 45K - 51K - 33K - 45K - 45K - 51K - 46K - 65K - 50K - 33K) |
| Yahoo Dataset (60K)       | Yes          | Long Text Dataset                   | No                   | 10          | 60000           | (6K - 6K - 6K - 6K - 6K - 6K - 6K - 6K - 6K - 6K)                                                                       |
| AG News Titles and Texts  | Yes          | Long Text Dataset                   | No                   | 4           | 127600          | (32K - 32K - 32K - 32K)                                                                                                 |
| CRISIS NLP - Resource #01 | Yes          | Short Text Dataset                  | Yes                  | 4           | 20514           | (3K - 9K - 4K - 5K)                                                                                                     |
| CRISIS NLP - Resource #12 | Yes          | Short Text Dataset                  | Yes                  | 4           | 8007            | (2K - 2K - 2K - 2K)                                                                                                     |
| CRISIS NLP - Resource #07 | Yes          | Short Text Dataset                  | Yes                  | 2           | 10941           | (5K - 6K)                                                                                                               |
| CRISIS NLP - Resource #17 | Yes          | Short Text Dataset                  | Yes                  | 10          | 76484           | (6K - 5K - 3K - 21K - 8K - 7K - 4K - 12K - 0.5K - 9K)                                                                   |
| AG News Titles            | Yes          | Short Text Dataset                  | No                   | 4           | 127600          | (32K - 32K - 32K - 32K)                                                                                                 |

+ If you want to see unavailable but analyzed datasets, please
  visit: [unavailable_datasets.md](./docs/Intermediate%20Results/unavailable_datasets.md).

## Deployment and Run

### Build

* For Linux, It is enough to run the following command for setting up virtual environment and install dependencies.

```bash
$ ./build_for_linux.sh
```

* For windows and other operating systems, install `python 3.8`, and install dependencies
  with `pip install -r requirements.txt`. Be careful about the package versions and make sure that you have the correct
  version in your current set up!

### Run

+ To run the Jupyter Notebook, just execute the following command:

```bash
$ ./run_jupyter.sh
```

**Note:** For windows and other operating systems, it can be done via Anaconda or similar tools.

+ Then, you can run the notebooks in `./notebooks`. There is one notebook for each algorithm and a general main runner
  that executes with a config parametrically.

### Evaluation Metrics

The following evaluation metrics are used for a metric based assessment of the produced topics:

* **Diversity Unique**: percentage of unique topic words; in [0,1] and 1 for all different topic words
* **Diversity Inverted Rank-Biased Overlap**: rank weighted percentage of unique topic words, words at higher ranks are
  penalized less; in [0,1] and 1 for all different topic words
* **Coherence Normalized Pointwise Mutual Information**: metric for coherence of topic words, how well do they fit
  together as topic?; in [-1,1] and 1 for perfect association
* **Coherence V**: metric for coherence of topic words evaluated by large sliding windows over the text together with
  indirect cosine similarity based on NPMI; in [0,1] and 1 for perfect association
* **Rand Index**: similarity measure for the two clusterings given by the topic model and the real labels, in [0,1] and
  1 for perfect match

## References

* Angelov: Top2vec: Distributed representations of topics: https://github.com/ddangelov/Top2Vec
* Grootendorst: BERTopic (https://github.com/MaartenGr/BERTopic)
* OCTIS Framework: https://github.com/MIND-Lab/OCTIS
* Dataset - CRISIS NLP: https://crisisnlp.qcri.org/.
* Dataset - 20NewsGroups: http://qwone.com/~jason/20Newsgroups/.
* Dataset - Yahoo: https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset.
* Csv to Markdown Table Converter #1: https://tableconvert.com/.
* Csv to Markdown Table Converter #2: https://markdown.co/tool/csv-to-markdown-table.
