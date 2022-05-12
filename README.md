# Topic modeling advancements

**Supervisor:** Professor Dr. Georg Groh

**Advisors:** PhD Candidate (M.Sc.) Miriam Anschütz, PhD Candidate (M.Sc.) Ahmed Mosharafa

**Project Scope:**

* Evaluating different Topic Modeling algorithms on short/long text dataset
* Drawing observations on the applicability of certain algorithms’ clusters to different types of datasets.
* Having an outcome including metric-based evaluation, as well as, human based evaluation to the algorithms.


## Milestone #01: Datasets and Data Exploration

* Explored the provided datasets to unveil the inherent characteristics.
* Obtained an overview of the statistical characteristics of the datasets.
* **Included Datasets:**
  * CRISIS NLP: https://crisisnlp.qcri.org/.
  * 20NewsGroups: http://qwone.com/~jason/20Newsgroups/.

### Outcome

| ResourceName | Type                                | HasTweetText? | EventCount | EventSizes                                                                                                                                                                                        | 
|--------------|-------------------------------------|---------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| Resource #01 | Short Text Dataset                  | HasText       | 12         | (2K;2K;2K;2K;2K;1K;2K;1K;2K;2K;2K;3K)                                                                                                                                                             | 
| Resource #02 | Short Text Dataset                  | HasText       | 7          | (1K;1K;2K;1K;1K;13K;5K)                                                                                                                                                                           | 
| Resource #03 | Short Text Dataset                  | HasText       | 6          | (2K;1K;9K;1K;2K;2K)                                                                                                                                                                               | 
| Resource #04 | A Tool for LSTM RNNs                | -             | -          | -                                                                                                                                                                                                 | 
| Resource #05 | Short Text Dataset                  | HasText       | 7          | (1K;4K;4K;4K;0.5K;1K;1K)                                                                                                                                                                          | 
| Resource #06 | Short Text Dataset                  | NoText        | 3          | (1207K;1096K;6506K)                                                                                                                                                                               | 
| Resource #07 | Short Text Dataset                  | HasText       | 2          | (12K;10K)                                                                                                                                                                                         | 
| Resource #08 | A Tool for Tweets Retrieval Tool    | -             | -          | -                                                                                                                                                                                                 | 
| Resource #09 | Image Dataset                       | -             | -          | -                                                                                                                                                                                                 | 
| Resource #10 | Short Text Dataset                  | HasText       | 2          | (12K;10K)                                                                                                                                                                                         | 
| Resource #11 | Text Dataset with Name Lists        | -             | -          | -                                                                                                                                                                                                 | 
| Resource #12 | Short Text Dataset                  | HasText       | 4          | (2K;2K;2K;2K)                                                                                                                                                                                     | 
| Resource #13 | Image Dataset                       | -             | -          | -                                                                                                                                                                                                 | 
| Resource #14 | Geolocation Data                    | -             | -          | -                                                                                                                                                                                                 | 
| Resource #15 | ImageDataset                        | -             | -          | -                                                                                                                                                                                                 | 
| Resource #16 | Short Text Dataset                  | HasText       | 61         | (2K;3K;2K;2K;2K;2K;2K;2K;20K;1K;1K;2K;2K;20K;2K;2K;2K;2K;20K;2K;2K;2K;2K;2K;2K;2K;2K;20K;4K;2K;2K;20K;2K;2K;2K;2K;20K;9K;4K;5K;6K;1K;3K;0K;4K;4K;4K;4K;19K;4K;25K;4K;1K;3K;53K;8K;8K;8K;1K;2K;2K) | 
| Resource #17 | Short Text Dataset                  | HasText       | 17         | (7K;2K;4K;2K;8K;6K;9K;9K;7K;2K;1K;2K;8K;2K;2K;2K;1K)                                                                                                                                              | 
| Resource #18 | Text Dataset for Sentiment Analysis | -             | -          | -                                                                                                                                                                                                 | 

## Deployment and Run

### Build

+ For Linux, It is enough to run the following command for setting up virtual environment and install dependencies.

```bash
$ ./build.sh
```

+ 
For windows and other operating systems, install `python 3.8`, and install dependencies
  with `pip install -r requirements.txt`. Be careful about the package versions and make sure that you have the correct
  version in your current set up!

### Activate Project

In order to activate virtual environment and shorten the absolute path displayed in the Linux Terminal,
you can simply run the following command:

```bash
$ source ./activate_project.sh
```

* For windows and other operating systems, it can be done via Anaconda or similar tools.

### Run

To run the Jupyter Notebook, just execute the following command:

```bash
$ ./run_jupyter.sh
```

* For windows and other operating systems, it can be done via Anaconda or similar tools.



## References

* Angelov: Top2vec: Distributed representations of topics
* Grootendorst: BERTopic (https://github.com/MaartenGr/BERTopic)
