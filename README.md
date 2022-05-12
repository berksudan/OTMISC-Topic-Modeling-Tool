# Topic modeling advancements

## Task description
- evaluate different Topic Modeling algorithms on short/long text dataset
- draw observations on the applicability of certain algorithms’ clusters to different types of datasets

## Datasets 
- https://crisisnlp.qcri.org/
- http://qwone.com/~jason/20Newsgroups/

For further information see the Survey_of_Topic_Modeling_Algorithm.pdf

## Deployment

### Build

+ For Linux, It is enough to run the following command for setting up virtual environment and install dependencies.

```bash
$ ./build.sh
```

+ For windows and other operating systems, install `python 3.8`, and install dependencies with `pip install -r requirements.txt`. Be careful about the package versions and make sure that you have the correct version in your current set up!

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
