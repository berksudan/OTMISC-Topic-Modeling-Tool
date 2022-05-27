# Top2Vec - Research Report


## 1. Advantages and Disadvantages

**1.1 Advantages:**

* No need to the following pre-processing methods:
  + To remove stop-words, because the words closest to a topic vector will rarely be stop-words, which has been confirmed in [1].
  + Lemmatization
  + Stemming

* No need to have a priori knowledge of the number of topics.

* Supports hierarchical topic reduction.

* In HDBSCAN, noise and variable density clusters can be handled.

**1.2 Disadvantages:**

* *1-Topic Policy:* 1 document is assigned to 1 topic only.

* *Balanced Dataset Requirement:* Hierarchical topic reduction biases topics with greater size, so a balanced dataset should be given if hierarchical topic reduction will be used.

* *Curse of Dimensionality:* It leads to Sparsity and Computational Cost but UMAP is used for dimensionality reduction before clustering.


## 2. Definitions

* **Descriptive Word Vectors:** The ones that are nearest to a document vector, are the most semantically descriptive of the document's topic.

* **Topic Vector:** Mean of document vectors

* **Topic:** Dense Areas in document vectors

* **Noise Documents:** The documents that are not descriptive of a prominent topic.

* **Topic Information Gain:** Measures the quality of the words in the topic and their assignment to documents.

* **Informative and Uninformative Words:** If topics contain words such as "the", "and", and "it" or other intuitively uninformative words, they will receive lower information gain values. Most informative topics are highly localized and the uninformative topics are spread out over many messages.

* **Size of a Topic:** Number of documents assigned to the topic.


## 3. Comparisons

**Low Dimensional Document Embedding Comparison: t-SNE vs UMAP**
  * t-SNE doesn't preserve global structure, UMAP preserves global and local structure [1].
  * t-SNE does not scale well for large datasets, UMAP scales well for very large datasets [1].

**Algorithms Comparison: top2vec vs LDA/PLSA**
* top2vec consistently finds topics that are more informative and representative of the corpus than probabilistic generative models like LDA and PLSA, for varying sizes of topics and number of top topic words.


## 4. Hyperparameters

###  4.1 Hyperparameters of ``doc2vec`` Embedding Model

**1.  Window Size:**
  + *Definition:* The number of words left and right of the context word
  + *Value Options:* A positive integer.
  + *Suggested Value:* 15, source: The Top2Vec paper [1] and [2].
  
**2. Output Layer:**
  + *Definition:* -
  + *Value Options:* "Negative Sampling" or "Hierarchical Softmax", which are both efficient approximations of the full softmax.
  + *Suggested Value:* "Hierarchical Softmax", source: The Top2Vec paper [1]

**3. Sub-sampling Threshold:**
  + *Definition:* According to [2], the most important hyperparameter. It determines the probability of high frequency words being discarded from a given context window. The smaller this number is, the more likely it is for a high frequency word to be discarded from the context window.
  + *Value Options:* A positive integer.
  + *Suggested Value:* 10⁵ (``10**5``)

**4. Minimum Count:**
  + *Definition:* Discards all words that have a total frequency that is less than that value from the model all together.
  + *Value Options:* A positive integer.
  + *Suggested Value:* 50, ***BUT*** this value largely depends on corpus size and its vocabulary.
 
**5. Vector Size:**
  + *Definition:* The size of the document and word vectors that will be learned, the larger they are the more complex information they can encode.  the larger they are the more complex information they can encode. The larger values will lead to better results, at greater computational cost.
  + *Value Options:* A positive integer.
  + *Suggested Value:* 300, source: [2].

**6. Number of Training Epochs:**
  + *Definition:* 
  + *Value Options:* A positive integer.
  + *Suggested Value:* 40 to 400 according to The Top2Vec paper [1] and 20 to 400 according to [2].

### 4.2 Hyperparameters of ``UMAP`` Dimensional Reduction Method

**1. Number Of Nearest Neighbors:**
  + *Definition:* Perhaps the most important one, which controls the balance between preserving global structure versus local structure in the low dimensional embedding. Larger values put more emphasis on global over local structure preservation. Since the goal is to find dense areas of documents which would be close to each other in the high dimensional space, ***local structure is more important*** in this application. 
  + *Value Options:* A positive integer.
  + *Suggested Value:* 15, gives more emphasis on local structure.

**2. Distance Metric:**
  + *Definition:* Measures the distance between points in the high dimensional space. 
  + *Value Options:* Usually "cosine similarity".
  + *Suggested Value:* "cosine similarity", because it measures similarity of documents irrespective of their size.

**3. Embedding Dimension:**
  + *Definition:* -
  + *Value Options:* A positive integer.
  + *Suggested Value:* 5, because it gives the best results for the downstream task of density based clustering.

### 4.3 Hyperparameters of ``HDBSCAN`` Clustering Method

**1. Minimum Cluster Size:**
  + *Definition:* Represents the smallest size that should be considered a cluster by the algorithm.
  + *Value Options:* A positive integer.
  + *Suggested Value:* 15, because larger values have a higher chance of merging unrelated document clusters

### 5. Experiments

**20 News Dataset:**
  + Number of topics found by ``top2vec``: 103
  + Number of Topics for LDA and PLSA Models: trained with 10 to 100 topics, with intervals of 10.
  + The topic information gain for top2vec is consistently higher than for LDA and PLSA.

**Yahoo Answers Dataset:** Contains 1.3 million labelled posts.
  + Number of topics found by ``top2vec``: 2,618
  + Number of Topics for LDA and PLSA Models: trained with 10 to 100 topics, with intervals of 10.
  + The topic information gain for top2vec is consistently higher than for LDA and PLSA.


## References

**\[1]:** Angelov, D., 2022. Top2Vec: Distributed Representations of Topics. [online] arXiv.org. Available at: <https://arxiv.org/abs/2008.09470> [Accessed 26 May 2022].

**\[2]:** Jey Han Lau and Timothy Baldwin. An empirical evaluation of doc2vec with practical insights into document embedding generation. In Proceedings of the 1st Workshop on Representation Learning for NLP, pages 78–86, Berlin, Germany, August 2016. Association for Computational Linguistics.
