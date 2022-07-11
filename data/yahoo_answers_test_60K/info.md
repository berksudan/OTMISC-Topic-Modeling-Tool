# Dataset Info

> **Full Name:** Yahoo! Answers Topic Classification Dataset (Version 2, Updated 09/09/2015)

## Adjustment for This Project

For this project, only `test.csv` is used for calculation, since it provides large enough dataset with 60K answers.
Only answers are used for training and further operations.

Note that, the following sections stand for the whole Yahoo Dataset, not just the reduced version in this project.

## Origin

The original Yahoo! Answers corpus can be obtained through the Yahoo! Research Alliance Webscope program. The dataset is
to be used for approved non-commercial research purposes by recipients who have signed a Data Sharing Agreement with
Yahoo!. The dataset is the Yahoo! Answers corpus as of 10/25/2007. It includes all the questions and their corresponding
answers. The corpus contains 4483032 questions and their answers.

The Yahoo! Answers topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above
dataset. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun.
Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (
NIPS 2015).

## Description

The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories. Each class contains
140,000 training samples and 6,000 testing samples. Therefore, the total number of training samples is 1,400,000 and
testing samples 60,000 in this dataset. From all the answers and other meta-information, we only used the best answer
content and the main category information.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 4 columns in them,
corresponding to class index (1 to 10), question title, question content and best answer. The text fields are escaped
using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a
backslash followed with an "n" character, that is "\n".

## Classes

+ Society & Culture
+ Science & Mathematics
+ Health
+ Education & Reference
+ Computers & Internet
+ Sports
+ Business & Finance
+ Entertainment & Music
+ Family & Relationships
+ Politics & Government
