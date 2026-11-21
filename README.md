# News Article Multiclass Classification

Applied Machine Learning, Experiment 10, UPES  
Karan Chaubey and Himani Phartiyal · Faculty: Dr. Pavinder Yadav

Given an English headline and article, predict one editorial desk: business, entertainment, politics, sport or tech. The executed notebook includes the full implementation, comparison tables, confusion matrix, an error analysis and an editable demo.

## Run in VS Code

1. Open the project folder containing `Experiment_10_News_Classification.ipynb`. Enable the Python and Jupyter extensions in VS Code.
2. On the project Mac, select the existing **Machine Learning/.venv** Python kernel. On a fresh installation, use Python 3.9–3.12, create a virtual environment and run `python -m pip install -r requirements.txt` in it. Select that environment as the notebook kernel.
3. The local project already contains both input files. For a fresh clone, create a `data` directory alongside the notebook and follow the dataset instructions below.
4. Open the notebook and choose **Run All**. Edit `new_article` in the final code cell and run that cell again for a live prediction. The notebook works offline once the data and packages are in place.

## Datasets

- **Suggested dataset audit.** Download the archive from [News Classification Dataset 2026](https://www.kaggle.com/datasets/mmumairkhattak/news-classification-dataset-2026-nlp-and-tc) by mmumairkhattak. Extract `news_classification_dataset.csv` into `data/`. Across 30,000 rows there are 8 unique bodies and 16 unique full texts. Each category has one body. A row-level split would reward memorisation, and a body-disjoint split cannot represent all eight topics in both training and test.
- **Classifier training.** Download the [BBC full-text ZIP](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip) and save it intact as `data/bbc-fulltext.zip`. It contains 2,225 news articles from 2004–2005. Folder names supply the five labels. Original articles are copyright BBC. The benchmark is available for non-commercial and research use. [Maintainers and citation](https://github.com/derekgreene/bbc-datasets).
- The inputs stay local and are excluded by `.gitignore`. A clone needs both downloads. The eight Kaggle labels and the five BBC labels describe separate datasets.

## Method

1. Audit repeated article text before splitting. For the BBC corpus, read headline and body, lowercase the text, extract English letter sequences, discard empty records and remove repeated cleaned bodies and headlines in deterministic filename order. This leaves 2,060 articles. Rewritten coverage of the same event can still remain.
2. Make stratified training (1,236), validation (412) and test (412) sets with seed 42. Filename, folder name and metadata never enter the features.
3. Convert single words to TF-IDF weights. Remove English stop words and terms seen in fewer than two training articles. Fit the vectorizer inside each pipeline. The initial training vocabulary has 12,245 terms.
4. Fit a majority baseline, multinomial Naive Bayes and logistic regression to the same training split. Select by validation macro F1, the average of the five per-desk F1 scores. Refit the chosen pipeline on the 1,648 development articles and evaluate it once on test.
5. Inspect the classification report, confusion matrix, strongest words and a mistaken headline. Word coefficients describe associations learned from this corpus.

## Results

| Model on validation | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 0.2427 | 0.0485 | 0.2000 | 0.0781 |
| Multinomial Naive Bayes | 0.9587 | 0.9637 | 0.9521 | 0.9567 |
| Logistic regression | 0.9733 | 0.9763 | 0.9695 | 0.9725 |

The selected logistic regression pipeline correctly classified **404 of 412** test articles. Test accuracy was **98.06%**, macro precision **0.9799**, macro recall **0.9807** and macro F1 **0.9802**. One politics story about UK firms adopting e-commerce went to tech. Some stories span desks. The scores describe one historical BBC split. A follow-up should test the frozen model on more recent news from other publishers.

## In-class demonstration

Karan can introduce the task, show why duplicate text matters and walk through the TF-IDF example. Himani can compare the models, explain the confusion matrix and change the final input live. Both presenters can rehearse the same run before class.

For a quick demonstration, the supplied sentence about a striker reaching the cup final predicts **sport**. Try another original sentence about a bank or an election. If the input has no words from the training vocabulary, the demo asks for more detail. An ambiguous article can receive a prediction even when another desk would be plausible.

## Reference

D. Greene and P. Cunningham. *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*. Proceedings of ICML, 2006.
