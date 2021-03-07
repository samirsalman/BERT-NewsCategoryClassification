# BERT-NewsCategoryClassification
**News Category Classification with BERT based architecture.** 
You can find the notebook in this repository or in my kaggle (https://www.kaggle.com/samirsalman97/newscategory).


## The task
Text classification datasets are used to categorize natural language texts according to content. For example, think classifying news articles by topic, or classifying book reviews based on a positive or negative response. Text classification is also helpful for language detection, organizing customer feedback, and fraud detection.

Category classification, for news, is a text classification problem. The goal is to assign one category to a news article.

![task](https://miro.medium.com/max/700/1*HgXA9v1EsqlrRDaC_iORhQ.png)


## Data

I used a news category dataset aviable on kaggle datasets list. You can find it here (https://www.kaggle.com/rmisra/news-category-dataset).
It's a json dataset and contains 202,372 records, each record has the following structure:

```json
{
  "category":string "CRIME"
  "headline":string "There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV"
  "authors":string "Melissa Jeltsen"
  "link":string "https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89"
  "short_description":string "She left her husband. He killed their children. Just another day in America."
  "date":string "2018-05-26"
}
```

In this notebook we classify documents by category class, the 41 categories in the dataset are:

```
['CRIME' 'ENTERTAINMENT' 'WORLD NEWS' 'IMPACT' 'POLITICS' 'WEIRD NEWS'
 'BLACK VOICES' 'WOMEN' 'COMEDY' 'QUEER VOICES' 'SPORTS' 'BUSINESS'
 'TRAVEL' 'MEDIA' 'TECH' 'RELIGION' 'SCIENCE' 'LATINO VOICES' 'EDUCATION'
 'COLLEGE' 'PARENTS' 'ARTS & CULTURE' 'STYLE' 'GREEN' 'TASTE'
 'HEALTHY LIVING' 'THE WORLDPOST' 'GOOD NEWS' 'WORLDPOST' 'FIFTY' 'ARTS'
 'WELLNESS' 'PARENTING' 'HOME & LIVING' 'STYLE & BEAUTY' 'DIVORCE'
 'WEDDINGS' 'FOOD & DRINK' 'MONEY' 'ENVIRONMENT' 'CULTURE & ARTS']
```

## Architecture

I use a Small-BERT encoder (https://huggingface.co/google/bert_uncased_L-4_H-512_A-8) for the input layer, a Dropout Layer with 0.4 probability value and a simple output layer for the classification.

* BERT encoder
* Dropout Layer
* Output Layer


## Hyperparameters


* **Optimizer**: AdamW
* **Learning Rate**: 2e-5
* **Loss**: CrossEntropy Loss
* **Optimizer Scheduler**: linear schedule with warmup (https://huggingface.co/transformers/main_classes/optimizer_schedules.html)
* **Batch Size**: 32
* **Test Batch Size**: 128
* **Epochs**: 5
* **Dropout**: 0.4






## Results
We use a few-shots learning approach with 5 epochs training. The results are the following:

| Accuracy | F1-measure |
| --- | ----------- |
| 67,15% | 56,50% |


## Testing

Model usage examples:
```
News text: Can Inter Milan win the league this year?
Category: SPORTS

================================================

News text: Trump and his strategy
Category: POLITICS

================================================

News text: Milan fashion-week, this is the week!
Category: STYLE
```



## Libraries

* Transformers (Tranformer-based architectures)
* PyTorch (Deep Learning Module)
* Seaborn (Charts Visualization)
* Sklearn (ML Library, I use it for Data manipulation)
* Pandas (DataFrame and Data)
* Numpy (Linear Algebra)



## Contributors

Samir Salman
