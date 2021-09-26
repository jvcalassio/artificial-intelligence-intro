# Introduction to Artificial Intelligence

This repository refers to the second project of Introduction to Artificial Intelligence course, UnB - 2020/2

### Notes about the first project
The first project is not addressed here, but is worth mentioning: it consists on the eight questions of the [first project of Berkley Intro to AI Course](http://ai.berkeley.edu/search.html).

## Project description

The goal of the second project is to implement the Random Forest and the K-Nearest Neighbors classifiers in order to create a prediction model on COVID-19 infection. 

It's also required to optimize the parameters of the Random Forest, and perform an exploratory data analysis on the dataset.

The dataset consists of COVID-19 exam data on anonymous patients.

### Classifiers

The Random Forest implementation is really similar to the one present on [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and uses it's Decision Tree classifier.

The KNN classifier uses euclidian distance to determine the nearest neighbors.

### Random Forest parameter optimization

The parameter optimization is made by [sklearn's RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) set to 100 iterations. It tries random combinations of params and selects the best one.

## Running

You'll need Python 3, and the required libs available in the requirements.txt file.

Run using
```
python3 main.py
```

It'll ask you to choose the classifier, and if you choose the RF it'll also ask if you want to optimize the parameters.

The results are the accuracy, precision, recall and the confusion matrix of the selected classifier.

## References

[Towards Data Science - Exploratory Data Analysis](https://towardsdatascience.com/exploratory-data-analysis-visualization-and-prediction-model-in-python-241b954e1731)
[Towards Data Science - Hyperparameter Tuning the Random Forest](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
[Towards Data Science - Building a Random Forest Classifier](https://towardsdatascience.com/building-a-random-forest-classifier-c73a4cae6781)