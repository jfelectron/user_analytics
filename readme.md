
The analysis and prediction pipeline is split into 3 classes within the asasa.user_adoption package.

* to execute code or run the contained Jupyter notebook add the asana package to your PYTHONPATH. There are no dependencies other than the standard scientific python stack. 

1) UserFeatures: encapsulates data transformation logic

2) AdoptionAnalyzer: segments user data into cohorts based on user defined criteria for adoption

3) AdoptionPredictor: runs RandomizedGridSearch using a RandomForest Classfier and outputs a report on model performance

I did not have time to write tests. However, the modular design and methods performning small amounts of work was designed with 
the intention of being easily testable. Furthermore, the modular design could permit future code re-use and allow parallel work 
by more than one person. 

An example output report from a run using all the features is included as adoption_random_search_all_features.txt


