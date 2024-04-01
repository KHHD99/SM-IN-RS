# Similarity Metrics Investigation in Recommender Systems (SMIRS)
Recommender systems are indispensable tools in modern digital platforms, guiding users through an overwhelming array of options by suggesting items that align with their preferences. At the heart of these systems lie similarity metrics, crucial for assessing the likeness between items and users' tastes to generate accurate recommendations. However, the effectiveness of these metrics depends on various factors, including dataset characteristics and the complexity of recommendation tasks.

In order to fully evaluate the effectiveness of similarity metrics, our project conducts a comprehensive simulation program. We have attentively selected seven distinct datasets that cover several domains and consumer preferences, so establishing an extensive testing ground for our assessment. We hope to get insight into these metrics' efficacy and applicability in a range of recommendation scenarios by rigorously evaluating them using popular performance indicators.

We use a rigorous approach, carefully planning experiments and accounting for confounding variables, consistent with best practices in recommender system research. Strict statistical methods guarantee the accuracy and repeatability of our conclusions, enabling us to approach data analysis methodically and identify trends. With this method, we hope to pinpoint anomalies and clarify the relative advantages and disadvantages of every similarity metric.

Ultimately, our project seeks to advance the state-of-the-art in recommender system research by shedding light on the effectiveness of similarity metrics. By elucidating the nuances of these metrics across different datasets and recommendation algorithms, we empower practitioners and researchers to make informed decisions. Through our comprehensive simulation and rigorous evaluation approach, we contribute valuable insights to enhance the performance and user experience of recommender systems in various real-world applications.

# Project Overview
This project is a machine learning project that aims to investigate the similarity of in recommender systems. The project involves cleaning and preprocessing the data, training several machine learning models, and selecting the best similarity metric that perform well based on evaluation metrics.

# Technologies Used
  - Python 3
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
 
# Datasets
The datasets used in this project are the followings:
  - Group-Lens datasets: 
    - Datasets from the Movie-Lens (https://movielens.org) have been compiled and made accessible by Group-Lens Research. The datasets were collected during various time periods, depending on their size (100K, 1M, 10M, 20M, 25M)
    - In the this project we used the MovieLens 100k, 1M, 10M (https://grouplens.org/about/what-is-grouplens/) and the datasets can be accessible here (https://grouplens.org/datasets/movielens/)

  - Jester dataset : 
    - A dataset from the Jester Joke, the dataset comprises feedback on 100 jokes from 73,496 users, with a total of 4.1 million ratings ranging from -10.00 to +10.00, (https://grouplens.org/datasets/jester/).

  - Epinions dataset : 
    - The Epinions dataset, which is openly accessible and commonly employed in the research of recommender systems, includes ratings provided by users for items and explicit trust/distrust relationships between users. Users assess other users based on the quality of their reviews about the item, (https://www.kaggle.com/datasets/masoud3/epinions-trust-network).
    
  - MovieTweetings dataset : 
    - A Movie Rating Dataset Collected From Twitter. The MovieTweetings dataset comprises ratings of movies that were expressed in well-organized tweets on Twitter. It was produced as a part of a research project by Simon Dooms (http://scholar.google.be/citations?user=owaD8qkAAAAJ) from Ghent University, Belgium, and presented during the CrowdRec 2013 workshop at the ACM RecSys 2013 conference, (https://www.kaggle.com/datasets/tunguz/movietweetings).
    
  - Electronics dataset : 
  -   - A collection of approximately 15,000 electronic products containing pricing information in 10 areas. Datafiniti's Product Database provides a list of over 15,000 electrical items with pricing information across 10 key variables. The dataset also includes brand, category, merchant, name, source, and other information (https://www.kaggle.com/datasets/datafiniti/electronic-products-prices), (https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews).

# Installation
  1 - Clone the repository of Microsoft Recommenders to your local machine, or install it using Anaconda.
  
  2 - Install the required packages using pip
  
  3 - Look for the `python_utils.py` using the following path `/recommenders/utuls`file then implement the similarity metric you want to add and evaluate
  
  4 - Look for the `/recommenders/evaluation/python_evaluation.py` file and add the  performance evaluation metrics desired to evaluate the similarity metrics on RS. 
  
  5 - Look for the `/recommenders/models/sar/sar_singlenode.py` update the file by adding the necessary scripts based on the similarity and the performance evaluation metric implemented

# Usage
  
  1- Navigate to the project directory in your local machine after the installation or clone the Microsoft Recommenders.
  
  2- fined the following file `/recommenders/models/sar/sar_singlenode.py` and replace it by the one (`sar_singlenode.py`) provided in this repository in the folder named `Installation Files`.
  
  3- Do the same for `/recommenders/utuls/python_utils.py` using `python_utils.py`of this repository as well.
  
  4- Run the `Adaptative_code.ipynb` to generate the results.
  
