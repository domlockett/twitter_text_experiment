# Twitter Text Experiment

## Overview

The Twitter Text Experiment aims to collect and analyze tweets from approximately 30 news outlets, including all comments on these tweets. The goal is to build a comprehensive dataset to study the variation in social commentary compared to professional news outlets.

## Goals
The primary goal of this project is to collect and analyze Twitter data from various news outlets and their associated comments. This analysis aims to compare social commentary with professional news content, providing observational data to complement previous experimental studies.

## Data

- **Observational Data:** Collects tweets from a list of approximately 30 news outlets, including all comments on those tweets.
- **Dataset:** The target is to gather around 5 million tweets per month, overcoming the API's limit of 500,000 tweets per month per account by distributing the scraping tasks among multiple developer accounts.

## Design

### Data Collection:
- **Twitter API:** Utilizes Twitter's API to scrape tweets and comments.
- **Multiple Accounts:** Distributes data collection across multiple developer accounts to meet the high volume target.
- **Automated Scraping:** Code runs in the background to ensure continuous data collection with minimal daily maintenance.

### Experiment Design:
- **Content Analysis:** Analyzes variations in social commentary and professional news content.
- **Text and Image Data:** Includes both textual and image data from tweets and comments.

### Statistical Analysis:
- **Advanced Techniques:** Employs techniques such as regression analysis and propensity score matching to isolate the effects of tweet content from other variables.

## Key Findings
1. **Variation in Commentary:** Identifies significant differences in the nature of social commentary compared to professional news content.
2. **Engagement Patterns:** Analyzes patterns in user engagement with news tweets and comments.
3. **Sentiment Analysis:** Studies sentiment variations across different news outlets and user comments.

## Methodological Contributions

My role in this project involved managing data from the design and collection phases through initial analyses. I ensured all hypotheses were tested and visualized for co-authors, collaborating to identify the most relevant information for publication.

## Detailed Code Descriptions

### Scrape News Comments SQL

**Description:**
This script scrapes tweets and comments from specified news outlets using the Twitter API and stores the data in an SQL database.


The [`scrape-newscomments-sql.py`](https://github.com/domlockett/twitter_text_experiment/blob/main/scrape-newscomments-sql.py) script is responsible for collecting tweets from specified news organizations. The script starts by setting up a connection to the Twitter API using authentication keys. The target news organizations and date ranges are defined within the script. 

**Content and Structure:**
- **API Requests:** The script makes GET requests to the Twitter API, fetching tweets in batches to handle rate limits.
- **Data Storage:** Retrieved tweets are parsed and stored in an SQL database. Each tweet's metadata, such as tweet ID, timestamp, text, user information, and retweet count, is also recorded.
- **Error Handling:** The script includes comprehensive error handling mechanisms to manage API rate limits, network issues, and invalid responses. It retries failed requests and logs errors for further inspection.

**Error Handling in Detail:**
- **Rate Limits:** The script checks for HTTP 429 status codes and implements a backoff strategy to wait before retrying.
- **Network Errors:** It catches network-related exceptions and retries the connection a predefined number of times.
- **Data Validation:** Before inserting data into the SQL database, it validates the tweet content to ensure it conforms to the expected format, preventing SQL injection and other data integrity issues.


### Processing Tweets
**Description:**
[`data_management/processing_tweets_comments.py'](https://github.com/domlockett/twitter_text_experiment/blob/main/processing_tweets_orig.py) processes the raw tweet data to prepare it for analysis, including text cleaning, tokenization, and feature extraction.

**Data Loading and Saving:**
- **Loading Data:** The script reads raw tweet data from the SQL database. It utilizes efficient SQL queries to load data in chunks, handling large datasets without overwhelming memory.
- **Saving Data:** Processed data is saved back to the database or written to CSV files for further analysis. Intermediate results are stored to allow for resuming processing in case of failures.

**Data Cleaning:**
- **Text Normalization:** The script removes URLs, special characters, and converts text to lowercase. Regular expressions are used to identify and strip unwanted patterns.
- **Tokenization:** The text is split into individual tokens using natural language processing libraries like NLTK or SpaCy. This step prepares the data for further analysis.
- **Stop Words Removal:** Commonly used words that do not contribute to the analysis, such as "and", "the", and "in", are removed. A predefined list of stop words is used, which can be customized based on the dataset.

**Debugging:**
- **Logging:** The script logs each step of the processing, including the number of tweets processed, time taken for each operation, and any errors encountered. This helps in identifying bottlenecks and errors.
- **Error Handling:** Specific try-except blocks are implemented to catch and handle exceptions during processing. Errors are logged, and problematic data entries are either corrected or flagged for manual review.

**Sentiment Analysis:**
- **Algorithm Application:** Sentiment analysis algorithms are applied to determine the sentiment of each tweet. Libraries like TextBlob or Vader are used for this purpose.
- **Storing Results:** The sentiment scores are added to the processed data and saved for further analysis.

**Prediction Template for `PartyPredictions-main/predictions.py`:**
- **Machine Learning Models:** Various machine learning models, such as logistic regression, SVM, and neural networks, are used to predict party affiliations based on tweet content.
- **Training and Validation:** Models are trained using labeled datasets and validated to ensure accuracy. The prediction results are stored for subsequent analysis and reporting.


### Beta Analysis
**Descriptions:**
The [`data_management/beta_analyses.r'](https://github.com/domlockett/twitter_text_experiment/blob/main/beta_analysis.py) file performs exploratory data analysis and initial hypothesis testing on the processed tweet data.

**Content and Structure:**

1. **Data Loading:**
   - **SQL Queries:** Utilizes SQL queries to load and manipulate data from SQLite databases efficiently.

2. **Preprocessing:**
   - **Text Cleaning:** Employs the `preprocessor` library and custom regular expressions to clean tweet text by removing URLs, special characters, and stopwords.
   - **Tokenization and Vectorization:** Uses libraries like `nltk` and `gensim` for tokenization and stopwords removal, preparing text for further analysis.

3. **Sentiment Analysis:**
   - **Model and Tokenizer:** Loads a pre-trained sentiment analysis model (`distilbert-base-uncased`) and tokenizer from the `transformers` library.
   - **Pipeline:** Sets up a pipeline for sentiment analysis using the pre-trained model to predict sentiment labels and scores for each tweet.
   - **Prediction:** Loops through tweet texts, applies the sentiment analysis pipeline, and stores predictions and scores.

4. **Descriptive Statistics:**
   - **Word Count Analysis:** Calculates average word counts for tweets and replies, both for the entire dataset and specific case studies (e.g., January 6th).
   - **Action Words:** Identifies tweets containing mobilizing calls to action and calculates the average number of such tweets and replies.

5. **Partisan Bias Analysis:**
   - **Bias Prediction:** Uses the sentiment analysis pipeline to predict partisan bias in tweets and replies.
   - **Statistical Analysis:** Calculates the average partisan bias for all comments and parent tweets, and plots the distribution of partisan bias.

6. **Visualization:**
   - **Plotting:** Utilizes `matplotlib` and `seaborn` for creating distribution plots of partisan bias in tweets and replies, highlighting differences in bias.


## Acknowledgement
This project is a collaborative effort involving significant contributions from various scholars. The provided files and scripts reflect the extensive work done to understand and address the impacts of social media on public discourse. Created for academic purposes, any anonymized data has been removed to ensure privacy and confidentiality. Developed for Washington University in Saint Louis Political Science Department, as well as Exeter and Princeton University. Special thanks to all authors for agreeing to publicize my contributions to the project.
