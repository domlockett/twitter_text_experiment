# Twitter text experiment

## Overall Aims of the Project
The project aimed to collect tweets from 30 news organizations over the period 2017-2022. The goal was to analyze the data for patterns in communication, sentiment, and party predictions using the collected data.

## Methods Employed

### Data Collection

#### Scrape-newscomments-sql.py

The `scrape-newscomments-sql.py` script is responsible for collecting tweets from specified news organizations. The script starts by setting up a connection to the Twitter API using authentication keys. The target news organizations and date ranges are defined within the script. 

**Content and Structure:**
- **API Requests:** The script makes GET requests to the Twitter API, fetching tweets in batches to handle rate limits.
- **Data Storage:** Retrieved tweets are parsed and stored in an SQL database. Each tweet's metadata, such as tweet ID, timestamp, text, user information, and retweet count, is also recorded.
- **Error Handling:** The script includes comprehensive error handling mechanisms to manage API rate limits, network issues, and invalid responses. It retries failed requests and logs errors for further inspection.

**Error Handling in Detail:**
- **Rate Limits:** The script checks for HTTP 429 status codes and implements a backoff strategy to wait before retrying.
- **Network Errors:** It catches network-related exceptions and retries the connection a predefined number of times.
- **Data Validation:** Before inserting data into the SQL database, it validates the tweet content to ensure it conforms to the expected format, preventing SQL injection and other data integrity issues.

### Data Processing

#### Processing Tweets and Comments

The `processing_tweets_comments` module involves several steps to clean and transform the collected data for analysis.

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

**Prediction Template for PartyPredictions:**
- **Machine Learning Models:** Various machine learning models, such as logistic regression, SVM, and neural networks, are used to predict party affiliations based on tweet content.
- **Training and Validation:** Models are trained using labeled datasets and validated to ensure accuracy. The prediction results are stored for subsequent analysis and reporting.

This detailed approach ensures comprehensive data collection, robust processing, and accurate predictions, facilitating insightful analysis of Twitter conversations.

### Automation via Executables
The executables in the project automate the entire process, from data scraping to final analysis. These scripts are scheduled to run at specified intervals, ensuring that the data collection and analysis are up-to-date without manual intervention.

## Acknowledgments

This project is a collaborative effort involving significant contributions from various scholars. The presented files provide a copy of the project containing my contributions and are not original or source project files. Data has been excluded from this repository for privacy purposes. The project was created for the Washington University in Saint Louis Political Science Department. 
