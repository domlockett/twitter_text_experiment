## Move Relevant Files

- In `TwitterConversations` folder, navigate to the new `Processing Tweets Materials` folder
<br>

- These contents will move to your **local** directory (and subdirectories)
<br>
- Where you keep/run the `scrape-newscomments-sql.py` and `automate_script_you.exe` 
  <br>

  - **Move** these files (and 1 folder) to the **main** directory:
    - *Actual function files*
      - `setup_tweets.py`
      - `processing_tweets.py`
    - *Executables*
      - `setup-script.txt`
      - `processing-script.txt`
    - *Review Code*
      - `Check Setup & Processing` &#8592; *folder*
  <br>
               
        
  - **Replace** the `processed` folder into your **`/User/Your Twitter Directory/data`** sub-directory
  <br>

    - `processed` &#8592; *folder*

## Make Executables

- Follow Erin's previous instructions :[^1]<br>
  1. Remove the ‘.txt’ extension from your unique file.  
     - Right click, click “Get Info”, and make sure the “.txt.” is removed from the file name under “Name & Extension”
<br>
  2. Run the following lines in your terminal:
     - Make sure the file path reflects your local data scraping directory

       - `chmod 755 /Users/LOCAL TWITTER DIRECTORY/setup-script`

       - `chmod 755 /Users/LOCAL TWITTER DIRECTORY/processing-script`



[^1]: Moved to `Box Sync\TwitterConversations\automate scrape directions and executables\automate_script_instructions.txt`


## Setup 

- Double click the `setup_script` executable. 
<br>
- You can confirm that it has run correctly 2 ways:
<br>  
    1. See two new files in your main directory:
          - `alembic.init` 
          - `alembic`  &#8592; *folder*.
<br> 
    2. Run code that is inside the `\Check Setup & Processing\check_setup.txt` file into your terminal
       - Be sure to change the first line to your directory
       - If 4 `True` values are returned you can now clean your data
<br>
- Minor startup cost running the setup 
  - A `unique_id` is added to easily access row data
    - Actual row data can change if something is deleted 
    - Static/unique SQL "primary key" in all tables is standard SQL procedure
  - For my 3 million it was a just a ~2 minutes.
    - [This is why we are using SQL-- no need to load information into memory!]

## Run

- Double-click `processing-script` executable.

- The tweets themselves get updated in batches of 10,000.
     - This allows quick review of updates and minimizes loss if it breaks for any reason.

- Allow `data/processed/logs/COMM-%date%.log` to update (watch size)
    - Confirm you are getting a list of values and no errors. 
      
    - If you have any errors paste them into Slack.

## Follow-ups

#### Test

- Allow the `processing-script` to run for a short period
    - Ensure you are not getting any errors in the most recent `data/processed/logs/COMM-%date%.log`
        - Close the terminal running `processing-script` (kill process)
        - Run  `processing-script` again
        - Compare the two log files 
        - Success is no overlaps
           - Log 1 last tweet cleaned:
             - `Tweet-Comment ID:   836962418106462208  Tweets Row ID: 110000`
           - Log 2 first tweet cleaned:
             - `Tweet-Comment ID: 836773568365084673  Tweets Row ID: 110001`
       - *But be mindful that there are logs for media and context data as well*

#### Check-in

- Review terminal to see if `processing script` is still running
- Review `data/processed/logs/COMM-%date%.log` periodically for errors
- Observe growth in `data/processed/logs/COMM-%date%.log` and `data/processed/clean_c.db`

#### Inspect

- Copy `Check Setup & Processing\review_tweets.txt` code into your terminal 
    - Be sure to change the directory on the first line
    - It should return a truncated view of your cleaned data.
        - I.e. no periods in any column names and 24 columns in the main `tweets` data


## Fin
- The process is very speedy; it cleaned my entire `tweet.db` in like 30  minutes!
<br>
- At some point you will finish cleaning
  - At that point wait to scrape some more data and run the `processing-script` again. 
  - Rinse and repeat.