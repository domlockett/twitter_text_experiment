#install first
library(RSQLite)
library(DBI)
##############################
# Example code
##############################
#wherever your data is stored
projhome <- 'C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/processed/'

#connect to the DATABASE BY feeding it the address 
original <- dbConnect(RSQLite::SQLite(), paste0(projhome,"tweets/clean-jan6_orig.db"))
comments <- dbConnect(RSQLite::SQLite(), paste0(projhome,"comments/clean-jan6_comm.db"))

# look at the different tables in each database
dbListTables(original)
dbListTables(comments)

#look at all the columns in the main table in the 2 databases
dbListFields(original, "tweets-o")
dbListFields(comments, "tweets")

#write sql query to gather the data into R
q1<-dbSendQuery(original, "SELECT * FROM 'tweets-o'")
q2<-dbSendQuery(comments, "SELECT * FROM 'tweets'")

#run the query
tweets<-  dbFetch(q1)
comms <- dbFetch(q2)


# Clear the result
dbClearResult(q1)
dbClearResult(q2)

# Disconnect from the database
dbDisconnect(original)
dbDisconnect(comments)
##############################


projhome <- 'C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/'
#projhome1 <- 'C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/ER_2-1-22/'
#projhome2 <- 'C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/sql/tweets/'

t <- dbConnect(RSQLite::SQLite(), paste0(projhome,"convo_ids.db"))

#t1 <- dbConnect(RSQLite::SQLite(), paste0(projhome1,"tweet_er-2.db"))

#t2 <- dbConnect(RSQLite::SQLite(), paste0(projhome2,"tweet_orig.db"))


dbListTables(t)
#dbListTables(t1)
#dbListTables(t2)

dbListFields(t, "complete_ids")



q1<-dbSendQuery(t, "SELECT t1.orig_ids FROM 'original_ids' t1 LEFT OUTER JOIN 'complete_ids' t2 ON t1.orig_ids = cast(t2.done_ids as float) WHERE cast(t2.done_ids as float) IS NULL")

#q2<-dbSendQuery(t1, "SELECT text FROM 'tweets'")
#q3<-dbSendQuery(t2, "SELECT * FROM 'tweets'")

recollect <-  dbFetch(q1)
#tweets <-  dbFetch(q2)
#tweets2 <- dbFetch(q3)

dbClearResult(q1)
dbClearResult(q2)
dbClearResult(q3)

dbDisconnect(t)
dbDisconnect(t1)
dbDisconnect(t2)
rownames(tweets)

class(tweets$text)
