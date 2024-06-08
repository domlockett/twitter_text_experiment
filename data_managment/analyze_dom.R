library(descr)
library(readr)
#library(RnBeads)
library(magick)
library(dplyr)
library(magrittr)
library(grid)
library(kableExtra)
library(stringr)
library(gridExtra)
library(knitr)
library(ggplot2)
library(gtable)
library(labelled)
library(haven)
library(gmodels)
library(png)
library(tidyverse)
library(cjoint)
library(stargazer)
library(Hmisc)
library(sjPlot)
library(labelled)
library(haven)
library(png)
library(magicfor)    
library(magicfor)  
library(fastDummies)
library(lme4)
library(coefplot)
library(ggtext)
library(RSQLite)
library(dplyr)
library(DBI)

#connect to the DATABASE BY feeding it the address 
con <- dbConnect(RSQLite::SQLite(), 'C:/Users/dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db') #all current




cnx <- dbConnect(RSQLite::SQLite(), 'C:/Users/dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/clean_orig.db') #all current
om <- dbReadTable(con, "tweets-media-o")
# ot<- dbReadTable(cnx, "jan-o")
# oc <- dbReadTable(cnx, "jan-context-o")




dbListTables(cnx)
ot <- dbReadTable(con, "jan-o")
ot <- dbReadTable(cnx, "tweets-o")


#comment length
wt <- dbGetQuery(con, "SELECT AVG(totalwords_o) FROM 'jan-o' ")
wc <- dbGetQuery(con, "SELECT AVG(totalwords), conversation_id FROM 'jan'")


wt$`AVG(totalwords_o)` - wc$`AVG(totalwords)` #difference in post length

#sentiment 

st <- dbGetQuery(cnx, "SELECT t.analysis_o, t.conversation_id_o FROM 'tweets-o' t")
sc<- dbGetQuery(cnx, "SELECT c.analysis, c.conversation_id FROM 'tweets' c")
table(st$analysis_o)/nrow(st)
table(sc$analysis)/nrow(sc)


#Term search 
polWords <- as.data.frame(matrix("",nrow=11,ncol=2))
numTweet <- dbGetQuery(cnx, "SELECT lang_o FROM 'jan-o'")
numComm <- dbGetQuery(cnx, "SELECT text FROM 'jan'")


demT <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o' 
      WHERE text_o 
      LIKE '%democrat%'")


demc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' 
      WHERE text
      LIKE '%democrat%'")


rownames(polWords)[1] <- 'Democrat'
colnames(polWords) <- c('Parent','Comment')

polWords[1,] <-  cbind(paste0(round(demT/nrow(numTweet),3)*100, '%'),paste0(round(demc/nrow(numComm),3)*100, '%'))




rept <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%republican%'")

repc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' t 
      WHERE text
      LIKE '%republican%'")
rownames(polWords)[2] <- 'Republican'
polWords[2,] <-  cbind(paste0(round(rept/nrow(numTweet),3)*100, '%'),paste0(round(repc/nrow(numComm),3)*100, '%'))



trumpt <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%trump%'")

trumpc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' 
      WHERE text
      LIKE '%trump%'")

rownames(polWords)[3] <- 'Trump'
polWords[3,] <-  cbind(paste0(round(trumpt/nrow(numTweet),3)*100, '%'),paste0(round(trumpc/nrow(numComm),3)*100, '%'))




clintt <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%clinton%'")

clintc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' 
      WHERE text
      LIKE '%clinton%'")

rownames(polWords)[4] <- 'Clinton'
polWords[4,] <-  cbind(paste0(round(clintt/nrow(numTweet),5)*100, '%'),paste0(round(clintc/nrow(numComm),5)*100, '%'))




bident <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%biden%'")

bidenc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan'
      WHERE text
      LIKE '%biden%'")


rownames(polWords)[5] <- 'Biden'
polWords[5,] <-  cbind(paste0(round(bident/nrow(numTweet),3)*100, '%'),paste0(round(bidenc/nrow(numComm),3)*100, '%'))




vott <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%vot%'")


votc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' 
      WHERE text
      LIKE '%vot%'")

rownames(polWords)[6] <- 'Vot*'
polWords[6,] <-  cbind(paste0(round(vott/nrow(numTweet),3)*100, '%'),paste0(round(votc/nrow(numComm),3)*100, '%'))



rest <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%research%'")

resc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' 
      WHERE text
      LIKE '%research%'")

rownames(polWords)[7] <- 'Research'
polWords[7,] <-  cbind(paste0(round(rest/nrow(numTweet),5)*100, '%'),paste0(round(resc/nrow(numComm),5)*100, '%'))




prott <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%protest%'")

protc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan'
      WHERE text
      LIKE '%protest%'")


rownames(polWords)[8] <- 'Protest'
polWords[8,] <-  cbind(paste0(round(prott/nrow(numTweet),5)*100, '%'),paste0(round(protc/nrow(numComm),5)*100, '%'))


riott <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%riot%'")

riotc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan'
      WHERE text
      LIKE '%riot%'")


rownames(polWords)[9] <- 'Riot'
polWords[9,] <-  cbind(paste0(round(riott/nrow(numTweet),5)*100, '%'),paste0(round(riotc/nrow(numComm),5)*100, '%'))


antt <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%antifa%'")

antc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan'
      WHERE text
      LIKE '%antifa%'")




rownames(polWords)[10] <- 'Anti-fa'
polWords[10,] <-  cbind(paste0(round(antt/nrow(numTweet),5)*100, '%'),paste0(round(antc/nrow(numComm),5)*100, '%'))



terrt <- dbGetQuery(cnx, "SELECT COUNT(text_o)
      FROM 'jan-o'
      WHERE text_o 
      LIKE '%terrorist%'")

terrc <- dbGetQuery(cnx, "SELECT COUNT(text)
      FROM 'jan' 
      WHERE text
      LIKE '%terrorist%'")


rownames(polWords)[11] <- 'Terrorist'
polWords[11,] <-  cbind(paste0((round(terrt/nrow(numTweet),5))*100, '%'),paste0(round(terrc/nrow(numComm),5)*100, '%'))


polWords


length <- dbGetQuery(cnx, "SELECT text, analysis_o, conversation_id_o  FROM 'jan-o' ")
(table(negTweet$analysis)/nrow(negTweet))*100
negTweet <- dbGetQuery(cnx, "SELECT text_o, analysis_o, analysis, t.conversation_id_o, c.conversation_id
      FROM 'jan-o' t
      LEFT JOIN jan c ON
      t.conversation_id_o = c.conversation_id
      WHERE analysis_o = 'Negative'")
(table(negTweet$analysis)/nrow(negTweet))*100


posTweet <- dbGetQuery(cnx, "SELECT text, analysis_o, analysis, t.conversation_id_o, c.conversation_id
      FROM 'jan-o' t
      LEFT JOIN jan c ON
      t.conversation_id_o = c.conversation_id
      WHERE analysis_o = 'Positive'")
(table(posTweet$analysis)/nrow(posTweet))*100


neutTweet <- dbGetQuery(cnx, "SELECT text, analysis_o, analysis, t.conversation_id_o, c.conversation_id
      FROM 'jan-o' t
      LEFT JOIN jan c ON
      t.conversation_id_o = c.conversation_id
      WHERE analysis_o = 'Neutral'")
(table(neutTweet$analysis)/nrow(neutTweet))*100

