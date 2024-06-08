from sqlalchemy import create_engine, Table, Column, Integer, Unicode, MetaData, String, Text, update, and_, select, func, types

# create engine, reflect existing columns, and create table object for oldTable
srcEngine = create_engine('sqlite:///data/sql/tweet.db') # change this for your source database
srcEngine._metadata = MetaData(bind=srcEngine)
srcEngine._metadata.reflect(srcEngine) # get columns from existing table
srcTable1 = Table('tweets', srcEngine._metadata)
srcTable2 = Table('tweets-media', srcEngine._metadata)
srcTable3 = Table('tweets-context', srcEngine._metadata)

# create engine and table object for newTable
destEngine = create_engine('sqlite:///data/sql/jan6_tweet.db') # change this for your destination database
destEngine._metadata = MetaData(bind=destEngine)
destTable1 = Table('tweet-o', destEngine._metadata)
destTable2 = Table('tweets-media-o', destEngine._metadata)
destTable3 = Table('tweets-context-o', destEngine._metadata)

# copy schema and create newTable from oldTable
for column in srcTable1.columns:
    destTable1.append_column(column.copy())
destTable1.create()

for column in srcTable2.columns:
    destTable2.append_column(column.copy())
destTable2.create()

for column in srcTable3.columns:
    destTable3.append_column(column.copy())
destTable3.create()