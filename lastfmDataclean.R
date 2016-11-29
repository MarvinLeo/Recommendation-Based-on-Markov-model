library(dplyr)

#read the dataset put the data into working directory first
fileName <- "userid-timestamp-artid-artname-traid-traname.tsv"
test <- read.csv(fileName, header = FALSE, sep = "\t")
name <- c('userid', 'timestamp', 'musicbrainz-artist-id', 'artist-name', 'musicbrainz-track-id', 'track-name')
colnames(test) <- name

#find the user's information
userInfo <- read.csv("userid-profile.tsv", sep = "\t")
user <- userInfo$X.id

#find all the songs X
trackName <- unique(test$'track-name')
#separate data by users
test <- split(test, test$userid)

#change the time type and save it into seperate csv file
for(df in test){
    #deal with time
    df$timestamp <- as.character(df$timestamp)
    df$timestamp <- gsub('T|Z', ' ', df$timestamp)
    df$timestamp <- as.POSIXlt(df$timestamp, format="%Y-%m-%d %H:%M:%S")  
    destName <- as.character(df$userid[1])
    destName <- paste(destName, '.csv', sep = '')
    write.csv(df, destName, row.names = FALSE)
}
