library(dplyr)
curWd <- getwd()
setwd('data/US')
files <- list.files(pattern="^user.*.csv")
usernum <- length(files)
id = 1
for(file in files){
    df <- read.csv(file)
    unlink(file)
    df <- mutate(df, item_index = as.numeric(as.factor(df$track.name)))
    df <- mutate(df, user_id = id)
    df$timestamp <- as.POSIXct(df$timestamp)
    df <- mutate(df, endtime = lag(df$timestamp))
    df <- mutate(df, interval = df$endtime - df$timestamp)
    df <- df[,c('musicbrainz.track.id', 'track.name', 'item_index', "user_id", "interval")]
    df <- df[df$interval > 30, ]
    df <- df[nrow(df):1,]
    rownames(df) <- NULL
    id <- id + 1
    write.csv(df, file)
}
setwd(curWd)

