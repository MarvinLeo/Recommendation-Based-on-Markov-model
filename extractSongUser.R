library(dplyr)
curWd <- getwd()
setwd('data')
files <- list.files(pattern="*.csv")
songlist <-factor()
for(file in files){
    df <- read.csv(file)
    songlist <- c(songlist, as.character(df$track.name))
    songlist <- unique(songlist)
}
df <- data.frame( name = songlist, index = as.numeric(as.factor(songlist)))
write.csv(df, "songlist.csv", row.names = FALSE)
setwd(curWd)
