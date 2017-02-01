library(dplyr)
curWd <- getwd()
setwd('data/US')
if(file.exists('songUser.csv')){
    unlink('songUser.csv')
}
files <- list.files(pattern="*.csv")
files <- files[files != 'songUserRevised.csv' & files != 'songlist.csv' ]
#songlist <-factor()
id = 1
for(file in files){
    if(!exists('songList')){
        songList <- read.csv(file)
        songList <- songList[songList$musicbrainz.track.id!="",1:6]
        songList <- group_by(songList, musicbrainz.track.id)
        songList <- summarise(songList, n())
        colnames(songList) <- c('name', 'times')
        songList <- songList[songList$times != 1,]
        print(id)
        id <- id+1
    }
    
    else{
        df1 <- read.csv(file)
        df1 <- df1[df1$musicbrainz.track.id!="",1:6]
        df1 <- group_by(df1, musicbrainz.track.id)
        df1 <- summarise(df1, n())
        colnames(df1) <- c('name', 'times')
        df1 <- df1[df1$times != 1,]
        songList <- merge(songList, df1, all = TRUE, by = c('name'))
        print(id)
        id <- id+1
        print(object.size(songList), units = "Mb")
    }
#    df <- read.csv(file)
#    songlist <- c(songlist, as.character(df$track.name))
#    songlist <- unique(songlist)
}
#df <- data.frame( name = songlist, index = as.numeric(as.factor(songlist)))
songList[is.na(songList)] <- 0
name <- as.character(1:length(songList[1,]))
colnames(songList) <- name
write.csv(songList, "songUser.csv", row.names = FALSE)
setwd(curWd)
