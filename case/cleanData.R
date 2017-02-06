library(dplyr)
library(jsonlite)
oriWd <- getwd()
dir_lis <- list.dirs('.', recursive=FALSE)
track_id <- c("")
similar_id <- c('')
track_similar <- read.table('tracks_with_similar.txt')
track_similar <- as.character(track_similar$V1)
for(dir0 in dir_lis){
    curWd <- getwd()
    setwd(dir0)
    dir_lis1 <- list.dirs('.', recursive=FALSE)
    for(dir1 in dir_lis1){
        curWd1 <- getwd()
        setwd(dir1)
        dir_lis2 <- list.dirs('.', recursive=FALSE)
        for(dir2 in dir_lis2){
            curWd2 <- getwd()
            setwd(dir2)
            files <- list.files(pattern="*.json")
            for(file in files){
                jsonData <- fromJSON(file)
                if(track_id %in% track_similar){
                    track_id <- c(track_id, jsonData$track_id)
                    similar_id <- c(similar_id, jsonData$similars[1,1])
                }
            }
            setwd(curWd2)
        }
        setwd(curWd1)
    }
    setwd(curWd)
    print(curWd)
}

setwd(oriWd)
                
    