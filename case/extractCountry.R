library(dplyr)
userid_pro <- read.csv('userid-profile.tsv', sep = '\t')
user_america <- userid_pro[userid_pro$country == 'United States', ]
user_uk <- userid_pro[userid_pro$country == 'United Kingdom', ]
curwd <- getwd()
setwd('data')
curwd1 <- getwd()
if(!dir.exists('US')){dir.create('US')}
if(!dir.exists('UK')){dir.create('UK')}
files <- list.files(pattern="*.csv")
for (id in as.character(user_america$X.id)) {
    id_tag = paste(id, '.csv', sep = "")
    if(id_tag %in% files){
        file.copy(id_tag, paste("US/", id_tag, sep = ""), overwrite = TRUE)
    }
}
for (id in as.character(user_uk$X.id)) {
    id_tag = paste(id, '.csv', sep = "")
    if(id_tag %in% files){
        file.copy(id_tag, paste("UK/", id_tag, sep = ""), overwrite = TRUE)
    }
}
setwd(curwd)
userid_pro <- group_by(userid_pro, country)
user_country <- arrange(summarise(userid_pro, n()), desc(`n()`))