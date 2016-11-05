library(ggplot2)
library(lubridate)
library(readr)
library(scales)

train <- read_csv("train.csv")
train$hour  <- hour(ymd_hms(train$datetime))
train$times <- as.POSIXct(strftime(ymd_hms(train$datetime), format="%H:%M:%S"), format="%H:%M:%S")
train$day   <- wday(ymd_hms(train$datetime), label=TRUE)

#People rent bikes for morning/evening commutes on weekdays, and daytime rides on weekends
p <- ggplot(train, aes(x=times, y=count, color=day)) +
     geom_smooth(se=FALSE, fill=NA, size=2) +
     xlab("Time of Day") +
     scale_x_datetime(breaks = date_breaks("4 hours"), labels=date_format("%I:%M %p")) + 
     ylab("Number of Bike Rentals") +
     scale_color_discrete("") 

ggsave("bike_rentals_by_time_of_day.png")