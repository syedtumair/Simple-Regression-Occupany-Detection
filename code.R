library(Metrics)
library(e1071)
library(plotly)

temp = read.csv("Occupancy_Detection.csv")
print("Check Sanity of Data : ")
str(temp)
print("Summary of Data : ")
summary(temp)

#Get only data of House 1 
onlyonehouse_data = subset(temp ,temp$House == "House_1")

#Make Testing data of house 1 with only Usage / #  of adults
usage_adult_data = onlyonehouse_data[,c(2,3)]
testing_data = usage_adult_data

#Make Training data of all houses and take coloumns Usage / # of adults 
allhouse_data = temp
usage_adult_data_allhouse = allhouse_data[,c(2,3)]
training_data = usage_adult_data_allhouse

#Make LM model with adults to usage_kw relation form Training data
lm_model_adult = lm(Adults ~ Usage_kW  , data = training_data)
summary(lm_model_adult)

#Forcast number of adults based on number of UsageKw in testing data (from House 1)
focasted = predict(lm_model_adult , testing_data)
summary(focasted)

#Rounding forcastiing result to nearest value
focasted_round = predict(lm_model_adult , testing_data)
focasted_round = round(focasted_round,0)
summary(focasted_round)

#Adding both focast value to adult data usage which also has actual value
usage_adult_data$Prediction = focasted
usage_adult_data$PredictionRoundedOff = focasted_round

#Checking Metrics

mae(usage_adult_data$Adults, focasted_round)




p1 <- ggplot() + geom_line(aes(y = usage_adult_data$Usage_kW, x = usage_adult_data$Adults),
                           data = usage_adult_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p1

p2 <- ggplot() + geom_line(aes(y = usage_adult_data$Usage_kW, x = usage_adult_data$PredictionRoundedOff),
                           data = usage_adult_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p2

######## LM model training for Childen  #############

#Get only data of House 1 
onlyonehouse_data = subset(temp ,temp$House == "House_1")

#Make Testing data of house 1 with only Usage / #  of child
usage_child_data = onlyonehouse_data[,c(2,4)]
testing_data_child = usage_child_data

#Make Training data of all houses and take coloumns Usage / # of child 
allhouse_data = temp
usage_child_data_allhouse = allhouse_data[,c(2,4)]
training_data_child = usage_child_data_allhouse

#Make LM model with child to usage_kw relation form Training data
lm_model_child = lm(Children ~ Usage_kW  , data = training_data_child)
summary(lm_model_child)

#Forcast number of child based on number of UsageKw in testing data (from House 1)
focasted = predict(lm_model_child , testing_data_child)
summary(focasted)

#Rounding forcastiing result to nearest value
focasted_round = predict(lm_model_child , testing_data_child)
focasted_round = round(focasted_round,0)
summary(focasted_round)

#Adding both focast value to child data usage which also has actual value
usage_child_data$Prediction = focasted
usage_child_data$PredictionRoundedOff = focasted_round
mae(usage_child_data$Children, focasted_round)



p3 <- ggplot() + geom_line(aes(y = usage_child_data$Usage_kW, x = usage_child_data$Childern),
                           data = usage_child_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p3

p4 <- ggplot() + geom_line(aes(y = usage_child_data$Usage_kW, x = usage_child_data$PredictionRoundedOff),
                           data = usage_child_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p4



mae(usage_child_data$Children, focasted_round)


######----##### SVR ######----#######


svr_model_adult = svm(Adults ~ Usage_kW, data = training_data, type = "eps-regression", 
                kernel = "linear")
summary(svr_model_adult)
forecasted_adult_svr = predict(svr_model_adult,testing_data)
#Adding SVM Forcasted Value to final Data
usage_adult_data$svrForcased_adults = forecasted_adult_svr


mae(usage_adult_data$Adults, forecasted_adult_svr)

 #for childern 
svr_model_child = svm(Children ~ Usage_kW, data = training_data_child, type = "eps-regression", 
                      kernel = "linear")
summary(svr_model_child)
forecasted_child_svr = predict(svr_model_child,testing_data_child)
mea(usage_adult_data$Adults , usage_adult_data$svrForcased_adults )
mea(usage_child_data$Children , usage_child_data$svrForcased_child)
#Adding SVM Forcasted Value to final Data
usage_child_data$svrForcased_child = forecasted_child_svr

mae(usage_adult_data$Adults , usage_adult_data$svrForcased_adults )
mae(usage_child_data$Children , usage_child_data$svrForcased_child)

p6 <- ggplot() + geom_line(aes(y = usage_adult_data$Usage_kW, x = usage_adult_data$Adults),
                           data = usage_adult_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))

p6

p6 <- ggplot() + geom_line(aes(y = usage_adult_data$Usage_kW, x = usage_adult_data$svrForcased_adults),
                           data = usage_adult_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p6


p5 <- ggplot() + geom_line(aes(y = usage_child_data$Usage_kW, x = usage_child_data$Childern),
                           data = usage_child_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p5

p5 <- ggplot() + geom_line(aes(y = usage_child_data$Usage_kW, x = usage_child_data$svrForcased_child),
                           data = usage_child_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p5


####-----ANN-------###
install.packages("neuralnet")
library("neuralnet")
nn_model_adult = neuralnet(Adults ~ Usage_kW, data = training_data, 
                     hidden = 3, linear.output = TRUE)

forecasted_nn_adult = predict(nn_model_adult , testing_data)

#Adding NN Forcasted Value to final Data
usage_adult_data$forecasted_nn_adult = forecasted_nn_adult


nn_model_childern = neuralnet(Children ~ Usage_kW, data = training_data_child, 
                           hidden = 3, linear.output = TRUE)

forecasted_nn_child = predict(nn_model_childern , testing_data_child)

#Adding NN Forcasted Value to final Data
usage_child_data$forecasted_nn_child = forecasted_nn_child


#Final Results
mae(usage_adult_data$Adults , usage_adult_data$forecasted_nn_adult )
mae(usage_child_data$Children , usage_child_data$forecasted_nn_child)

p7 <- ggplot() + geom_line(aes(y = usage_adult_data$Usage_kW, x = usage_adult_data$Adults),
                           data = usage_adult_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))

p7

p8 <- ggplot() + geom_line(aes(y = usage_adult_data$Usage_kW, x = usage_adult_data$forecasted_nn_adult),
                           data = usage_adult_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p8


p9 <- ggplot() + geom_line(aes(y = usage_child_data$Usage_kW, x = usage_child_data$Childern),
                           data = usage_child_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p9

p10 <- ggplot() + geom_line(aes(y = usage_child_data$Usage_kW, x = usage_child_data$forecasted_nn_child),
                           data = usage_child_data) + scale_x_continuous(breaks=seq(0,10,0.2)) + scale_y_continuous(breaks=seq(0,8,0.2))
p10




