################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(! require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(! require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# learn basic information
edx_names <- names(edx)
edx %>% summarize (n_users=n_distinct(userId), n_movies = n_distinct(movieId))

# edx data is divided into train_set (95%) and test_set(5%) 
set.seed(100)
edx_testindex <- createDataPartition(y = edx$rating, times = 1, p = 0.05, list = FALSE)
train_set <- edx[-edx_testindex,]
edx_temp <- edx[edx_testindex,]
test_set <- edx_temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
edx_removed <- anti_join(edx_temp, test_set)
train_set <- rbind(train_set, edx_removed)

# RMSE function
RMSE <- function(true_ratings, predicted_ratings){sqrt(mean((true_ratings-predicted_ratings)^2))}

# nrating records the number of each rating in train_set
nrating <- train_set %>% group_by(rating) %>% summarize(n=n())%>% .$n
# plot the number of ratings versus ratings
barplot(nrating,names.arg = c("0.5","1","1.5","2","2.5","3","3.5","4","4.5","5"),col="black",xlab = "Rating scores",ylab = "The number of each rating")
title(main = "Number of ratings versus ratings")

# mu is the average rating 
mu <- mean(train_set$rating)
# model 1. calculate RMSE of test_set using naive model
edx_rmse_1 <- RMSE(test_set$rating, test_set%>%mutate(newrating=mu)%>%.$newrating)

# plot average rating for each movie in train_set
train_set %>% group_by(movieId) %>% summarize(mr = (mean(rating)))%>% ungroup() %>%  ggplot(aes(mr)) + xlab("average rating for each movie") + geom_histogram(bins=30) + labs(title ="Average rating for each movie in train_set") + geom_vline(aes(xintercept = mu), color="red",linetype = "dashed")

# plot average rating score versus months
train_set %>% mutate(date = as_datetime(timestamp)) %>% mutate(date = round_date(date, unit = "month")) %>%
       group_by(date) %>%
       summarize(rating = mean(rating)) %>%
       ggplot(aes(date, rating)) +
       geom_point() +
       geom_smooth() + labs(title = "Average rating change with months") + theme(plot.title = element_text(hjust = 0))

# model 2: add terms about movie, user, genre and date respectively
# calculate b_i, b_u, g_ui, d_ui
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i=mean(rating-mu))
user_avgs <- train_set %>% left_join(movie_avgs,by="movieId")%>% group_by(userId) %>% summarize(b_u = mean(rating-mu - b_i))
genre_avgs <- train_set %>% left_join(movie_avgs,by="movieId") %>% left_join(user_avgs,by = "userId") %>% group_by(genres) %>% summarize(g_ui = mean(rating- mu - b_i - b_u))
# date_avgs: first create a new object date and group them in months
date_avgs <- train_set %>% left_join(movie_avgs,by="movieId") %>% left_join(user_avgs,by = "userId") %>% left_join(genre_avgs, by = "genres") %>% mutate(date = as_datetime(timestamp))%>%mutate(date = round_date(date, unit = "month")) %>% group_by(date) %>% summarize(d_ui = mean(rating-mu-b_i-b_u-g_ui))
# calculate RMSE for test_set using model 2
predicted_ratings <- test_set %>% mutate(date=as_datetime(timestamp))%>% mutate(date = round_date(date,unit="month"))%>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>% left_join(genre_avgs, by="genres") %>% left_join(date_avgs, by = "date") %>% mutate(pred = mu + b_i + b_u + g_ui + d_ui)%>%.$pred
edx_rmse_2 <- RMSE(predicted_ratings, test_set$rating)

# model 3. regularization
 lambdas <- seq(0,10,1) # choose eleven different lambdas
 rmses <- sapply(lambdas,function(l){
 movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i=sum(rating-mu)/(n()+l))
 user_avgs <- train_set %>% left_join(movie_avgs,by="movieId")%>% group_by(userId) %>% summarize(b_u = sum(rating-mu - b_i)/(n()+l))
 genre_avgs <- train_set %>% left_join(movie_avgs,by="movieId") %>% left_join(user_avgs,by = "userId") %>% group_by(genres) %>% summarize(g_ui = sum(rating- mu - b_i - b_u)/(n()+l))
 date_avgs <- train_set %>% left_join(movie_avgs,by="movieId") %>% left_join(user_avgs,by = "userId") %>% left_join(genre_avgs, by = "genres") %>% mutate(date = as_datetime(timestamp))%>%mutate(date = round_date(date, unit = "month")) %>% group_by(date) %>% summarize(d_ui = sum(rating-mu-b_i-b_u-g_ui)/(n()+l))
 predicted_ratings <- test_set %>% mutate(date=as_datetime(timestamp))%>% mutate(date = round_date(date,unit="month"))%>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>% left_join(genre_avgs, by="genres") %>% left_join(date_avgs, by = "date") %>% mutate(pred = mu + b_i + b_u + g_ui + d_ui) %>%.$pred
 return(RMSE(predicted_ratings, test_set$rating))})
 # plot RMSE versus lambdas
 d <- data.frame(lambdas = lambdas, rmses = rmses)
 d %>% ggplot(aes(lambdas, rmses)) +
   geom_point() + labs(title = "RMSE versus lambdas") + theme(plot.title = element_text(hjust = 0))
 # find the best lambda
 index <- which.min(rmses)
 lambda <- lambdas[index]
 # RMSE of test_set with model 3
 edx_rmse_3 <- rmses[index]
 
 # calculate RMSE for validation set with the omptimal lambda 
 movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i=sum(rating-mu)/(n()+lambda))
 user_avgs <- train_set %>% left_join(movie_avgs,by="movieId")%>% group_by(userId) %>% summarize(b_u = sum(rating-mu - b_i)/(n()+lambda))
 genre_avgs <- train_set %>% left_join(movie_avgs,by="movieId") %>% left_join(user_avgs,by = "userId") %>% group_by(genres) %>% summarize(g_ui = sum(rating- mu - b_i - b_u)/(n()+lambda))
 date_avgs <- train_set %>% left_join(movie_avgs,by="movieId") %>% left_join(user_avgs,by = "userId") %>% left_join(genre_avgs, by = "genres") %>% mutate(date = as_datetime(timestamp))%>%mutate(date = round_date(date, unit = "month")) %>% group_by(date) %>% summarize(d_ui = sum(rating-mu-b_i-b_u-g_ui)/(n()+lambda))
valid_predicted <- validation %>% mutate(date=as_datetime(timestamp))%>% mutate(date = round_date(date,unit="month"))%>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>% left_join(genre_avgs, by="genres") %>% left_join(date_avgs, by = "date") %>% mutate(pred = mu + b_i + b_u + g_ui + d_ui) %>%.$pred
# valid_rmse is the final RMSE value
valid_rmse <- RMSE(valid_predicted, validation$rating)

rm(d, date_avgs, edx_removed, edx_testindex, genre_avgs, movie_avgs, user_avgs, edx_names, index, lambdas, nrating, predicted_ratings, rmses)

# print RMSE score
print(valid_rmse)
