import pyspark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import pyspark.sql.functions as F
from math import sqrt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

conf = SparkConf()
sc = SparkContext(conf=conf)

spark = SparkSession \
    .builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .appName("MovieLens CF") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

Movies_df = spark.read.csv("ml-latest-small/movies.csv",header=True)
Ratings_df = spark.read.csv("ml-latest-small/ratings.csv",header=True)

Ratings_df=Ratings_df.withColumn('rating', Ratings_df['rating'].cast("float"))
Ratings_df=Ratings_df.withColumn('userId', Ratings_df['userId'].cast("integer"))
Ratings_df=Ratings_df.withColumn('movieId', Ratings_df['movieId'].cast("integer"))
Ratings_df = Ratings_df.drop(*['timestamp'])

user_ratings = Ratings_df.groupBy("userId").count()

movie_ratings = Ratings_df.groupBy("movieId").count()

# Create test and train set
(Train, Test) = Ratings_df.randomSplit([0.9, 0.1], seed = 0)

Train_user_ratings = Train.groupBy("userId").count()
Train_users=Train_user_ratings.count()
Train_user_ratings=Train_user_ratings.select('userId').rdd.map(lambda x: x[0]).collect()

Test_user_ratings = Test.groupBy("userId").count()
Test_users=Test_user_ratings.count() 
Test_user_ratings=Test_user_ratings.select('userId').rdd.map(lambda x: x[0]).collect()

Train_movie_ratings = Train.groupBy("movieID").count()
Train_movie=Train_movie_ratings.count()
Train_movie_ratings=Train_movie_ratings.select('movieID').rdd.map(lambda x: x[0]).collect()

Test_movie_ratings = Test.groupBy("movieID").count()
Test_movie=Test_movie_ratings.count()
Test_movie_ratings=Test_movie_ratings.select('movieID').rdd.map(lambda x: x[0]).collect()

# We need to check if the Test df doesn't have new users or new movies that aren't in the Train set

def NotUnique(TrainList,TestList):
    
    for item in TestList:
        if item not in TrainList:
            print('This Train Test Split is not perfect (a part of the test Dataset will have to be ignored)')
            return
    print('This Train Test Split is perfect')

# Check for users
NotUnique(Train_user_ratings,Test_user_ratings)
# Check for movies
NotUnique(Train_movie_ratings,Test_movie_ratings)

def transformRating(Id_1,rating,Id_2,items):
    rating_list = [rating if ele == Id_1 else None for ele in items]
    return ([Id_2]+[rating_list])

def RatingJunction(a,b):

    
    n=0
    for ind in b:   #Here I didn't run the whole list but only up to the index of b, this works because b is always an RDD with only one entry. Helps reduce computing time
        if ind != None:
            break
        n=n+1

    c=a
    c[n]=b[n]

    return c

# This List has all of the movies in the dataset, in the order that will appear in the Train_user_RDD
items_movies = Train.select('movieId').rdd.map(lambda data:data.movieId).collect()
items_movies = list(dict.fromkeys(items_movies))
item_movies_len = len(items_movies)

Train_user_RDD =Train.rdd.map(lambda data:(data.movieId,data.rating,data.userId))

Train_user_RDD=Train_user_RDD.map((lambda data:transformRating(data[0],data[1],data[2],items_movies)))

Train_user_RDD=Train_user_RDD.map(lambda item: (item[0],item[1]))

Train_user_RDD=Train_user_RDD.reduceByKey(lambda data_1,data_2:RatingJunction(data_1,data_2))

df_user_to_show=Train_user_RDD.toDF()

# This List has all of the users in the dataset, in the order that will appear in the Train_movie_RDD
items_users = Train.select('userId').rdd.map(lambda data:data.userId).collect()
items_users = list(dict.fromkeys(items_users))
item_users_len = len(items_users)

Train_movie_RDD =Train.rdd.map(lambda data:(data.userId,data.rating,data.movieId))

Train_movie_RDD=Train_movie_RDD.map((lambda data:transformRating(data[0],data[1],data[2],items_users)))

Train_movie_RDD=Train_movie_RDD.map(lambda item: (item[0],item[1]))

Train_movie_RDD=Train_movie_RDD.reduceByKey(lambda data_1,data_2:RatingJunction(data_1,data_2))

df_movie=Train_movie_RDD.toDF()

def Pearson_step1(item):
    ratings=item[1]
    ratings_Ex = list(filter(None,ratings))
    mean=sum(ratings_Ex)/len(ratings_Ex)
    n=0
    for rat in ratings:
        if rat != None:
            ratings[n]=ratings[n]-mean
        else:
            ratings[n]=0.0
        n=n+1
    return (item[0], ratings)

Similarity_RDD = Train_movie_RDD.map(lambda item: Pearson_step1(item))

df_movie_s=Similarity_RDD.toDF()

# Join DFs
df_join = df_movie_s.crossJoin(df_movie_s.select('_1', F.col("_2").alias("ratings_2")))

Data_list = ["movieId_1","ratings_1","movieId_2","ratings_2"]
 
df_join = df_join.toDF(*Data_list)

JoinedRDD= df_join.rdd.map(lambda x: ((x.movieId_1,x.movieId_2),x.ratings_1,x.ratings_2))

def cosine_sim(item):
    rating_1=item[1]
    rating_2=item[2]

    #prod is the dividend of cosine similarity
    prod_list=[]
    for n in range(0,item_users_len):
        number=rating_1[n]*rating_2[n]
        prod_list.append(number)
    prod=sum(prod_list)

    #prod2 is the divider of cosine similarity
    square_1=sqrt(sum([ x**2 for x in rating_1 ]))
    square_2=sqrt(sum([ x**2 for x in rating_2 ]))

    prod2=square_1*square_2

    # if prod2 is 0, we can't use it as the divider, so we change it to a very small number
    if prod2==0:
        prod2=0.000000000000000001

    similarity=prod/prod2

    return (item[0],similarity)

similarityRDD=JoinedRDD.map(lambda data: cosine_sim(data))

#Correlation values smaller than 0.30 are considered weak
similarity_filter_RDD=similarityRDD.filter(lambda x: x[1]>0.30)

# This dictionary has all of the meaningfull similarities between movies, the similarities are duplicated (each similarity appears 2 times, but with the items displaied in a different order)

Similarity_Dict=similarity_filter_RDD.collectAsMap()

def scores(item):
    
    user=item[0]
    ratings_change=item[1]
    ratings_non_change=ratings_change[:]
    
    for n in range(0,item_movies_len):
        
        if ratings_change[n]==None:
            
            i=items_movies[n]            
            i_dict = {}
            
            # i_dict is going to be a list with top 10 similaritys with movie i, that the user saw
            for item, value in Similarity_Dict.items():
                if (item[0] == i) and ratings_non_change[items_movies.index(item[1])]!=None:
                    i_dict[item] = (value, ratings_non_change[items_movies.index(item[1])])                 
            i_dict = sorted(i_dict.items(), key=lambda x:-x[1][0])[:10]           
            
            # calculate score
            term1=0
            term2=0
            for item, value in i_dict:
                term1=term1+(value[0]*value[1])
                term2=term2+value[0]

            #if the divider is 0, we have to change it to a very small number to continue the calculations    
            if term2==0:
                term2=0.0000000000000000001
                
            score=term1/term2
            ratings_change[n]=score
            
        else:
            ratings_change[n]=-1
            
    return (user,ratings_change)

ScoresRDD=Train_user_RDD.map(lambda data: scores(data))

#Change here the users you want to check

User_Check = [1,2,3,4,5,6,7,8,9,10]

# Create RDD with just the users we want to check

User_Scores_RDD=ScoresRDD.filter(lambda x: x[0] in User_Check)

User_Scores = User_Scores_RDD.take(len(User_Check))

# Create Dictionary Movie Code:Movie Name

Movies_df=Movies_df.drop(*['genres'])

Movies_CodeRDD = Movies_df.rdd.map(tuple)

Movies_Code_Dict=Movies_CodeRDD.collectAsMap()

for item in User_Scores:
    user=item[0]
    print('Some movies with highest recomendation to user: ', user)
    ratings=item[1]
    n=0
    recomended=[]
    for rate in ratings:
        if rate>=4.5:  #Change here the threshold of score you want to analyse
            recomended.append(Movies_Code_Dict[str(items_movies[n])])
        n=n+1
    m=0
    for movie in recomended:
        if m>20:
            print('More ', len(recomended)-20, ' movies.' )
            break
        print(movie, end = ' | ')   
        m=m+1
    print()
    print()

Test_RDD =Test.rdd.map(lambda data:(data.userId, data.movieId, data.rating,))

Test_RDD=Test_RDD.filter(lambda x: x[0] in User_Check)

user_ordered = [item[0] for item in User_Scores]

def PredictedScore(item):
    
    if item[0] in Train_user_ratings and item[1] in Train_movie_ratings:
        user_index=user_ordered.index(item[0])
        
        predicted=User_Scores[user_index][1][items_movies.index(item[1])]                       

        if predicted==0:
            # In this case the algorithm wasn't able to discover a score, becase the similaritys between this movie and other movies are low.
            return(None, None, None, None)
    
        return (item[0], item[1], item[2], predicted)
    
    # This rating ins't possible to get because the Train dataset dind't had the movie related to it 
    else:
        return(None, None, None, None)

Test_RDD=Test_RDD.map(lambda data: PredictedScore(data))

Test_RDD = Test_RDD.filter( lambda x: x[0]!=None)

Test_predict = Test_RDD.toDF()

Test_predict=Test_predict.toDF(*['userId','movieId','rating','prediction'])

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 

RMSE = evaluator.evaluate(Test_predict)
print('RMSE', RMSE)

Top10_list=[]
Top10_user_list=[]
Top10_len_list=[]
for userID in User_Check:

    userID_RDD = Test_RDD.filter(lambda x: x[0]== userID)
    userID_RDD = userID_RDD.filter(lambda x: x[3]>= 3.5)
    userID_RDD = userID_RDD.map(lambda x: [x[2], x[3]])
    userID_list = userID_RDD.collect()

    if len(userID_list)>=1:
        userID_list = sorted(userID_list, key=lambda x: x[1], reverse=True)
        userID_list = userID_list[:10]
        quantity=0
        pred=0
        for x in userID_list:
            if x[0]>= 3.5: # defined threshold
                pred = pred+1
            quantity=quantity+1
        Top10_list.append(pred/quantity)
        Top10_user_list.append((userID,pred/quantity))
        Top10_len_list.append((userID,len(userID_list)))


print('Length of the predicted data:')
print(Top10_len_list)

print()

print('Top 10 algorithm by user:')
print(Top10_user_list)

print()

print('Top 10 algorithm mean:')
Top10_mean=sum(Top10_list)/len(Top10_list)

print(Top10_mean)

