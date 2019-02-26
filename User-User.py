
#%%
import pandas as pd
import numpy as np
import math


#%%
Ratings=pd.read_csv("~/WSU_SPRING_2019/DATA_MINING/hw2/Recommender-Systems/ratings.csv",encoding="ISO-8859-1")
Movies=pd.read_csv("~/WSU_SPRING_2019/DATA_MINING/hw2/Recommender-Systems/movies.csv",encoding="ISO-8859-1")
Tags=pd.read_csv("~/WSU_SPRING_2019/DATA_MINING/hw2/Recommender-Systems/tags.csv",encoding="ISO-8859-1")


''' Calculating the mean rating and subtracting from each rating of a user to calculate the adjusted rating. '''
#%%
Mean= Ratings.groupby(['userId'], as_index = False, sort = False).mean().rename(columns = {'rating': 'rating_mean'})[['userId','rating_mean']]
Ratings = pd.merge(Ratings,Mean,on = 'userId', how = 'left', sort = False)
Ratings['rating_adjusted']=Ratings['rating']-Ratings['rating_mean']
Ratings

''' Finding the top 30 similar user profiles for each user. '''
#%%
distinct_users=np.unique(Ratings['userId'])

user_data_append=pd.DataFrame()


user_data_all=pd.DataFrame()
    
user1_data=  Ratings[Ratings['userId']==320]
user1_mean=user1_data["rating"].mean()
user1_data=user1_data.rename(columns={'rating_adjusted':'rating_adjusted1'})
user1_data=user1_data.rename(columns={'userId':'userId1'})
user1_val=np.sqrt(np.sum(np.square(user1_data['rating_adjusted1']), axis=0))


distinct_movie=np.unique(Ratings['movieId'])

i=1

for movie in distinct_movie[91:92]:

    item_user =  Ratings[Ratings['movieId']==movie]

    distinct_users1=np.unique(item_user['userId'])
    
    j=1
    
    for user2 in distinct_users1:

        if j%200==0:

            print (j , "out of ", len(distinct_users1), i , "out of ", len(distinct_movie))

        user2_data=  Ratings[Ratings['userId']==user2]
        user2_data=user2_data.rename(columns={'rating_adjusted':'rating_adjusted2'})
        user2_data=user2_data.rename(columns={'userId':'userId2'})
        user2_val=np.sqrt(np.sum(np.square(user2_data['rating_adjusted2']), axis=0))

        user_data = pd.merge(user1_data,user2_data[['rating_adjusted2','movieId','userId2']],on = 'movieId', how = 'inner', sort = False)
        user_data['vector_product']=(user_data['rating_adjusted1']*user_data['rating_adjusted2'])


        user_data= user_data.groupby(['userId1','userId2'], as_index = False, sort = False).sum()

        user_data['dot']=user_data['vector_product']/(user1_val*user2_val)


        user_data_all = user_data_all.append(user_data, ignore_index=True)

        j=j+1

    user_data_all=  user_data_all[user_data_all['dot']<1]
    user_data_all = user_data_all.sort_values(['dot'], ascending=False)
    user_data_all = user_data_all.head(30)
    user_data_all['movieId']=movie
    user_data_append = user_data_append.append(user_data_all, ignore_index=True)
    i=i+1

''' Calculating the predicted rating for each item and ignoring the item if less than 2 similar neighbours. '''
#%%
User_dot_adj_rating_all=pd.DataFrame()

distinct_movies=np.unique(Ratings['movieId'])

j=1
for movie in distinct_movies[91:92]:
    
    user_data_append_movie=user_data_append[user_data_append['movieId']==movie]
    User_dot_adj_rating = pd.merge(Ratings,user_data_append_movie[['dot','userId2','userId1']], how = 'inner',left_on='userId', right_on='userId2', sort = False)
    
    if j%200==0:
    
        print (j , "out of ", len(distinct_movies))
        
    User_dot_adj_rating1=User_dot_adj_rating[User_dot_adj_rating['movieId']==movie]
    
    if len(np.unique(User_dot_adj_rating1['userId']))>=2:
        
        User_dot_adj_rating1['wt_rating']=User_dot_adj_rating1['dot']*User_dot_adj_rating1['rating_adjusted']
        
        User_dot_adj_rating1['dot_abs']=User_dot_adj_rating1['dot'].abs()
        User_dot_adj_rating1= User_dot_adj_rating1.groupby(['userId1'], as_index = False, sort = False).sum()[['userId1','wt_rating','dot_abs']]
        User_dot_adj_rating1['Rating']=(User_dot_adj_rating1['wt_rating']/User_dot_adj_rating1['dot_abs'])+user1_mean
        User_dot_adj_rating1['movieId']=movie
        User_dot_adj_rating1 = User_dot_adj_rating1.drop(['wt_rating', 'dot_abs'], axis=1)
        
        User_dot_adj_rating_all = User_dot_adj_rating_all.append(User_dot_adj_rating1, ignore_index=True)
    
    j=j+1
        
User_dot_adj_rating_all = User_dot_adj_rating_all.sort_values(['Rating'], ascending=False)

#%%
userID = 1203
distinct_movie.tolist().index(userID)
#%%
User_dot_adj_rating_all

#%%
[i for i,x in enumerate(distinct_movie) if x == 4878]
