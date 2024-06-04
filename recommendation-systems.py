#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

#Data review
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
user_data = pd.read_csv('Data/u.data', sep='\t', names=column_names)
print(user_data.head())

movie_title = pd.read_csv('Data/Movie_Id_Titles')
print(movie_title.head())

#Merge data sets    
merge_title_user_data = pd.merge(user_data, movie_title, on='item_id')
print(merge_title_user_data.head())

#Setting up the data
mean_data_values = merge_title_user_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
count_data_values = merge_title_user_data.groupby('title')['rating'].count().sort_values(ascending=False).head()
print('Mean rating movies', mean_data_values)
print('Count movies',count_data_values)
ratings = pd.DataFrame(merge_title_user_data.groupby('title')['rating'].mean())
print('Ratings mean',ratings.head())
ratings['num of ratings'] = pd.DataFrame(merge_title_user_data.groupby('title')['rating'].count())
print('Ratings numbers',ratings.head())

#Data visualization
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70) 

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
#plt.show()

#Classification movies
moviemat = merge_title_user_data.pivot_table(index='user_id',columns='title',values='rating')
print(moviemat.head())
ratings.sort_values('num of ratings',ascending=False).head(10)
print(ratings.head())

#Two movies valoration
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
print(starwars_user_ratings.head())

#Correlation between movies     
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

#Correlation movie one
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())

#Organize data  
print(corr_starwars.sort_values('Correlation',ascending=False).head(10))

#Filtering out movies
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars.head())

#Sorting values
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())

#Correlation movie two
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()) 