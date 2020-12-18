import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
def compute_f_mat(mat_rat,user_count,movie_count):
    """
    compute the f matrix
    :param mat_rat: user`s rating matrix([user number,movie number]) where 1 means user likes the index movie.
    :param user_count: statistics of moive numbers that user have watch.
    :param movie_count: statistics of user numbers that movie have been rated.
    :return: f matrix
    """
    temp = (mat_rat / user_count.reshape([-1,1]) )/ movie_count.reshape([1,-1])
    W = np.dot(mat_rat.T, temp)

    f = np.dot(W, mat_rat.T).T

    return f

def roc_pic(f_mat,user_count,mat_rat,mat_dislike,num = 80):
    threshold_rate = np.linspace(0,1,num)

    sort_result = np.argsort(-f_mat, axis=1)
    th_fprs = np.zeros(num)
    th_tprs = np.zeros(num)
    for i,threshold in enumerate(threshold_rate):
        recommond_num = int(mat_rat.shape[1] * threshold)
        fprs = np.zeros(user_count.shape[0])
        tprs = np.zeros(user_count.shape[0])
        for user in range(user_count.shape[0]):
            recommond_movie = sort_result[user,0:recommond_num]#recommand movies
            user_like = np.where(mat_rat[user,:] == 1)[0]
            user_dislike = np.where(mat_dislike[user,:] == 1)[0]

            like = np.intersect1d(recommond_movie, user_like)
            dis_like = np.intersect1d(recommond_movie, user_dislike)
            if len(user_dislike) ==0:
                fprs[user] = 0
            else:
                fprs[user] = len(dis_like) / len(user_dislike)
            if len(user_like) ==0:
                tprs[user] = 0
            else:
                tprs[user] = len(like) / len(user_like)

        th_fprs[i] = fprs.mean()
        th_tprs[i] = tprs.mean()
    roc_auc = auc(th_fprs,th_tprs) #compute the roc value
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(th_fprs, th_tprs, color='red',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='aqua', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Link prediction-ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    #导入数据
    curpath=os.path.abspath('.')#当前地址
    filename = curpath+"\\ml-1m\\ratings.dat"
    all_ratings = pd.read_csv(filename, header=None, sep="::", names=["UserId",  "MovieId",  "Rating",  "Datetime"], engine="python")
    '''
    #print1
    #shape of all_ratings and The first several lines of all_ratings
    print(all_ratings.shape)
    print(all_ratings.head())
    '''

    #以UsersId随机分类
    train_ratings, test_ratings, _, _ = train_test_split(all_ratings, all_ratings['UserId'], test_size=0.1)
    '''
    #print2
    #The ratio of test set to training set
    print('tarin_ratings : test_ratings =\n',len(train_ratings),':',len(test_ratings))
    '''

    userId_col = all_ratings['UserId']
    movieId_col = all_ratings['MovieId']

    user_count = np.array(userId_col.value_counts())  # count number，every element of array meas number of this ID index
    movie_count = np.array(movieId_col.value_counts())  # count number，every element of array meas number of this ID index
    movie_index = np.array(movieId_col.value_counts().index)

    userId_max = user_count.shape[0]  # all number
    movieId_max = movie_count.shape[0]  # all number

    mat = np.zeros([userId_max, movieId_max])#create empty matrix

    #count the rating of users
    for row in train_ratings.itertuples(index=True, name='Pandas'):
        mat[row.UserId - 1, np.where(movie_index == row.MovieId)[0][0]] = row.Rating
    # set zero when elements smaller that threshold
    threshold=3
    mat_like = (mat > threshold) + 0
    mat_dislike = ((mat > 0) + 0) * ((mat <= threshold)+0)
    '''
    #print3
    #mat of the first four rows and the first six columns
    tmp1 = mat[:4][:6]
    print('原二维数组：用户电影推荐：\n', tmp1)
    #mat_like of the first four rows and the first six columns
    tmp2 = mat_like[:4][:6]
    print('二分网络图,阈值为3\n',tmp2)
    '''
    f_mat = compute_f_mat(mat_like,user_count,movie_count)
    '''
    #print4
    #f_mat of the first four columns
    print('number of users:',user_count.shape[0])
    print('number of movies',movie_count.shape[0])
    print(f_mat.shape)
    tmp3 = f_mat[:4][:]
    print(tmp3)
    '''

    #ROC曲线
    roc_pic(f_mat, user_count, mat_like, mat_dislike)
