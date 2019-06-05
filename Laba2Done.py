from PIL import Image, ImageDraw
from scipy.stats import skew, kurtosis
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sys, scipy
from itertools import zip_longest
import statistics as stt
import matplotlib.pyplot as plt
import glob
import fitter
import pickle

COLOR = {'Red': 0,
         'Green': 1,
         'Blue': 2}
DIRNAME="D:/10semester/Progonov/Лабораторные работы/mirflickr/"

Mean_Vector_R = []
Mean_Vector_G = []
Mean_Vector_B = []

Var_Vector_R = []
Var_Vector_G = []
Var_Vector_B = []

Skew_Vector_R = []
Skew_Vector_G = []
Skew_Vector_B = []

Kurt_Vector_R = []
Kurt_Vector_G = []
Kurt_Vector_B = []

# Main Function solve problem

def Solve():
    g = open('OutPut2.txt', "w")
    pp = PdfPages("AllHistogram.pdf")
    np.seterr(divide='ignore', invalid='ignore')
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")  # ignore some warnings from system
    g.write('Task 1----------------------------------------------\n')
    cnt = 0  # count how many images had taken
    for filename in glob.glob('D:/10semester/Progonov/Лабораторные работы/mirflickr/*.jpg') :
        photo = np.array(Image.open(filename))
        g.write('Output for Image number {}\n\n'.format(cnt + 1))
        for name, num in COLOR.items():
            a = photo[:, :, num].ravel()
            Mean=np.mean(a)
            Var=np.var(a)
            Skew=skew(a)
            Kurt=kurtosis(a)
            g.write('Mean value  of {} channel is : {}\n'.format(name,Mean))
            g.write('The Variance of {} channel is  : {}\n'.format(name,Var))
            g.write('Skewness and Kurtosis of {} channel are : {} {}\n\n'.format(name,Skew, Kurt))
            if num==0:
                Mean_Vector_R.append(round(Mean,3))
                Var_Vector_R.append(round(Var,3))
                Skew_Vector_R.append(round(Skew,3))
                Kurt_Vector_R.append(round(Kurt,3))
            elif num ==1:
                Mean_Vector_G.append(round(Mean, 3))
                Var_Vector_G.append(round(Var, 3))
                Skew_Vector_G.append(round(Skew, 3))
                Kurt_Vector_G.append(round(Kurt, 3))
            else:
                Mean_Vector_B.append(round(Mean, 3))
                Var_Vector_B.append(round(Var,3))
                Skew_Vector_B.append(round(Skew, 3))
                Kurt_Vector_B.append(round(Kurt, 3))
        cnt += 1
        if cnt >= 100:
            break

    MATRIX_MEAN_ARRAY = np.array((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B))
    MATRIX_VAR_ARRAY = np.array(
        (Mean_Vector_R, Mean_Vector_G, Mean_Vector_B, Var_Vector_R, Var_Vector_G, Var_Vector_B))
    MATRIX_SKEW_ARRAY = np.array((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B, Var_Vector_R, Var_Vector_G, Var_Vector_B, Skew_Vector_R, Skew_Vector_G, Skew_Vector_B))
    MATRIX_KURT_ARRAY = np.array((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B, Var_Vector_R, Var_Vector_G, Var_Vector_B, Skew_Vector_R, Skew_Vector_G, Skew_Vector_B, Kurt_Vector_R,
                                       Kurt_Vector_G, Kurt_Vector_B))




    g.write("Size of Matrix Mean Array : {}\n".format((MATRIX_MEAN_ARRAY.shape)))
    g.write("Matrix Mean Array:\n"+str(MATRIX_MEAN_ARRAY)+"\n\n")
    Matrix_Mean_cov=np.cov(np.vstack((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B)))
    g.write("Size of Matrix Covariance Mean Array: {}\n".format(Matrix_Mean_cov.shape))
    g.write('Matrix Covariance Mean Array:\n'+str(Matrix_Mean_cov)+"\n\n")

    g.write("Size of Matrix Mean Array and Variance : {}\n".format((MATRIX_VAR_ARRAY.shape)))
    g.write("Matrix Mean Array and Variance:\n"+str(MATRIX_VAR_ARRAY)+"\n\n")
    Matrix_Var_cov = np.array(np.cov(np.vstack((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B, Var_Vector_R, Var_Vector_G, Var_Vector_B))))
    g.write("Size of Matrix Covariance Mean Array and Variance: {}\n".format(Matrix_Var_cov.shape))
    g.write('Matrix Covariance Mean Array and Variance:\n'+str(Matrix_Var_cov)+'\n\n')

    g.write("Size of Matrix Mean Array, Variance and Skewness : {}\n".format((MATRIX_SKEW_ARRAY.shape)))
    g.write('Matrix Mean Array, Variance and Skewness:\n'+str(MATRIX_SKEW_ARRAY)+'\n\n')
    Matrix_Skew_cov = np.array(np.cov(np.vstack((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B, Var_Vector_R, Var_Vector_G, Var_Vector_B, Skew_Vector_R, Skew_Vector_G, Skew_Vector_B))))
    g.write("Size of Matrix Covariance Mean Array, Variance and Skewness: {}\n".format(Matrix_Skew_cov.shape))
    g.write('Matrix Covariance Mean Array, Variance and Skewness:\n'+str(Matrix_Skew_cov)+'\n\n')

    g.write("Size of Matrix Mean Array, Variance , Skewness and Kurtosis : {}\n".format((MATRIX_KURT_ARRAY.shape)))
    g.write('Matrix Mean Array, Variance , Skewness and Kurtosis:\n' + str(MATRIX_KURT_ARRAY) + '\n\n')
    Matrix_Kurt_cov = np.cov(np.vstack((Mean_Vector_R, Mean_Vector_G, Mean_Vector_B, Var_Vector_R, Var_Vector_G, Var_Vector_B, Skew_Vector_R, Skew_Vector_G, Skew_Vector_B, Kurt_Vector_R, Kurt_Vector_G, Kurt_Vector_B)))
    g.write("Size of Matrix Covariance Mean Array, Variance , Skewness and Kurtosis: {}\n".format(Matrix_Kurt_cov.shape))
    g.write('Matrix Covariance Mean Array, Variance , Skewness and Kurtosis:\n' + str(Matrix_Kurt_cov) + '\n\n')
###################################################################################
    g.write('Task 2--------------------------------------------\n')

    image = Image.open("D:/10semester/Progonov/Лабораторные работы/mirflickr/im16.jpg")
    width = image.size[0]
    height = image.size[1]
    g.write("Size of image: {}x{}\n".format(width,height))
    draw = ImageDraw.Draw(image)
    pix = image.load()
    R_new = np.zeros((width, height))   #create an empty list with the same size as image
    for i in range(width):
        for j in range(height):
            R_new[i][j] = pix[i,j][0]
    g.write('matrix R_new \n'+str(R_new)+'\n')

    U, S, Vt = np.linalg.svd(R_new)
    # restored_matrix = np.dot(U, S, Vt)

    # number=100
    error_count=[]

    for number in range(min(width,height)):
        S_add = np.zeros((width, height))  # Create an empty matrix for singular numbers
        for i in range(width):
            for j in range(height):
                if i == j:
                    S_add[i][j] = S[i]
        for i in range(number, width):
            for j in range(number, height):
                S_add[i][j] = 0
        US = np.dot(U, S_add)
        restored_matrix = np.dot(US, Vt)  # dot of 2 array
        g.write("Restored matrix :\n{}\n".format(str(restored_matrix)))
        if number==250:          #how many components we take
            #Restore the image
            for i in range(width):
                for j in range(height):
                    a = int(restored_matrix[i][j])
                    b = pix[i, j][1]
                    c = pix[i, j][2]
                    draw.point((i, j), (a, b, c))

            image.save("RestoredIm16.jpg", "JPEG")
            del draw

        total_error = 0
        for i in range(width):
            for j in range(height):
                diff = abs(R_new[i][j] - restored_matrix[i][j])
                total_error += diff
        EPS = total_error / (width * height) * 100
        g.write('The error of the original matrix and the result: {}%\n'.format(EPS))
        error_count.append(EPS)
    x_number_list=list(range(0,min(width,height)))
    plt.plot(x_number_list,error_count,linewidth=2, grid=True)
    plt.title('Error count depend on quantity of components ')
    plt.xlabel('Components')
    plt.ylabel('Error, %')
    plt.grid()
    pp.savefig()
    pp.close()

###################################################################################
    # Task 3
    g.write('Task 3-------------------------------------------------------\n')
    def transform_matrix_LR(trans):  #tranform matrix to stochastic matrix from left to right
        for i in range(len(trans)):
            for j in range(len(trans[i])-1):
                k=trans[i][j]
                l=trans[i][j+1]
                Stochastic_matrix[int(k)][int(l)]+=1
        for i in range(256):
            sum_row = 0
            for j in range(256):
                sum_row += Stochastic_matrix[i][j]
            Stochastic_matrix[i] /= sum_row
    def transform_matrix_RL(trans):  #tranform matrix to stochastic matrix from right to left
        for i in range(len(trans)):
            temp=len(trans[i])
            for j in range(temp-1):
                k=trans[i][temp-j-2]
                l=trans[i][temp-j-1]
                Stochastic_matrix[int(k)][int(l)]+=1
        for i in range(256):
            sum_row = 0
            for j in range(256):
                sum_row += Stochastic_matrix[i][j]
            Stochastic_matrix[i] /= sum_row

    # Stochastic matrix from the left to right
    Stochastic_matrix = np.zeros((256, 256))
    sum1 = 0
    g.write('Stochastic matrix L -> R: \n')
    transform_matrix_LR(R_new)

    g.write(str(Stochastic_matrix)+'\n')
    Stochastic_matrix=np.linalg.matrix_power(Stochastic_matrix,5)
    g.write('Stochastic matrix after 5 intergration:\n{}\n'.format(str(Stochastic_matrix)))

    # Sum of items divided by the number of items
    for i in range(256):
        for j in range(256):
            sum1 += Stochastic_matrix[i][j]
    g.write('Sum of elements divided by the number of lines = {}\n'.format(sum1/256))

    # Stochastic matrix from the right to left
    Stochastic_matrix = np.zeros((256, 256))
    sum1 = 0
    g.write('Stochastic matrix R -> L: \n')
    transform_matrix_RL(R_new)
    g.write(str(Stochastic_matrix)+'\n')
    #Sum of items divided by the number of items
    for i in range(256):
        for j in range(256):
            sum1 += Stochastic_matrix[i][j]
    g.write('Sum of elements divided by the number of lines = {}\n'.format(sum1 / 256))

    g.close()

Solve()
