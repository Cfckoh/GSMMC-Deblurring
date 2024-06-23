import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy
from PIL import Image
from scipy.linalg import fractional_matrix_power

## convert a 2-d image into a 1-d vector
def vec(image):
    sh = image.shape
    return image.reshape((sh[0]*sh[1]))

## convert a 1-d vector into a 2-d image of the given shape
def im(x, shape):
    return x.reshape(shape)

## display a 1-d vector as a 2-d image
def display_vec(vec, shape, scale = 1):
    image = im(vec, shape)
    plt.imshow(image, vmin=0, vmax=scale * np.max(vec), cmap='gray')
    plt.axis('off')
    plt.show()
    
## write a 1-d vector as a 2-d image file
def save_vec(vec, shape, filename, scale = 1):
    image = im(vec, shape)
    plt.imshow(image, vmin=0, vmax=scale * np.max(vec), cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    
## a helper function for creating the blurring operator
def get_column_sum(spread):
    length = 40
    raw = np.array([np.exp(-(((i-length/2)/spread[0])**2 + ((j-length/2)/spread[1])**2)/2) 
                    for i in range(length) for j in range(length)])
    return np.sum(raw[raw > 0.0001])

## blurs a single pixel at center with a specified Gaussian spread
def P(spread, center, shape):
    image = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            v = np.exp(-(((i-center[0])/spread[0])**2 + ((j-center[1])/spread[1])**2)/2)
            if v < 0.0001:
                continue
            image[i,j] = v
    return image

## matrix multiplication where A operates on a 2-d image producing a new 2-d image
def image_mult(A, image, shape):
    return im( A @ vec(image), shape)

## construct our vector x_true
def build_x_true():
    dx = 10
    dy = 10
    up_width = 10
    bar_width= 5
    size = 64

    h_im = np.zeros((size, size))
    for i in range(size):
        if i < dy or i > size-dy:
            continue
        for j in range(size):
            if j < dx or j > size - dx:
                continue
            if j < dx + up_width or j > size - dx - up_width:
                h_im[i, j] = 1
            if abs(i - size/2) < bar_width:
                h_im[i, j] = 1

    x_exact = vec(h_im)
    return x_exact

## construct our blurring matrix with a Gaussian spread and zero boundary conditions
def build_A(spread, shape):
    #normalize = get_column_sum(spread)
    m = shape[0]
    n = shape[1]
    A = np.zeros((m*n, m*n))
    count = 0
    for i in range(m):
        for j in range(n):
            column = vec(P(spread, [i, j],  shape))
            A[:, count] = column
            count += 1
    normalize = np.sum(A[:, int(m*n/2 + n/2)])
    A = 1/normalize * A
    return A

# construct regularization matrix of first derivative operator 1D
def FirstDerOperator_1D(n):

    d = np.ones(n-1)
    D = np.diag(d,-1)
    L = np.identity(n)-D
    return L

# construct regularization matrix of first derivative operator 2D
def FirstDerOperator_2D(n):
    L1 = FirstDerOperator_1D(n)
    KP1 = np.kron(np.identity(n), L1)
    KP2 = np.kron(L1, np.identity(n))
    L = np.vstack((KP1, KP2))
    return L

def blur_and_noise(image,A,mag,shape,seed=0):
    """
    given a gray scale image, blur kernel and noise magnitude return the blurred image

    image: 1d vector rep of noise
    A: blur kernel
    mag: noise magnitude
    shape: shape of 2d image
    """
    rng = np.random.default_rng(seed)
    b_true = A@image
    noise = rng.random(shape[0]*shape[1])
    e = mag * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
    return b_true + e


def calculate_omega(image, R, sigma):
    """
    Calculates the adjacency matrix defined as Omega from the paper "Fractional
    graph Laplacian for image reconstruction" by Stefano Aleotti, Alessandro
    Buccini, and Marco Donatelli.

    This is done in the following steps:
    1. Create the empty matrix to store the adjacency matrix Omega.
    2. Get the indices for Omega, where every row and column corresponds to a
       pixel in the image.
    3. Convert the indices from Omega into the indices of the image.
    4. Calculate the spatial distance between pixels using the image indices.
    5. Create a mask to select the pixels that are within the radius R.
    6. Calculate the weighted values for the adjacency matrix Omega for pixels
       that are within radius R of each other.
    """
    number_of_pixels = np.prod(image.shape)
    Omega = np.zeros((number_of_pixels, number_of_pixels))
    pixel_index_i, pixel_index_j = np.indices(Omega.shape)
    image_index_i = np.vstack(np.unravel_index(pixel_index_i.flatten(), image.shape))
    image_index_j = np.vstack(np.unravel_index(pixel_index_j.flatten(), image.shape))
    distance = np.linalg.norm(image_index_i - image_index_j, ord = np.inf, axis = 0)
    mask = np.logical_and(0 < distance, distance <= R)
    pixels_i = image[image_index_i[0, mask], image_index_i[1, mask]]
    pixels_j = image[image_index_j[0, mask], image_index_j[1, mask]]
    Omega[mask.reshape(Omega.shape)] = np.exp(-np.square(pixels_i - pixels_j) / sigma)
    return Omega


def generate_graph_laplacian(x_hat,R,sigma,shape):
    """
    Generates L from a given guess x_hat a shape, a theshold value R, and shape
    """
    omega = calculate_omega(im(x_hat, shape),R,sigma)
    D = np.diag(np.sum(omega,axis=1))
    L = (D - omega)/np.linalg.norm(omega) # this is the frobenius norm
    return L

def load_normalize_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpeg',".png",".JPEG",".jpg")):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img)/255
            images.append(img_array)
    return images

def load_normalize_image(image_path):
    """
    Loads an image into vector format and returns the normalized image and the shape
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    return vec(np.array(img)/255), np.array(img).shape

def deblur_2norm(b,L,A,lamb):
    """
    Solves the l2 regularization problem for a given image b with matrix L and given lambda

    PARAMS
        b: blurred image as a vector
        L: L in regularizer
        A: blur matrix 
        lamb: lambda coeffcient of the regularizer
    RETURNS
        deblurred image as a vector
    """
    mat = np.vstack((A,lamb*L))
    vec = np.pad(b,(0, mat.shape[0]-b.size))    
    xTik = scipy.sparse.linalg.lsmr(mat,vec)[0]

    
    
    return xTik

def deblur_2norm_lambda_iteration(b,x_true,L,A):
    """
    Solves the l2 regularization problem for a given image b with matrix L and given lambda

    PARAMS
        b: blurred image as a vector
        x_true: unblurred image for optimization of lambda
        L: L in regularizer
        A: blur matrix 
        lamb: lambda coeffcient of the regularizer
    RETURNS
        best deblurred image as a vector
        best lambda value

    """
    lamb = 1/16
    min_error = np.inf
    best_lamb=lamb
    

    while lamb > 1e-04:
        xTik = deblur_2norm(b,L,A,lamb)
        error = np.linalg.norm(xTik - x_true)/np.linalg.norm(x_true)
        if error < min_error:
            min_xTik = xTik
            min_error = error
            best_lamb = lamb
        lamb /=2
        

    return min_xTik, best_lamb



def deblur_1norm(b,L,A,lamb,iter_limit=10):
    """
    Solves the l1 regularization problem for a given image b with matrix L and given lambda

    PARAMS
        b: blurred image as a vector
        L: L in regularizer
        A: blur matrix 
        lamb: lambda coeffcient of the regularizer
    RETURNS
        deblurred image as a vector
    """
    p=1 # not sure what this is but used in the iterative solver
    x_tick = A.T @ b
    for _ in range(iter_limit):
        x_old = x_tick
        w =np.array(abs(L.dot(x_old)))    
        w[w < 0.02] = 0.001  
        W = np.diag(w)
        W = fractional_matrix_power(W, (p-2)/2)
        mat = np.vstack((A,lamb*W@L))
        vec = np.pad(b,(0, mat.shape[0]-b.size))    
        x_tick = scipy.sparse.linalg.lsmr(mat,vec)[0]


        #x_tick = np.linalg.inv( A.T.dot(A) + lamb**2*L.T.dot(W).dot(L)) @ A.T.dot(b) 

    return x_tick



def deblur_1norm_lambda_iteration(b,x_true,L,A):
    """
    Solves the l2 regularization problem for a given image b with matrix L and given lambda

    PARAMS
        b: blurred image as a vector
        x_true: unblurred image for optimization of lambda
        L: L in regularizer
        A: blur matrix 
        lamb: lambda coeffcient of the regularizer
    RETURNS
        best deblurred image as a vector
        best lambda value

    """
    lamb = 1/16
    min_error = np.inf
    best_lamb=lamb
    

    while lamb > 1e-04:
        xTik = deblur_1norm(b,L,A,lamb)
        error = np.linalg.norm(xTik - x_true)/np.linalg.norm(x_true)
        if error < min_error:
            min_xTik = xTik
            min_error = error
            best_lamb = lamb
        lamb /=2
        

    return min_xTik, best_lamb