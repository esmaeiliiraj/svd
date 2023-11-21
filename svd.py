from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16,8]

A = imread('cat.jpg')
X = np.mean(A, -1)

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# SVD. full_matrices equal to False means "economy SVD"
U, S, VT = np.linalg.svd(X, full_matrices=False)
j = 0
for r in (5, 20, 100):
    # Construct approximate images
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    Xapp = np.concatenate([Xapprox, X], axis=1)
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapp)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r= ' + str(r))
    plt.show()

X.shape
# Given that the size of original image is (1280, 960), we see that with 100 columns instead of 960 we get an
# acceptable result while reducing the size considerably!

# Plotting Singular values
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cululative sum')
plt.show()
