#!/usr/bin/env python3.5

from PIL import Image
import PIL.ImageOps, pickle, random, os.path
import numpy as np


def creation_entrainement(n):
	#tableau de contenant les données de n images
	#n<=60000
	f = open('train-images.idx3-ubyte', 'rb')
	f.seek(16) #Offset de 16 -> début des images
	g = open('train-labels.idx1-ubyte', 'rb')
	g.seek(8)  #8 -> début des labels
	trainset = []

	for k in range(n):
		imgbytes = f.read(784)
		imgint = []
		for a in imgbytes:
			imgint.append(a)

		labelbytes = g.read(1)
		trainset.append( (np.asarray(imgint),int.from_bytes(labelbytes,byteorder='big')) ) #np.asarray transforme les listes en tableaux numpy
	return trainset


def creation_images(trainset):
	#crée les images contenues dans le tableau dans le dossier courant
	img1 = Image.new('L',(28,28))
	k = 0
	for couple in trainset:
		pixlist = couple[0]
		img1.putdata(pixlist)
		img1 = PIL.ImageOps.invert(img1)
		img1.save("test"+str(k)+".png")
		k+=1

def f(x):
	return 1.71519*np.tanh((2/3)*x)

def deriv_f(x):
	#print(x)
	return 1.16793/(np.cosh((2*x)/3)**2)



def poids_aleat(n,p):
	#crée une matrice de poids initialisés au hasard de taille n,p
	poids = np.random.normal(0,(n/2)**(1/2),(n,p))
	return poids

def creation_reseau(taille_couche_cachee):
	#réseau = liste qui contient les 2 matrices des poids
	poids0 = poids_aleat(784,taille_couche_cachee)
	poids1 = poids_aleat(taille_couche_cachee,10)
	return [poids0,poids1]

def logic(reseau,x0):
	x0 = x0[np.newaxis,:]
	x1 = f(np.dot(x0,reseau[0]))
	x2 = f(np.dot(x1,reseau[1]))
	return(x2)

def prop(x0,reseau,resultat_voulu,alpha):
	x0 = x0[np.newaxis,:]
	x1 = f(np.dot(x0,reseau[0]))
	x2 = f(np.dot(x1,reseau[1]))
	d = [-1 for k in range(10)]
	d[resultat_voulu] = 1
	d = np.asarray(d)
	d = d[np.newaxis,:]


	np.dot(x1,reseau[1])

	err_x2 = x2-d
	print(np.mean(err_x2))
	delta_1 = err_x2*deriv_f(np.dot(x1,reseau[1]))

	grad_err_1 = np.dot(x1.T,delta_1)
	delta_0 = np.dot(delta_1,reseau[1].T) * deriv_f(np.dot(x0,reseau[0]))
	grad_err_0 = np.dot(x0.T,delta_0)

	reseau[1] = reseau[1] - alpha*grad_err_1
	reseau[0] = reseau[0] - alpha*grad_err_0
	return reseau,err_x2
#si trainset n'existe pas, on le crée
if not os.path.isfile('trainset'):
	print("trainset n'exite pas, création...")
	h = open('trainset','wb')
	pickle.dump(creation_entrainement(60000),h)
	h.close()
	print('trainset a été créé')

if __name__ == '__main__':
	h = open('trainset','rb')
	trainset = pickle.load(h)
	random.shuffle(trainset)
	print('lol')



# reseau=creation_reseau(300)
# trainset = creation_entrainement(1)
# print(trainset[0][1])
# np.random.seed(1)
# for i in range(10000):
# 	for k in range(len(trainset)):
# 		reseau = prop(trainset[k][0], reseau, trainset[k][1], 1)[0]
# 	if i%1000 == 0:
# 		print(i)

# print(logic(reseau,trainset[0][0]))



