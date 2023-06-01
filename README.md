# InPainting

## Introduction

Ce proramme est une implémentation de la méthode InPainting. Il permet de reconstruire une image à partir d'une image masquée. Pour cela, il utilise la méthode de l'InPainting qui consiste à reconstruire l'image en utilisant les pixels autour du masque.

## Installation

Pour installer le programme, il suffit de cloner le repository et d'installer les dépendances.

```bash
pip3 install matplotlib numpy progressbar2 scipy skimage python-opencv
```

## Utilisation

L'entièreté du programme réside dans la fonction run de la classe InPainting.

### Lancement par ligne de commande (Recommandé)

Pour lancer le programme, il suffit de lancer le fichier main.py avec des arguments :

```bash
python3 main.py -i image.jpg -m mask.ppm
```

Pour plus d'informations sur les arguments, lancer le programme avec l'argument -h.

```bash
python3 main.py -h

Help :
	-i : image path
	-m : mask path
	-p : patch size
	-r : result (save, print, return)
	-v : verbose (True, False)
	-s : save (True, False)
	-plot : plot (True, False)
	-d : distance method (SSD, SSDED, MC)
	-dis : discretisation (float)
	-t : number of thread (int)
	-dyn : dynamic patches (True, False)
	-h : help
Example : python main.py -i image.jpg -m mask.jpg -p 9 -r save -v True -s True -plot False -d SSDED -dis 1 -t 1 -dyn False

```


### Lancement direct par la fonction main

Pour lancer le programme directement depuis un autre fichier python, il suffit d'importer la classe InPainting et de créer une instance de main et de lancer la fonction run.
    
```python
from main import InPainting

m = Main()
main.run(<args>)
```

#### Liste des arguments

| Argument | Description | Exemple | Valeur par défaut |
| --- | --- | --- | --- |
| image_path | Chemin vers l'image à reconstruire| image.jpg | Aucune |
| mask_path | Chemin vers le masque de l'image | mask.ppm | Aucune |
| patch_size | Longueur des côtés du patch carré (doit être impaire afin d'avoir un centre) | 5 | 9 |
| result | Action de fin de la fonction main, "save" sauvegarde l'image dans le dossier log_image, "print" affiche l'image, "return" retourne l'image en sortie de fonction | "print" | "save" |
| verbose | Affiche les informations de progression | True | False |
| save | Sauvegarde les image dans log_image à chaque remplacement de patch | True | False |
| distance_method | Méthode de comparaison entre deux patchs ("SSD" Somme des carrés des différences, "MC" comparaison des coulerus moyennes, "SSDED" Somme des carrés des différences auquel on ajout la distance euclidienne entre les patchs, flottant f entre 0 et 1 mix SSD\*f+MC\*(1-f)) | "MC" | "SSDED" |
| discretisation | Distance entre chacun des patchs de comparaison, plus cette discretisation est grande, plus le programme est rapide mais moins précis | 5 | 1 |
| nb_thread | Nombre de thread utilisé pour le calcul du meilleur patch | 12 | 1 |
| dynamic_patches | Utilise les patchs générés en tant que patchs de remplacement | True | False |

## Image de test

Certaines images sont disponibles dans le dossier image et les masques dans le dossier mask.
À une image x.jpg correspondent les masques x.ppm et x_y.ppm.