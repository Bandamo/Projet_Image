# InPainting

## Introduction

Ce proramme est une implémentation de la méthode InPainting. Il permet de reconstruire une image à partir d'une image masquée. Pour cela, il utilise la méthode de l'InPainting qui consiste à reconstruire l'image en utilisant les pixels autour du masque.

## Installation

Pour installer le programme, il suffit de cloner le repository et d'installer les dépendances.

```bash
pip3 install matplotlib numpy progressbar2 scipy skimage python-opencv
```

## Utilisation

L'entièreté du programme réside dans la fonction main de la classe Main.

### Lancement par ligne de commande

Pour lancer le programme, il suffit de lancer le fichier main.py avec les arguments suivants :

```bash
python3 main.py <image.jpg> <mask.ppm>
```

### Lancement direct par la fonction main

Pour lancer le programme directement depuis un autre fichier python, il suffit d'importer la classe Main et de créer une instance de main et de lancer la fonction main.
    
```python
from main import Main

m = Main()
main.main(<args>)
```

#### Liste des arguments

| Argument | Description | Exemple | Valeurs par défaut |
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

