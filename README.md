# Détection de langue par Deep Learning

## Contexte académique

Ce projet a été réalisé dans le cadre de l'**UE Apprentissage Profond** du
**Master 2 Sciences Cognitives — parcours Intelligence Artificielle**.

L'objectif de ce projet était de mettre en œuvre une **méthode de deep learning
vue en cours** sur un problème concret. Nous avons choisi d'appliquer les
**réseaux de neurones récurrents (RNN)** à la tâche de **détection automatique de
la langue** d'un texte saisi par l'utilisateur.

## Objectif

À partir d'un texte fourni, le modèle prédit la **langue** dans laquelle il est
écrit, parmi 22 langues possibles.

## Données

Le fichier [`dataset.csv`](dataset.csv) contient **22 000 textes** répartis en
**22 langues** (1 000 échantillons par langue). Il comporte deux colonnes :

- **Text** : le texte dans une langue donnée ;
- **language** : la langue correspondante (label à prédire).

## Méthode (Deep Learning)

L'approche repose sur un **réseau de neurones récurrent** implémenté avec
**PyTorch**. Le pipeline est le suivant :

1. **Tokenisation au niveau des caractères** — adaptée à la détection de langue,
   car chaque langue utilise des alphabets et caractères différents.
2. **Construction d'un vocabulaire** de caractères et encodage en indices entiers.
3. **Couche d'embedding** transformant chaque caractère en un vecteur dense.
4. **Couche récurrente** (RNN / LSTM / GRU, avec option bidirectionnelle) pour
   modéliser la séquence de caractères.
5. **Couche de classification** produisant une probabilité pour chacune des
   22 langues.

Le modèle est entraîné par **descente de gradient** avec une fonction de perte
d'**entropie croisée** (`CrossEntropyLoss`), puis évalué en **précision
(accuracy)** sur un jeu de validation.

## Structure du projet

```
Détection_Langue/
├── dataset.csv                    # Jeu de données (22 000 textes, 22 langues)
├── PROJET-DETECTION-LANGUE.ipynb  # Notebook principal du projet
└── README.md
```

## Technologies

- Python
- PyTorch
- pandas / NumPy
- scikit-learn (encodage des labels)
- Matplotlib (visualisation des résultats)

## Utilisation

1. Ouvrir le notebook [`PROJET-DETECTION-LANGUE.ipynb`](PROJET-DETECTION-LANGUE.ipynb).
2. Exécuter les cellules dans l'ordre pour :
   - charger et prétraiter les données,
   - construire et entraîner le modèle,
   - évaluer ses performances et tester la prédiction sur de nouveaux textes.
