# Project Réseau de Neurones : Évaluation de la Contribution des Attributs

## Description

Ce projet a pour objectif principal d'explorer et de rendre **interprétable le comportement local** d'un réseau de neurones artificiels (RNA). Les RNA, bien que puissants, sont souvent considérés comme des "boîtes noires" en raison de leur complexité. Ce travail met en œuvre une approche qui permet de comprendre **pourquoi le modèle prend une décision spécifique pour une instance donnée**, en évaluant la contribution de chaque attribut d'entrée à cette prédiction.

Le projet met en œuvre :
* Un réseau de neurones artificiels pour une tâche de classification sur un jeu de données étendu des fleurs d'Iris.

* Une méthodologie pour sélectionner des instances cibles (correctement et incorrectement classées).

* La génération d'instances perturbées autour de ces cibles.

* L'entraînement de modèles linéaires interprétables (via régression softmax) sur ces données perturbées pour approximer localement le comportement du RNA.

* Des analyses détaillées des performances du RNA (exactitude, matrice de confusion, suivi de l'erreur).

* Des visualisations des contributions des attributs .

Pour une description complète du sujet et des objectifs, veuillez consulter le fichier `Rapport final IA.pdf`.

## Technologies Utilisées

Ce projet est développé en Python et utilise les bibliothèques principales suivantes :

* **Python 3.10+**
* **NumPy** : Pour les opérations numériques et les calculs matriciels.
* **Pandas** : Pour la manipulation et l'analyse des données (chargement de `iris_extended.csv`).
* **Matplotlib** : Pour la génération de toutes les visualisations et graphiques inclus dans le rapport.
* **scikit-learn** : Utilisé pour la division des ensembles de données (`train_test_split`) et le calcul de métriques (matrice de confusion, exactitude).


## Installation

Pour exécuter le projet, suivez les étapes ci-dessous :

1.  **Cloner le dépôt Git :**
    ```bash
    git clone [https://github.com/aivilo-adamah/r-seau-de-neurones-artificiels.git](https://github.com/aivilo-adamah/r-seau-de-neurones-artificiels.git)
    cd project-reseau-de-neurones
    ```

2.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    source venv/bin/activate # Sur Linux/macOS
    # venv\Scripts\activate # Sur Windows
    ```

3.  **Installer les dépendances :**
    Créez un fichier `requirements.txt` à la racine du projet.
    ```
    Ensuite, installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

Pour exécuter l'analyse complète du projet, incluant l'entraînement du réseau de neurones, la génération des perturbations, l'entraînement des modèles locaux et la visualisation des résultats :

1.  Assurez-vous d'avoir suivi les étapes d'installation.
2.  Exécutez le script principal :
    ```bash
    python main.py
    ```
    Ce script affichera séquentiellement les différents graphiques d'évaluation et d'interprétabilité décrits dans le rapport.

## Structure du Projet

Le dépôt est organisé comme suit :

* `data/` : Contient le jeu de données `iris_extended.csv` utilisé pour l'entraînement et l'évaluation du modèle.
* `Local_model.py` : Définit la classe `LinearLocalModel`, l'implémentation de notre modèle linéaire interprétable pour l'explicabilité locale.
* `NeuralNet.py` : Contient la classe `NeuralNet`, notre implémentation du réseau de neurones artificiels.
* `Rapport final IA.pdf` : Le document de rapport final du projet, présentant les méthodologies et les résultats obtenus.
* `3.ContributionAttributs.pdf` : Le document original détaillant l'énoncé du projet.
* `Utility.py` : Ce module centralise diverses fonctions mathématiques essentielles au projet, notamment des fonctions d'activation (`tanh`, `sigmoid`, `relu`, `softmax`) et des fonctions de coût (`cross_entropy_cost`, `MSE_cost`). Il supporte ainsi les calculs internes du réseau de neurones et potentiellement du modèle local.
* `main.py` : Le point d'entrée principal du projet, contient toutes les étapes de l'analyse (chargement des données, entraînement du RNA, sélection des instances, génération des perturbations, entraînement des modèles locaux et affichage des graphiques).


