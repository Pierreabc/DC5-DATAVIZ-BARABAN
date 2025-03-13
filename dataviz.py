import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
file_path = "dataset_marketing_dataviz.csv"  # Remplace par ton chemin si nécessaire
df = pd.read_csv(file_path)

# Nettoyage des données
## Convertir la colonne 'Date' en format datetime
df["Date"] = pd.to_datetime(df["Date"])

## Supprimer la colonne inutile si elle ne sert pas
df.drop(columns=["Inutile"], inplace=True, errors='ignore')

## Remplacement des valeurs manquantes par la médiane
df.fillna(df.median(numeric_only=True), inplace=True)

## Suppression des valeurs négatives
df = df[(df["Clics"] >= 0) & (df["Conversions"] >= 0) & (df["Coût"] >= 0)]

## Limitation des valeurs extrêmes pour les conversions
df["Conversions"] = df["Conversions"].clip(upper=df["Conversions"].quantile(0.99))

# Configuration du style des graphiques
sns.set_theme(style="whitegrid")

# Activer le mode interactif pour afficher tous les graphiques
plt.ion()

# 1. Histogramme des impressions par campagne (simplifié)
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Impressions", bins=15, color="blue", alpha=0.7)
plt.title("Impressions par campagne")
plt.xlabel("Impressions")
plt.ylabel("Fréquence")
plt.show()

# 2. Évolution des clics au fil du temps (simplifié)
plt.figure(figsize=(8, 4))
df_resampled = df.set_index("Date").resample("D").sum().reset_index()
sns.lineplot(data=df_resampled, x="Date", y="Clics", color="blue")
plt.title("Clics au fil du temps")
plt.xlabel("Date")
plt.ylabel("Clics")
plt.xticks(rotation=45)
plt.show()

# 3. Scatterplot Clics vs Conversions (simplifié)
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Clics", y="Conversions", alpha=0.5, color="blue")
plt.title("Clics vs Conversions")
plt.xlabel("Clics")
plt.ylabel("Conversions")
plt.show()

# 4. Heatmap des corrélations (simplifié)
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Corrélations")
plt.show()

# Désactiver le mode interactif et afficher tous les graphiques
plt.ioff()
plt.show()
