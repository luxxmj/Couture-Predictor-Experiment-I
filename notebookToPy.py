# %% [markdown]
# # Import dependancies

# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import en_core_web_lg

# %% [markdown]
# # Prepare data

# %% [markdown]
# ## Import
# Concatanate into one prada products dataframe and clean data as needed
# - Reformatted material proportions for easier embedding

# %%
denim = pd.read_csv("Prada\\pradaProducts_Denim.csv")
knitwear = pd.read_csv("Prada\\pradaProducts_Knitwear.csv")
leatherClothing = pd.read_csv("Prada\\pradaProducts_Leather-Clothing.csv")
outerwear = pd.read_csv("Prada\\pradaProducts_Outerwear.csv")
suits = pd.read_csv("Prada\\pradaProducts_Suits.csv")

pradaDataset = pd.concat(
    [denim, knitwear, leatherClothing, outerwear, suits], ignore_index=True
)
print(pradaDataset.head())
pradaDataset["price"].describe()

# %%
materials = []


def deriveMaterials(data):
    materialDict = {}

    materialLi = ""
    splitData = data.split(" ")

    for i, n in enumerate(splitData):
        if "%" in n:
            perc = str(float(n.strip("%")) / 100)
            materialDict[perc] = ""
        else:
            if i == 0:
                material = n.strip().lower().strip(",")
                if material not in materials:
                    materials.append(material)
                return f"1.0-{material}"
            else:
                material = n.strip().lower().strip(",")
                materialDict[perc] += (
                    f"{' ' if len(materialDict[perc]) > 0 else ''}{material}"
                )

    percs = [z for z in materialDict.keys()]
    percs.sort(key=lambda y: float(y), reverse=True)

    # Lists unique materials
    for ii, p in enumerate(percs):
        mat = materialDict.get(p)
        if mat not in materials:
            materials.append(mat)
        materialLi += f"{'|' if ii > 0 else ''}{p}-{mat}"
    return materialLi


print(pradaDataset["material"].head())
pradaDataset["derivedMaterial"] = pradaDataset["material"].apply(deriveMaterials)
pradaDataset["derivedMaterial"].head()

# %% [markdown]
# ## Feature Engineering
# Manually transform features as needed
# - Created material embeddings
# - Vectorized description & details
# - Label encoded clothing categories

# %%
print(len(materials), materials)

# Embed material feature
for row, value in enumerate(pradaDataset["derivedMaterial"]):
    mats = value.split("|")
    for variant in mats:
        proportion = float(variant.split("-")[0])
        material = variant.split("-")[1]
        pradaDataset.at[row, material] = proportion

# Fill n/a values for non-present material columns in rows
pradaDataset = pradaDataset.fillna(value=0.00)

# %%
# load nlp model and vectorized descriptions and details
cloneDF = pradaDataset.copy()
nlp = en_core_web_lg.load()


def vectorize(text):
    doc = nlp(text)
    return doc.vector_norm


cloneDF["description"] = pradaDataset["description"].apply(vectorize)
cloneDF["details"] = pradaDataset["details"].apply(vectorize)

cloneDF["description"].head(), cloneDF["details"].head()

# %%
label_encoder = LabelEncoder()
cloneDF["type"] = label_encoder.fit_transform(cloneDF["type"])
cloneDF["type"].unique()

# %% [markdown]
# # Review data and experiment

# %% [markdown]
# ## Visualize correlations
# - Materials : Price (feature significance)
# - Description/Detail : Price (feature significance)
# - Materials : Category (Classifiability)
# - Description/Detail : Category (Classifiability)

# %%
for material in materials:
    px.scatter(pradaDataset, x=material, y="price", range_x=[0.2, 1.0]).show()

px.scatter(cloneDF, x="description", y="price").show()
px.scatter(cloneDF, x="details", y="price").show()

px.box(cloneDF, y="description", x="type").show()
px.box(cloneDF, y="details", x="type").show()

for material in materials:
    px.box(
        pradaDataset,
        y=material,
        x="type",
        range_y=[0.0045, 1.05],
        title=f"Amount of {material} used per category",
    ).show()

# %% [markdown]
# ## Split data and prepare models
# - Category (type), description, details, material vector for features
# - 15 Random Forest Regressor models (nestimators and maxdepth variants)

# %%
droppedLabels = [
    "name",
    "id",
    "material",
    "price",
    "colors",
    "sizes",
    "derivedMaterial",
]
targetLabel = ["price"]

X = cloneDF.drop(droppedLabels, axis=1)
Y = cloneDF[targetLabel].to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    train_size=0.2,
    random_state=2124,
)

# %%
estimators = [50, 100, 200, 300, 350]
depths = [None, 1, 2]

models = []
vvv = 0

for est in estimators:
    for depth in depths:
        models.append(
            RandomForestRegressor(n_estimators=est, max_depth=depth, random_state=2124)
        )
        print(f"{depth} depth, {est} n-estimator model queued. Pos. {vvv}")
        vvv += 1

# %% [markdown]
# ## Assess models

# %%
for vv, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model {vv} \nMAE: {mae}\nR2: {r2}\n")
