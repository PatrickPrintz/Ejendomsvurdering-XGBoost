import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import seaborn as sns
#import shap
from shapely.geometry import Point
from matplotlib.colors import ListedColormap
#from sklearn.impute import KNNImputer
#from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from matplotlib import gridspec
from scipy import stats
from scipy.stats import zscore


#####################################################################
##############Data fra Data_construction!############################
#####################################################################

# Koden markeret med kommentar er grunddata hentet fra Data_construction.py og efterfulgt af cleaning
# Vil datasættet hentes på ny, kør Data_construction.py, og fjerne efterfølgende disse kommentarer.
# Vær dog opmærksom på, at data ikke vil kunne rekonstrueres fuldstændigt, hvis boliga har ændret deres hjemmeside.

#####################################################################
#  ## Data efter konstruktion og initial cleaning. Vil kunne opnås ved at køre DataConstruction
# df = pd.read_csv("Data_After_construction_and_initial_cleaning.csv") # Her skal der ændres til den fil, der er genereret i Data_construction.py
# df.shape
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# anvendelse_map = {"Fritliggende enfamiliehus" : "Enfamiliehus", 
#                   "Række-, kæde- eller dobbelthus" : "Rækkehus",
#                    "etageejendom, flerfamiliehus eller to-familiehus" : "Ejerlejlighed"}
# tag_mapping = {
#     1: 'Tagpap', 10: 'Fibercement',11: 'Plastmaterialer',12: 'Glas',
#     2: 'Tagpap hældning',20: 'Levende',3: 'Fibercement asbest',4: 'Betontagsten',
#     5: 'Tegl',6: 'Metal',7: 'Stråtag',80: np.nan,90: 'Andet materiale'
# }
# vaeg_mapping = {
#     1: 'Mursten',10: 'Fibercement uden asbest',11: 'Plastmaterialer',12: 'Glas',2: 'Letbetonsten',3: 'Fibercement herunder asbest',
#     4: 'Bindingsværk',5: 'Træ',6: 'Betonelementer',8: 'Metal',80: np.nan,90: 'Andet materiale'
# }
# df.rename(columns={'Varmeinstallation_tekst': 'varmesinstallation'}, inplace=True)
# varmeinstal_map = {
#     'Centralvarme (1)': 'Centralvarme',
#     'Centralvarme (2)': 'Centralvarme'
# }
# energylabel_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}


# # Replace values in the 'varmesinstallation' column
# df['varmesinstallation'] = df['varmesinstallation'].replace(varmeinstal_map)
# df['anvendelse'] = df['anvendelse'].map(anvendelse_map)
# df['tagkode'] = df['tagkode'].map(tag_mapping)
# df['vægmateriale'] = df['vægmateriale'].map(vaeg_mapping)
# df['Energylabel'] = df['Energylabel'].map(energylabel_map)

# df = df.rename(columns={'Zone': 'Byzone'})
# df['Byzone'] = df['Byzone'].map({'Byzone': 1, 'Landzone': 0})


# Load Consumer Price Index (CPI) data with date and CPI values
# cpi_data = pd.read_excel("/Users/patrickprintz/Desktop/Universitet/Projekter/9. Semester projekt/Kodning og dataset/Soup_test/Regioner_data/prisindeks.xlsx")
# index_2022 = cpi_data.loc[cpi_data['year'] == 2022, 'cpi']
# cpi_data['inflation'] = cpi_data['cpi'].apply(lambda x: ((index_2022 - x) / x)) + 1

# merged_data = pd.merge(df, cpi_data[['year','inflation']], on='year', how='left')
# merged_data['price_2022_1'] = merged_data['price'] * merged_data['inflation_x']

# df = merged_data.copy()



# df = df[(df['discount'] <= 20) & (df['discount'] >= -20)]
# df.describe().T
# df.drop(['price', 'index_old', 'enhed_anv','Opvarmning', 'year', 'inflation_x','address','Energylabel_old','mean_price_index',
#          'Varmeinstallation', 'Storkreds','AntalEtager'], axis=1, inplace=True)

# df.rename(columns={'price_2022': 'price'}, inplace=True)
# df.rename(columns={'Varmeinstallation_tekst': 'varmesinstallation'}, inplace=True)
# df['Energylabel'] = df['Energylabel'].replace('None', np.nan)

# df.isna().sum()
# df = df[df['price'] <= 100000000]


# df['epoch'] = df['date'].apply(lambda x: time.mktime(x.timetuple()))

df = pd.read_csv("Samlet_uden_index.csv")
df.drop('mean_prices_5',axis=1, inplace=True)
df.isnull().sum()
df.dropna(subset=['interest_30_maturity','tagkode','vægmateriale','ombygaar','Energylabel'], inplace=True)

######################################################################
#df.to_csv("XGBoost_data.csv", index=False)
df = pd.read_csv("XGBoost_data.csv")
df['price'] = np.log(df['price'])
z_scores = zscore(df['price'])

# Definerer Z-score grænsen
z_score_threshold = 3

# Bestemmer hvilke rækker der skal fjernes
mask = (z_scores >= -z_score_threshold) & (z_scores <= z_score_threshold)

# Fjerner rækkerne
df = df[mask]
df['price'] = np.exp(df['price'])
df = df.drop(['date', 'price_per_sqm','discount', 'epoch'], axis=1)

df.groupby(['anvendelse'])['Region'].value_counts(normalize=False)

#df.to_csv("XGBoost_data_no_out.csv", index=False) # Til selve modelopbygningen, er det datasæt der er videreført til XGBoost_model_og_shap1.ipynb, der anvendes.


#Plot Denmark boundary
denmark_gdf = gpd.read_file("Shape/DNK_adm0.shp") #Shape fil til Danmark kort

enfamiliehus_gdf = gpd.GeoDataFrame(df[df['anvendelse'] == 'Enfamiliehus'], geometry=df['point'], crs=denmark_gdf.crs)
ejerlejlighed_gdf = gpd.GeoDataFrame(df[df['anvendelse'] == 'Ejerlejlighed'], geometry=df['point'], crs=denmark_gdf.crs)
rækkehus_gdf = gpd.GeoDataFrame(df[df['anvendelse'] == 'Rækkehus'], geometry=df['point'], crs=denmark_gdf.crs)


fig, ax = plt.subplots(figsize=(25 , 10))
denmark_gdf.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
enfamiliehus_gdf.plot(ax=ax, color='darkblue', marker='o', markersize=0.2, label="Enfamiliehus")
#plt.xlabel('Breddegrad', fontsize=16)
plt.ylabel('Længdegrad', fontsize=16)
plt.title('Fordeling af enfamiliehuse i Danmark', fontsize=24)
plt.legend()
ax.set_xlim([5, 18])
ax.set_ylim([54.5, 57.8])
plt.show()

df[['varmesinstallation','tagkode','vægmateriale','Energylabel']]['Energylabel'].value_counts(normalize=True).round(3)*100
## Price distribution
plt.figure()
from matplotlib.ticker import MultipleLocator
fig, ax = plt.subplots(figsize=(12, 4))
sns.histplot(df['price'], bins=100, kde=True, ax=ax)
ax.set_title('Logaritmisk tranformation' , fontsize=18)
ax.set_xlabel('Log(Handelspris)', fontsize=12)
ax.set_ylabel('Antal ejendomme', fontsize=12)
note_text = """Annotering: Log(Handelspris).
Kilde: Egen formidling på baggrund af data"""
fig.text(0.05, 0, note_text, fontsize=12, color='black', ha='left', va='bottom')

ax.xaxis.set_major_locator(MultipleLocator(0.5))
plt.show()

df[df['price'] >= 10000000]['anvendelse'].value_counts()

#### Numerisk
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(df['price'].describe().T)

# Build year distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df['build_year'], bins=100, kde=False)
plt.show()

# Distances distribution numerical
df.filter(like='dist').describe().T.round(3)

## Area distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['area'], bins=100, kde=True)
plt.title('Boligareal')
plt.xlabel('Kvadratmeter m2')
plt.ylabel('Antal ejendomme')
plt.show()




# Korrelationen med pris

subset_df = pd.get_dummies(df.drop(['Sogn','point'], axis = 1, inplace=False), prefix_sep='()' ,drop_first=False)
subset_df.columns = [
    col.split('()')[0].capitalize() + ' (' + col.split('()')[1] + ')' if '()' in col else col
    for col in subset_df.columns
]
subset_df = subset_df[subset_df['Anvendelse (Enfamiliehus)'] == 1]
# Beregning
correlation_with_price = subset_df.corr()['price'].sort_values(ascending=False)[1:]
correlation_with_price = correlation_with_price[(np.abs(correlation_with_price) > 0.01) & (~correlation_with_price.isna())]

reversed_palette = sns.color_palette("coolwarm", n_colors=len(correlation_with_price))
reversed_palette = reversed_palette[::-1]

# Plot
plt.figure()
plt.figure(figsize=(10, 15))
sns.barplot(x=correlation_with_price.values, y=correlation_with_price.index, palette=reversed_palette)
plt.xlabel('Correlation with Price')
plt.ylabel('Variables')
plt.title('Correlation of Variables with Price')
plt.show()