###
"""
Følgende dokument beskriver hvordan dataet er konstrueret på baggrund af adskillige API løsninger og webscraping.
Der er i det følgende sideløbende ført en gennemgående data-cleaning, da flere af disse processor tager lang tid at kører.
Derfor er det valgt sideløbende at rense ud i dataet, for at minimere antallet af specifikke ejendomme, som køres, for at spare tid.
Dataet er ligeledes blevet kørt i UCLOUD på flere maskiner ad gangen, hvortil de enkelte datasets navne ikke nødvendigvis stemmer overens fra hver trin.
Fremgangsmåden og rækkefølgen stemmer dog ovorens i konstruktionen af dataet.
"""
###




# Importer pakker
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import csv
import re
import time
from tqdm import tqdm
import warnings
import ast
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import httpx
import asyncio
import aiohttp
import seaborn as sns
import matplotlib as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Step 1: Scraper data
# Giver output.csv med boligdata fra boliga
def scrape_page(page):
    response = requests.get(page)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    address = [element.text for element in soup.select('.text-primary.fw-bolder.text-start')]
    information = [element.text for element in soup.select('.table-col.d-print-table-cell.text-center')]
    
    liste = [i for i in information if i != ' Aktuel værdi ']
    
    logo_letters = [element.text for element in soup.select('.icon')]
    logo_letters = logo_letters[4:]
    
    data_list = []
    
    
    if len(address) == len(liste) // 6 and len(liste) % 6 == 0 and len(logo_letters) >= len(address):
        for i in range(len(address)):
            data_dict = {
                'address': address[i],
                'price': liste[i * 6],
                'date_type': liste[i * 6 + 1],
                'squarmeters': liste[i * 6 + 2],
                'rooms': liste[i * 6 + 3],
                'build_year': liste[i * 6 + 4],
                'discount': liste[i * 6 + 5],
                'logo_letter': logo_letters[i],
            }
            data_list.append(data_dict)
    
    return data_list


if __name__ == '__main__':
    base_url = "https://www.boliga.dk/salg/resultater?searchTab=1&sort=date-d&roomsMin=1&sizeMin=50&salesDateMin=2000&saleType=1&propertyType=1,2,3,6&page="
    page_urls = [base_url + str(page) for page in range(1, 10000)]
    
    data_list = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(scrape_page, page_urls), total=len(page_urls), desc="Processing Pages"))

    for result in results:
        data_list.extend(result)

    
    keys = data_list[0].keys()
    with open('output.csv', 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data_list)

df = pd.read_csv('output.csv')
df.drop_duplicates(subset=['address'], inplace=True)
df.shape

df['price'] = df['price'].str.replace('kr.', '')
df['price'] = df['price'].str.replace('.', '')
df['price'] = pd.to_numeric(df['price'].str.strip())

df['date'] = df['date_type'].str.extract(r'(\d{2}-\d{2}-\d{4})')
df['type'] = df['date_type'].str.replace(r'\d{2}-\d{2}-\d{4}', '').str.strip()
df.drop('date_type', axis = 1, inplace = True)

df['area'] = df['squarmeters'].str.extract(r'(\d+\.?\d*) m²')
df['price_per_sqm'] = df['squarmeters'].str.extract(r'(\d+\.?\d*) kr/m²')

df.drop('squarmeters', axis = 1, inplace = True)
df['discount'].fillna(0, inplace =True)

#####################################################################################################

# Step 2: Udvider data med enhed_id, koordinater, sogn, storkreds, region og zone


csv_file_path = 'output_udvidet.csv'
header = ['address', 'ids', 'Coordinates', 'Sogn', 'Storkreds', 'Region', 'Zone', 'BFE Number']


address_list = df['address']  


session = requests.Session()

def process_address(address):
    add = f"https://api.dataforsyningen.dk/adresser?q={address}"
    try:
        with session.get(add) as response:
            response.raise_for_status()
            data = response.json()

            if data:
                koordinater = data[0]['adgangsadresse']['adgangspunkt']['koordinater']
                sogn_navn = data[0]['adgangsadresse']['sogn']['navn']
                storkreds = data[0]['adgangsadresse']['storkreds']['navn']
                region_navn = data[0]['adgangsadresse']['region']['navn']
                zone_value = data[0]['adgangsadresse']['zone']
                ids = data[0].get('id')
                
                dataloaderurl = f"https://services.datafordeler.dk/DAR/DAR_BFE_Public/1/REST/adresseTilEnhedBfe?username=TAKWDMRRWE&password=Patrick!1&adresseId={ids}"
                
                with session.get(dataloaderurl) as data_loader_response:
                    data_loader_response.raise_for_status()
                    BFE_number = data_loader_response.json()

                result = [address, ids, koordinater, sogn_navn, storkreds, region_navn, zone_value, BFE_number]

                return result
            else:
                print(f"Empty data for address: {address}")
                result = [1, 1, 1, 1, 1, 1, 1, 1]
                return result
    except requests.RequestException as e:
        print(f"Error fetching data for address {address}: {e}")
        return None


with concurrent.futures.ThreadPoolExecutor() as executor:
    count = 0
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for result in executor.map(process_address, address_list):
            if result is not None:
                count += 1
                csv_writer.writerow(result)

                if count % 100 == 0:
                    print(f"Processed {count} addresses. Saving to CSV...")


udvidet_output = pd.read_csv('output_udvidet.csv')

data = pd.merge(df, udvidet_output, on='address', how='left')


#####################################################################################################

# Step 3: Udvid yderligerer data med bygnings id, enhed anvendelse, boligtype, antal etager, varmeinstallation, opvarmning og grund id

address_list = data['ids']  

csv_file_path = '/Users/patrickprintz/Desktop/Universitet/Soup_test/output_udvidet_udvidet.csv'
header = ['ids', 'byg_id', 'enhed_anv','boligtype', 'AntalEtager', 'Varmeinstallation', 'Opvarmning', 'grund_id']

session = requests.Session()

def process_address(address):
    add = f"https://api.dataforsyningen.dk/bbrlight/enheder?adresseid={address}"
    try:
        with session.get(add) as response:
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                for i in range(len(data)):
                    data_entry_run = data[i]
                    boligtype = data_entry_run.get('BOLIGTYPE_KODE', None)
                    enhed_anv = data_entry_run.get('ENH_ANVEND_KODE', None)
                    if boligtype in ['1', '2', '4'] and int(enhed_anv) < 149:
                        data_entry = data[i]
                        
                        boligtype = data_entry.get('BOLIGTYPE_KODE', None)
                        enhed_anv = data_entry.get('ENH_ANVEND_KODE', None)
                        
                        bygning = data_entry.get('bygning', None)
                        if bygning:
                            
                            byg_id = bygning.get('Bygning_id', None)
                            AntalEtager = bygning.get('ETAGER_ANT', None)
                            Varmeinstallation = bygning.get('VARMEINSTAL_KODE', None)
                            Opvarmning = bygning.get('OPVARMNING_KODE', None)
                            grund_id = bygning.get("Grund_id", None)
                        else:
                            byg_id = AntalEtager = Varmeinstallation = Opvarmning = None

                        result = [address, byg_id, enhed_anv ,boligtype, AntalEtager, Varmeinstallation, Opvarmning, grund_id]

                        return result
                    else:
                        print(f"Skipping address {address} as not foint in {i}")

                
                print(f"No entry with boligtype equal to 1 for address: {address}")
                return None
            else:
                print(f"Empty or invalid data for address: {address}")
                result = [999, 999, 999, 999, 999, 999, 999, 999]
                return result
    except requests.RequestException as e:
        print(f"Error fetching data for address {address}: {e}")
        return None


with concurrent.futures.ThreadPoolExecutor() as executor:
    count = 0
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for result in executor.map(process_address, address_list):
            if result is not None:
                count += 1
                csv_writer.writerow(result)

                if count % 100 == 0:
                    print(f"Processed {count} addresses. Saving to CSV...")


output_udvidet_udvidet = pd.read_csv('/Users/patrickprintz/Desktop/Universitet/Soup_test/output_udvidet_udvidet.csv')

data_1 = pd.merge(data, output_udvidet_udvidet, on='ids', how='left')

data_1.to_csv("/Users/patrickprintz/Desktop/Universitet/Soup_test/dataset_m_relevant.py", index=False)

#####################################################################################################

# Step 4: Tilføj antal bad og toiletter:

full_df = pd.read_csv("/Users/patrickprintz/Desktop/Universitet/Soup_test/dataset_m_relevant.py")


csv_file_path = '/Users/patrickprintz/Desktop/Universitet/Soup_test/dataset_m_relevant_med_bad.py'
header = ['ids', "AntalBad", "AntalToilet"]

address_list = full_df['ids']  
session = requests.Session()

def process_address(address):
    add = f"https://services.datafordeler.dk/BBR/BBRPublic/1/rest/enhed?format=JSON&username=TAKWDMRRWE&password=Patrick!1&AdresseIdentificerer={address}"
    try:
        with session.get(add) as response:
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                entry = data[0]
                antal_vandskyllede_toiletter = entry.get("enh065AntalVandskylledeToiletter")
                antal_badeværelser = entry.get("enh066AntalBadeværelser")
                result = [address, antal_badeværelser, antal_vandskyllede_toiletter]
            else :
                print(f"Empty or invalid data for address: {address}")
                result = [999, 999, 999]
            
            return result

    except requests.RequestException as e:
        print(f"Error fetching data for address {address}: {e}")
        return None



with concurrent.futures.ThreadPoolExecutor() as executor:
    count = 0
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for result in executor.map(process_address, address_list):
            if result is not None:
                count += 1
                csv_writer.writerow(result)

                if count % 1000 == 0:
                    print(f"Processed {count} addresses. Saving to CSV...")


med_bad = pd.read_csv("/Users/patrickprintz/Desktop/Universitet/Soup_test/Filtreret_data1_med_bad.csv")


#######################################################################################################

# Udvider data med grundstørrelse og matrikelnummer

address_list = full_df['grund_id']
csv_file_path = "samlet_med_grund_str.csv"
header = ["grund_id", "matrikel_nr", "grund_str"]

session = requests.Session()
session_1 = requests.Session()


def process_address(address):
  add = f"https://api.dataforsyningen.dk/bbrlight/grunde?id={address}"
  try:
        with session.get(add) as response:
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                entry = data[0]
                matrikel_nr = entry.get("MatrSFE_id")

                if matrikel_nr:
                  url = f"https://services.datafordeler.dk/MATRIKEL/Matrikel/1/REST/SamletFastEjendom?username=TAKWDMRRWE&password=Patrick!1&Format=JSON&SFEBFEnr={matrikel_nr}"

                    
                  with session_1.get(url) as response_1:
                        response_1.raise_for_status()
                        grund = response_1.json()

                        if grund:
                            grund_areal = grund['features'][0]['properties']['jordstykke'][0]['properties']['registreretAreal']
                            result = [address, matrikel_nr, grund_areal]
                            return result
                            
                        else:
                            print(f"Grund not found for: {address}")
                            result = [address, None, "Error"]
                            return result


            else:
              print(f"ingen matrikelnummer for: {address}")
              result = [address, "Error", None]
              return result

  except requests.RequestException as e:
        print(f"Error fetching data for address {address}: {e}")
        return None


with concurrent.futures.ThreadPoolExecutor() as executor:
    count = 0
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for result in executor.map(process_address, address_list):
            if result is not None:
                count += 1
                csv_writer.writerow(result)

                if count % 100 == 0:
                    print(f"Processed {count} addresses. Saving to CSV...")


##########################################################################################
df = pd.read_csv("samlet_med_grund_str.csv")

df.info()
# Data cleaning
df['discount'] = df['discount'].str.replace('%', '')  
df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)  

df['date'] = df['date_type'].str.extract(r'(\d{2}-\d{2}-\d{4})')
df.drop('date_type', axis = 1, inplace = True)

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')


enheds_anvendelse_mapping = {140: "etageejendom, flerfamiliehus eller to-familiehus", 120: "Fritliggende enfamiliehus",
                             132: "Række-, kæde- eller dobbelthus", 131: "Række-, kæde- eller dobbelthus", 130: "Række-, kæde- eller dobbelthus",
                             110: "Stuehus v. landbrugsejendom", 121: "Sammenbygget enfamiliehus",
                             122: "Fritliggende enfamiliehus i tæt-lav bebyggelse"}

df = df[df['enhed_anv'].isin([140, 120, 132, 131, 130])]
df['anvendelse'] = df['enhed_anv'].map(enheds_anvendelse_mapping)


boligtype_mapping = {1: "Egentlig beboelseslejlighed", 2: "Blandet erhverv og bolig"}
df = df[df['boligtype'].isin([1])]
df.drop('boligtype', axis = 1, inplace = True)


varmeinstal_mapping = {1 : "Fjernvarme/blokvarme", 2: "Centralvarme (1)", 3 : "Ovn", 5: "Varmepumpe", 6: "Centralvarme (2)",
                       7: "Elvarme", 8: "Gasradiator", 9 : "Ingen varmeinstallation", 99: "Blandet"}

df['Varmeinstallation'] = df['Varmeinstallation'].fillna(0).astype(int)
df['Varmeinstallation'].replace(0, None, inplace = True)
df['Varmeinstallation_tekst'] = df['Varmeinstallation'].map(varmeinstal_mapping)

df['AntalEtager'] = df['AntalEtager'].fillna(9999999).astype(int)
df['AntalEtager'].replace(9999999, None, inplace = True)

df.drop(['BFE Number', 'unique_id'], axis=1, inplace=True)

df.loc[df['anvendelse'] == 'etageejendom, flerfamiliehus eller to-familiehus', 'grund_str'] = 0

df['grund_str'] = df['grund_str'].replace('Error', None).fillna(0).astype(float).astype(int)
df.groupby('grund_str')['address'].count().sort_values(ascending = False)


# Filtrering baseret på værdi og sandsynlige fejl
df = df[(df['AntalBad'] >= 1) & (df['AntalBad'] <= 10)]
df.shape
df = df[(df['AntalToilet'] >= 1) & (df['AntalToilet'] <= 10)]
df.shape
df = df[(df['rooms'] >= 1) & (df['rooms'] <= 20)]
df.shape
df = df[df['area'] <= 1000]
df.shape
df = df[(df['price'] >= 100000) & (df['price'] <= 100000000)]
df.shape
df = df[df['build_year'] >= 1200]
df.shape
df = df[df['discount'] <= 100]
df.shape
df = df[df['Zone'].isin(['Byzone', 'Landzone'])]
df.shape



df.drop('Coordinates', axis = 1, inplace = True)


df = df.dropna(subset=df.columns.difference(['Opvarmning']))
df = df.reset_index(drop=False)
df.to_csv("boligdata.csv", index = False)
df.groupby('Region')['address'].count().sort_values(ascending = False)


##########################################################################################
# Step 5: Beregn middelpris for 20 tætteste boliger efter boligtype og region


df = pd.read_csv("samlet_med_grund_str.csv") # Rettet til for sammenhængens skyld

## Note: KØRT I UCLOUD HVOR DER ER OPDELT EFTER REGION PÅ 5 INDIVIDUELLE MASKINER


def distance(lat1, lon1, lat2, lon2):
    R = 6371  # radius of the earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x = (lon2 - lon1) * np.cos(0.5 * (lat2 + lat1))
    y = lat2 - lat1
    d = R * np.sqrt(x*x + y*y)
    return d

def distance_points_optimized(house, df=df):
    specific_point = (house['latitude'], house['longitude'])

    distances = distance(specific_point[0], specific_point[1], df['latitude'].values, df['longitude'].values)
    distances_sorted_indices = np.argsort(distances)[:21]
    distances = distances[distances_sorted_indices][1:] + 1e-6

    
    weights = 1 / distances
    prices = df['price'].values[distances_sorted_indices][1:]

    
    weighted_price = np.sum(prices * weights) / np.sum(weights)

    return weighted_price

df['weighted_price'] = df.progress_apply(distance_points_optimized, axis=1)

df.to_csv('mean_price_data.csv', index=False)

##############################################################################################################################

# Tilføjer energimærkning
boligdata = pd.read_csv("mean_price_data.csv")

BATCH_SIZE = 1000

async def fetch_energy_label(session, url, headers, params):
    async with session.get(url, headers=headers, params=params) as response:
        data = await response.json()
        energy_label = data.get('SearchResults')
        if energy_label:
            for x in energy_label:
                if x.get('LabelStatus') == "VALID":
                    label = x.get('EnergyLabelClassification')
                    return label
        else:
            return 'None'

async def process_batch(session, batch):
    tasks = []
    for _, row in batch.iterrows():
        X = row['latitude']  
        Y = row['longitude']
        url = 'https://emoweb.dk/EMOData/EMOData.svc/GetEnergyLabelFromCoordinatesWithSearchRadius'
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Basic cHByaW50MTlAc3R1ZGVudC5hYXUuZGs6MTIzNDU2Nzg='  
        }
        params = {
            'coordinateX': X,
            'coordinateY': Y,
            'pagesize': 100,
            'pageNumber': 1,
            'searchRadius': 0.00018
        }
        task = asyncio.ensure_future(fetch_energy_label(session, url, headers, params))
        tasks.append(task)
    energy_labels = await asyncio.gather(*tasks)
    return energy_labels

async def process_dataframe(df):
    num_rows = len(df)
    num_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches), desc="Processing batches"):
        async with aiohttp.ClientSession() as session:
            start = i * BATCH_SIZE
            end = min((i + 1) * BATCH_SIZE, num_rows)
            batch = df.iloc[start:end]

            energy_labels = await process_batch(session, batch)
            df.loc[start:end-1, 'Energylabel'] = energy_labels

new_loop = asyncio.new_event_loop()
asyncio.set_event_loop(new_loop)
new_loop.run_until_complete(process_dataframe(boligdata))

##############################################################################################################################
# Step 6: Tilføj derefter renteniveauet
boligdata_energy = pd.read_csv("/Users/patrickprintz/Desktop/Universitet/Projekter/9. Semester projekt/Kodning og dataset/Soup_test/Regioner_data/Boligdata_energy.csv")
boligdata_energy['date'] = pd.to_datetime(boligdata_energy['date'], format='%Y-%m-%d')
interest_rate = pd.read_excel("/Users/patrickprintz/Desktop/Universitet/Projekter/9. Semester projekt/Kodning og dataset/Soup_test/interestrates.xlsx")

boligdata_energy['week'] = boligdata_energy['date'].dt.isocalendar().week
boligdata_energy['year'] = boligdata_energy['date'].dt.isocalendar().year

boligdata = pd.merge(boligdata_energy, interest_rate, how='left', left_on=['year', 'week'], right_on=['year', 'week'])

boligdata = boligdata.drop(columns=['week', 'year'])
#boligdata.to_csv('/Users/patrickprintz/Desktop/Universitet/Projekter/9. Semester projekt/Kodning og dataset/EDA/færdig_data.csv', index=False)

###################################################################################################################################################
# Tilføjer tagmateriale, vægmateriale og ombygningsår
boligdata = pd.read_csv("/Users/patrickprintz/Desktop/Universitet/Projekter/9. Semester projekt/Kodning og dataset/EDA/dist.csv")


csv_file_path = '/Users/patrickprintz/Desktop/Universitet/Projekter/9. Semester projekt/Kodning og dataset/EDA/dist_med_tag.csv'
header = ['address', "tagkode", "vægmateriale", "ombygaar"]

address_list = boligdata['address']

session = requests.Session()
session1 = requests.Session()

def process_address(address):
    add = f"https://api.dataforsyningen.dk/adresser?q={address}"
    try:
        with session.get(add) as response:
            response.raise_for_status()
            data = response.json()
            if data:
                id = data[0].get('id')
                add1 = f"https://api.dataforsyningen.dk/bbrlight/enheder?adresseid={id}"
                try:
                    result = [address, None, None, None]  
                    with session1.get(add1) as response1:
                        response1.raise_for_status()
                        data1 = response1.json()
                        if data1 and isinstance(data1, list) and len(data1) > 0:
                            entry1 = data1[0]
                            bygning = entry1.get('bygning', None)
                            if bygning:
                                tagkode = bygning.get("TAG_KODE", None)
                                vægmateriale = bygning.get("YDERVAEG_KODE", None)
                                ombygaar = bygning.get("OMBYG_AAR", None)
                                result = [address, tagkode, vægmateriale, ombygaar]
                            else:
                                print(f"Ingen bygning for address: {address}")
                                result = [address, None, None, None]
                except requests.RequestException as e:
                    print(f"Session 2 mislykkedes ved {address}: {e}")
                    result = [address, None, None, None]
            else:
                
                result = [address, None, None, None]
            return result

    except requests.RequestException as e:
        print(f"Error fetching data for address {address}: {e}")
        result = [address, None, None, None]
        return result


with concurrent.futures.ThreadPoolExecutor() as executor:
    count = 0
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for result in executor.map(process_address, address_list):
            if result is not None:
                count += 1
                csv_writer.writerow(result)

                if count % 1000 == 0:
                    print(f"Processed {count} addresses. Saving to CSV...")


#### Tilføjer distancer

# Importing shapefiles:
## Denmark (aka. coast)
denmark_gdf = gpd.read_file("/work/coast/hy248pm2476.shp")
denmark_gdf = denmark_gdf.to_crs(epsg=4326)

buffered_geometries = [geometry.buffer(0) for geometry in denmark_gdf.geometry]
denmark_gdf_buffered = gpd.GeoDataFrame(geometry=buffered_geometries, crs=denmark_gdf.crs)

## Highways
highways = gpd.read_file("/work/highways/hotosm_dnk_roads_lines.shp")
highways = highways.to_crs(epsg=4326)
highways = highways[highways['highway']=="motorway"]

## Railroads
railroads = gpd.read_file("/work/railroads/hotosm_dnk_railways_lines.shp")
railroads = railroads.to_crs(epsg=4326)

## Major Airports
airports = gpd.read_file("/work/airports/osm-world-airports.shp")
airports = airports.to_crs(epsg=4326)
airports = airports[airports['country'] =="Denmark"].drop(181, axis=0)

def Distances(house, denmark_buffered, highways_union, railroads_union, airports_union):
    # Specific point
    specific_point = Point(house['longitude'], house['latitude'])

    # Coast
    nearest_boundary_point = nearest_points(specific_point, denmark_buffered.boundary)[1]
    distance_km_coast = geodesic((specific_point.x, specific_point.y), (nearest_boundary_point.x, nearest_boundary_point.y)).kilometers

    # Highways
    nearest_highway_point = nearest_points(specific_point, highways_union)[1]
    distance_km_highway = geodesic((specific_point.x, specific_point.y), (nearest_highway_point.x, nearest_highway_point.y)).kilometers

    # Railroads
    nearest_railroad_point = nearest_points(specific_point, railroads_union)[1]
    distance_km_railroads = geodesic((specific_point.x, specific_point.y), (nearest_railroad_point.x, nearest_railroad_point.y)).kilometers

    # Major airports
    nearest_airport_point = nearest_points(specific_point, airports_union)[1]
    distance_km_airports = geodesic((specific_point.x, specific_point.y), (nearest_airport_point.x, nearest_airport_point.y)).kilometers

    return distance_km_coast, distance_km_highway, distance_km_railroads, distance_km_airports


denmark_buffered = denmark_gdf.unary_union.buffer(0)
highways_union = highways.unary_union
railroads_union = railroads.unary_union
airports_union = airports.unary_union


df['dist_coast'], df['dist_highway'], df['dist_railroads'], df['dist_airports'] = zip(*df.apply(Distances, axis=1, denmark_buffered=denmark_buffered, highways_union=highways_union, railroads_union=railroads_union, airports_union=airports_union))
df.to_csv("udvidet_1_samlet_med_dist.csv", index=False)

df = pd.read_csv("/work/udvidet_1_samlet_med_dist.csv")


## Schools
schools_gdf = gpd.read_file("Shape/gis_osm_pois_a_free_1.shp")
schools_gdf = schools_gdf.to_crs(epsg=4326)

university = schools_gdf[schools_gdf['fclass'] == 'university']
kindergarden = schools_gdf[schools_gdf['fclass'] == 'kindergarten']
schools = schools_gdf[(schools_gdf['fclass'] == 'school') | (schools_gdf['fclass'] == 'college')]
university['geometry'] = university['geometry'].apply(lambda x: (x.representative_point()))
kindergarden['geometry'] = kindergarden['geometry'].apply(lambda x: (x.representative_point()))
schools['geometry'] = schools['geometry'].apply(lambda x: (x.representative_point()))

## Forests
forests_gdf = gpd.read_file("Shape/gis_osm_landuse_a_free_1.shp")
forests_gdf = forests_gdf.to_crs(epsg=4326)
forests_gdf['geometry'] = forests_gdf['geometry'].apply(lambda x: (x.representative_point()))
forests_gdf = forests_gdf[forests_gdf['fclass'] == "forest"]

## Waterlines
waterlines_gdf = gpd.read_file("Shape/gis_osm_water_a_free_1.shp")
waterlines_gdf = waterlines_gdf.to_crs(epsg=4326)
waterlines_gdf['geometry'] = waterlines_gdf['geometry'].apply(lambda x: (x.representative_point()))
waterlines_gdf = waterlines_gdf[waterlines_gdf['fclass'] == "water"]
#######################################################################################################################################
#######################################################################################################################################


def distance(lat1, lon1, lat2, lon2):
    R = 6371  # radius of the earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x = (lon2 - lon1) * np.cos(0.5 * (lat2 + lat1))
    y = lat2 - lat1
    d = R * np.sqrt(x*x + y*y)
    return d

# Function for calculating distance between a point and a geodataframe vector optimized for speed
def distance_points_vectorized(row, gdf):
    specific_point = (row['latitude'], row['longitude'])
    gdf['distance'] = distance(specific_point[0], specific_point[1], gdf['geometry'].y, gdf['geometry'].x)
    min_distance = np.min(gdf['distance'])
    return min_distance

tqdm.pandas()
df['dist_waterlines'] = df.progress_apply(lambda row: distance_points_vectorized(row, waterlines_gdf), axis=1)
df.to_csv("dist_waterlines.csv", index=False)
tqdm.pandas()
df['dist_forests'] = df.progress_apply(lambda row: distance_points_vectorized(row, forests_gdf), axis=1)
df.to_csv("dist_forests.csv", index=False)
tqdm.pandas()
df['dist_kindergarden'] = df.progress_apply(lambda x: distance_points_vectorized(x, kindergarden), axis=1)
df.to_csv("dist_kindergarden.csv", index=False)
tqdm.pandas()
df['dist_school'] = df.progress_apply(lambda x: distance_points_vectorized(x, schools), axis=1)
df.to_csv("dist_school.csv", index=False)
tqdm.pandas()
df['dist_uni'] = df.progress_apply(lambda x: distance_points_vectorized(x, university), axis=1)
df.to_csv("dist_full.csv", index=False)

df.to_csv("dist_full.csv", index=False)










###### Appendiks

## Alternativ beregning til middelpris
def Distances(p1, p2) -> float:

    """ Calculates the distance (in meters) between p1 and p2, where
    each point is represented as a tuple (lat, lon) """

    return geodesic(p1, p2).meters

def Mean_price(house, df, number=5):
    """
    Function to be applied to hole df, that calculate the mean price of the 5 (default) nearest houses within eh same 'sogn', that share similar
    characteristics like 'Varmeinstallation', 'AntalBad', 'AntalToilet', 'anvendelse', 'grund_str', 'date', 'rooms', 'area'.
    """
    
    house_point = np.array([house['latitude'], house['longitude']])
    house_sogn = house['Sogn']
    house_storkreds = house['Storkreds']
    house_region = house['Region']
    house_varmeinstal = house['Varmeinstallation']
    house_bad = house['AntalBad']
    house_toilet = house['AntalToilet']
    house_rooms = house['rooms']
    house_type = house['anvendelse']
    house_ground_size = house['grund_str'] 
    house_sold_data = house['date']
    house_area = house['area']

    #  date range
    date_range = pd.DateOffset(years=6)


    # Filteres dataframe
    filtered_df = df[
        (df['Sogn'] == house_sogn) &
        (df['Storkreds'] == house_storkreds) &
        (df['Region'] == house['Region']) &
        (df['Varmeinstallation'] == house_varmeinstal) &
        ((df['AntalBad'] >= house_bad - 2) & (df['AntalBad'] <= house_bad + 2)) &
        ((df['AntalToilet'] >= house_toilet - 2) & (df['AntalToilet'] <= house_toilet + 2)) &
        (df['anvendelse'] == house_type) &
        ((df['grund_str'] >= house_ground_size * 0.7) & (df['grund_str'] <= house_ground_size * 1.3)) &
        (df['date'] >= house_sold_data - date_range) & (df['date'] <= house_sold_data + date_range) &
        ((df['rooms'] >= house_rooms - 2) & (df['rooms'] <= house_rooms + 2)) &
        ((df['area'] >= house_area * 0.7) & (df['area'] <= house_area * 1.3))
    ]

    # Calculate distances
    distances = filtered_df.apply(lambda x: Distances(house_point, (x['latitude'], x['longitude'])), axis=1).sort_values(ascending=True)[1:number+1]
    index_distances = distances.index
    mean_prices_nearest = df.iloc[index_distances]['price'].mean()
    return mean_prices_nearest, list(index_distances)