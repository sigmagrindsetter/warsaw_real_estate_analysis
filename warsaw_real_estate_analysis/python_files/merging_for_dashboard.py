import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

pd.set_option('display.max_columns', None)

data_folder = 'data'
all_data = pd.DataFrame()

for file in os.listdir(data_folder):
    parts = file.split('_')
    year_month = parts[-2] + '-' + parts[-1].split('.')[0]
    
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)
    df = df[df['city'].str.lower() == 'warszawa']
    df = df.drop(columns=['city'])
    df['month'] = year_month
    all_data = pd.concat([all_data, df])

shapefile_path = "dzielnice_Warszawy.shp"
gdf = gpd.read_file(shapefile_path)

all_data['geometry'] = all_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
all_data_gdf = gpd.GeoDataFrame(all_data, geometry='geometry', crs='EPSG:4326')
gdf = gdf.to_crs(all_data_gdf.crs)
all_data_with_districts = gpd.sjoin(all_data_gdf, gdf[['geometry', 'nazwa_dzie']], how='left', predicate='within')
all_data_with_districts = all_data_with_districts.rename(columns={'nazwa_dzie': 'district'})
all_data_with_districts['district'] = all_data_with_districts['district'].fillna('suburbs')
all_data_with_districts = all_data_with_districts.drop(columns=['ownership', 'index_right', 'geometry'])

print(all_data_with_districts.head())

all_data_with_districts.to_csv('rental_data_warsaw.csv', index=False)