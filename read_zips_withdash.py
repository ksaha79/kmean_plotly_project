import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cartopy.crs as ccrs

data = pd.read_json('./georef-md.json')
zips=pd.DataFrame({"zip_code": data.zip_code.values, "city": data.usps_city.values, "population": data.population.values})
lat=[]
lon=[]
for row in data['geo_point_2d']:
    try:
        lon.append(list(row.values())[0])
        lat.append(list(row.values())[1])
    except:
        lon.append(np.NaN)
        lat.append(np.NaN)
zips['lat']=lat
zips['lon']=lon
zips['ZIP2'] = zips['zip_code'].apply(lambda x: str(x)[3:5])


#def cluster(X):
#    feature = X[cols].to_numpy().reshape((len(X), 1))
#    k_means = KMeans(n_clusters=3).fit(feature)
#    X['cluster'] = k_means.labels_
#    return X

cols=zips.columns[0]
df=zips
#df=df.groupby('ZIP2').apply(cluster)   


kmeans = KMeans(n_clusters = 5, init ='k-means++')
kmeans.fit(df[df.columns[3:5]])
df['cluster']=kmeans.fit_predict(df[df.columns[3:5]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(df[df.columns[3:5]])


import plotly.express as px
#fig=px.scatter_geo(zip_new.zip_code, lat=zip_new['lat'], lon=zip_new['lon'], color=zip_new['cluster'])
fig=px.scatter_geo(df.zip_code, lat=df['lat'], lon=df['lon'], color=df['cluster'], hover_name=df.city, size=df.cluster+1)
fig.update_layout(title = "Zip codes of Maryland", geo_scope='usa')
fig.show()






#colors={'206':'tab:blue', '207':'tab:orange', '208':'tab:green', '209':'tab:red', '210':'tab:purple', '211':'tab:brown', '212':'tab:pink', '214':'tab:yellow', '215':'tab:cyan', '216':'tab:gray', '217':'tab:olive', '218':'magenta', '219':'hotpink'}



