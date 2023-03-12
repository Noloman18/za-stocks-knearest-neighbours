import re

import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tabulate import tabulate

if __name__ == '__main__':
    METRIC_DICTIONARY = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'T': 1000000000000}
    data = pd.read_csv('stock_data.csv', delimiter='|')
    data = data[['name_trans','viewData.symbol','industry_trans','price2bk_us','ttmpr2rev_us', 'eq_market_cap','turnover_volume','eq_revenue','yld5yavg_us','qtotd2eq_us','qcurratio_us','qquickrati_us','margin5yr_us','eq_one_year_return']]
    print(tabulate(data.head(10),headers='keys', tablefmt='psql'))
    data['yld5yavg_us'] = data['yld5yavg_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))) if x != None and x!= '' and x!='-' else 0)
    data['margin5yr_us'] = data['margin5yr_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))) if x != None and x!= '' and x!='-' else 0)
    data['price2bk_us'] = data['price2bk_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))) if x != None and x!= '' and x!='-' else None)
    data['ttmpr2rev_us'] = data['ttmpr2rev_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))* METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['qtotd2eq_us'] = data['qtotd2eq_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))* METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['qcurratio_us'] = data['qcurratio_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))* METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['qquickrati_us'] = data['qquickrati_us'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))* METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['eq_one_year_return'] = data['eq_one_year_return'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))* METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['eq_market_cap'] = data['eq_market_cap'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x)) * METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['turnover_volume'] = data['turnover_volume'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))*METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data['eq_revenue'] = data['eq_revenue'].apply(lambda x: (float(re.sub(r'[^\d.]', '', x))*METRIC_DICTIONARY.get(x[-1].upper(), 1)) if x != None and x!= '' and x!='-' else None)
    data.dropna(inplace=True)
    data_numeric = data.iloc[:, 5:]

    # Normalize the data using z-scores
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data_numeric)

    # Calculate pairwise distances between stocks
    distances = []
    for i in range(len(data_norm)):
        row_distances = []
        for j in range(len(data_norm)):
            if i == j:
                row_distances.append(0)
            else:
                dist = euclidean(data_norm[i], data_norm[j])
                row_distances.append(dist)
        distances.append(row_distances)

    # Set k to 5
    k = 5

    # Find the k nearest neighbors for each stock
    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed')
    nbrs.fit(distances)
    neighbors = nbrs.kneighbors(return_distance=False)

    # 'name_trans' 0,'viewData.symbol' 1,'industry_trans' 2,'price2bk_us' 3,'ttmpr2rev_us' 4
    # Print the groups
    for i in range(len(neighbors)):
        group = neighbors[i].tolist()
        group.append(i)
        stocks = [f"(name:{data.iloc[idx, 0]}, PB:{data.iloc[idx, 3]},PS:{data.iloc[idx, 4]})" for idx in group]
        print(f"(name:{data.iloc[i, 0]}, PB:{data.iloc[i, 3]},PS:{data.iloc[i, 4]}) :=> {stocks}")