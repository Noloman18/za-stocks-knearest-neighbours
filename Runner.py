import math
import re

import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tabulate import tabulate
import numpy as np

if __name__ == '__main__':
    METRIC_DICTIONARY = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'T': 1000000000000}
    data = pd.read_csv('stock_data.csv', delimiter='|')
    data = data[['name_trans','viewData.symbol','industry_trans','price2bk_us','ttmpr2rev_us', 'eq_market_cap','turnover_volume','eq_revenue','yld5yavg_us','qtotd2eq_us','qcurratio_us','qquickrati_us','margin5yr_us','eq_one_year_return']]
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

    scaler = StandardScaler()
    financial_metric_nbrs = NearestNeighbors(n_neighbors=7, metric='precomputed')
    # company_size_metric_nbrs = NearestNeighbors(n_neighbors=math.floor(len(data_numeric)/10), metric='precomputed')
    company_size_metric_nbrs = NearestNeighbors(n_neighbors=20, metric='precomputed')
    # nbrs.fit(distances)
    # neighbors = nbrs.kneighbors(return_distance=False)
    def get_company_size_distances(data_numeric_param, indexes):
        data_numeric_param = data_numeric_param.iloc[:, indexes]
        data_numeric_param = scaler.fit_transform(data_numeric_param)
        distances = []
        for i in range(len(data_numeric_param)):
            row_distances = []
            for j in range(len(data_numeric_param)):
                if i == j:
                    row_distances.append(0)
                else:
                    dist = euclidean(data_numeric_param[i], data_numeric_param[j])
                    row_distances.append(dist)
            distances.append(row_distances)

        return distances

    def get_neighbours(chosen_model, distances):
        chosen_model.fit(distances)
        return chosen_model.kneighbors(return_distance=False)

    # 'name_trans' 0,'viewData.symbol' 1,'industry_trans' 2,'price2bk_us' 3,'ttmpr2rev_us' 4
    # Print the groups
    def difference(first,second):
        return 100*(second-first)/first if first!= None and first>0 else 0


    repeated_string = ','.join(['neighbour_name,neighbour_pb,neighbour_ps' for i in range(7)])
    output = []
    output.append(f'index,stock_name,pb,ps,median_pb,difference_pb,median_ps,difference_ps,{repeated_string}')

    company_size_neighbours = get_neighbours(company_size_metric_nbrs, get_company_size_distances(data,[5,6,7]))

    for i in range(len(company_size_neighbours)):
        company_size_group = company_size_neighbours[i].tolist()
        my_index = len(company_size_group)
        company_size_group.append(i)
        sub_sample_data = data.iloc[company_size_group,:]
        financial_metric_neigbours = get_neighbours(financial_metric_nbrs, get_company_size_distances(sub_sample_data, [8, 9, 10, 11, 12, 13]))
        group = financial_metric_neigbours[my_index].tolist()
        median_pb = np.median([sub_sample_data.iloc[idx,3] for idx in group])
        median_ps = np.median([sub_sample_data.iloc[idx,4] for idx in group])
        stocks = ','.join([f"{sub_sample_data.iloc[idx, 0]},{sub_sample_data.iloc[idx, 3]},{sub_sample_data.iloc[idx, 4]}" for idx in group])
        output.append(f"{i+1},{sub_sample_data.iloc[my_index, 0]},{sub_sample_data.iloc[my_index, 3]},{sub_sample_data.iloc[my_index, 4]},{median_pb:.2f},{difference(sub_sample_data.iloc[my_index, 3],median_pb):.2f},{median_ps:.2f},{difference(sub_sample_data.iloc[my_index, 4],median_ps):.2f},{stocks}")

    with open('output.csv', 'w', encoding='utf-8') as f:
        for line in output:
            print(line)
            f.write(line + '\n')