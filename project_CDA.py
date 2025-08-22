import pandas as pd
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from sklearn.manifold import Isomap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import scipy.sparse.linalg as ll
import plotly.graph_objs as go
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans

def main():
    tuning = False
    export_final = False
    hostile_analysis = False
    using_spectral = False
    create_2d = True
    
    df = read_data()
    df = clean_data(df)
    df = create_country_dyads(df)

    if hostile_analysis:
        df["intensity"] = df["intensity"] * -1

    grouped_df = group_countries(df)

    mat_dict, countries, indices = create_matrices(grouped_df)
    
    if create_2d:
        create_excel_graph(df, countries)

    if tuning:
        tune_parameters(mat_dict, using_spectral=using_spectral)
    else:
        mat_dict, adj_mat = reduce_dimensions(mat_dict)
        clusters = cluster_matrices(mat_dict, n_clusters=5)
        cluster_scores = score_clusters(clusters, mat_dict, 5, 30)
        export_df(cluster_scores, "final_cluster_scores.csv")
        visualize_graph(clusters, mat_dict, countries)
        find_similar_countries(clusters, indices)

    if export_final:
        export_data_to_excel(mat_dict, clusters, countries)
    
def read_data():
    dir_name = os.path.dirname(__file__)
    dir_name = f"{dir_name}/data"
    file_path = f"{dir_name}/filtered_data.csv"
    
    if not os.path.isfile(file_path):
        print("Files require aggregation and filtering first")
        aggregate_data(dir_name)

    df = pd.read_csv(file_path, sep=",", header=0, encoding="ISO-8859-1")
    return df

def clean_data(df):
    df = clean_col_names(df)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['month'] = df['event_date'].dt.month
    return df

def clean_col_names(df):
    prev_cols = list(df.columns)
    new_cols = [col.replace(" ","_").lower() for col in prev_cols]
    mapping = dict(zip(prev_cols, new_cols))
    df = df.rename(columns=mapping)
    return df

def aggregate_data(dir_name):
    '''Aggregates all into a single filtered file in the project/data directory'''
    print("Aggregating data\n")
    

    years = ["2022"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    full_df = pd.DataFrame(columns=["Event Date",
                                    "Intensity",
                                    "Actor Country",
                                    "Recipient Country"])
    
    for year in years:
        for month in months:
            temp_df = read_raw_data(dir_name, year, month)
            temp_df = temp_df[["Event Date",
                               "Intensity",
                               "Actor Country",
                               "Recipient Country"]]
            
            full_df = pd.concat([full_df, temp_df], ignore_index = True)  

    print("Exporting data\n")
    
    export_df(full_df, "filtered_data.csv")
    print("Export complete!")

def read_raw_data(dir_name, year, month):
    file_name = f"ngecEventsDV-{year}-{month}.txt"
    file_path = f"{dir_name}/{file_name}"
    df = pd.read_csv(file_path, sep="\t", header=0, encoding="ISO-8859-1")
    print(f"{month} {year}")
    return df

def create_country_dyads(df):
    '''
    Creates dyads for each row that has an actor-recipient pair
    
    It will explode the rows that have semi-colon separated country lists
    
    Once exploded, it filters out any rows that have "None" as either an actor
    or a recipient country 
    '''
    df["actor_list"] = df.actor_country.str.split(";")
    df["recipient_list"] = df.recipient_country.str.split(";")
    df = df.explode("actor_list")
    df = df.explode("recipient_list")

    df = df[["month", "intensity", "actor_list", "recipient_list"]]
    df = df.rename(columns={"actor_list":"actor_country",
                            "recipient_list":"recipient_country"})
    
    df = df.query("actor_country != 'None' "
                  + "& recipient_country != 'None'"
                  + "& actor_country != recipient_country")
    
    df["actor_country"] = df["actor_country"].str.strip()
    df["recipient_country"] = df["recipient_country"].str.strip()

    return df

def group_countries(df):
    avg_df = (df
              .groupby(["month", "actor_country", "recipient_country"],
                       as_index=False)
              .mean()
              .rename(columns={"intensity":"intensity_avg"}))
    
    count_df = (df
              .groupby(["month", "actor_country", "recipient_country"],
                       as_index=False)
              .count()
              .rename(columns={"intensity":"intensity_count"}))
    
    merged_df = avg_df.merge(count_df, 
                             on=["month", "actor_country", "recipient_country"])
    merged_df["combined_score"] = merged_df.intensity_count * merged_df.intensity_avg
    # merged_df["combined_score"] = merged_df.intensity_avg

    return merged_df   

def create_index_dict(df):
    '''
    Creates a dictionary where the keys are the country names and the
    values are the indices in the matrix.
    '''
    countries = pd.concat([df.actor_country, df.recipient_country], 
                          ignore_index=True)
    countries = list(np.sort(pd.unique(countries)))
    keys = list(countries)
    values = list(range(len(countries)))
    idx_dict = dict(zip(keys, values))
    ctry_dict = dict(zip(values, keys))

    return idx_dict, ctry_dict

def create_matrices(df):
    '''
    Converts the dataframe to an nxn matrix where n is the number of countries,
    the row indices are the acting countries and the column indices are the
    recipient countries. Also, scales the data from 0.0 to 1.0.
    
    The column used as the value is given by the VAL_COLUMN constant.
    '''
    VAL_COLUMN = "combined_score"    

    month_range = (1,13)
    mat_dict = dict.fromkeys(range(month_range[0], month_range[1]))

    idx_dict, ctry_dict = create_index_dict(df)
    n = len(idx_dict.keys())

    df["actor_index"] = df.actor_country.map(idx_dict)    
    df["recipient_index"] = df.recipient_country.map(idx_dict)
    df[VAL_COLUMN] = ((df[VAL_COLUMN] - df[VAL_COLUMN].min()) 
                      / (df[VAL_COLUMN].max() - df[VAL_COLUMN].min()))
    
    for month in range(month_range[0], month_range[1]):
        temp_df = df.query("month == @month")    
        mat_dict[month] = coo_matrix((temp_df[VAL_COLUMN], 
                            (temp_df.actor_index, temp_df.recipient_index)),
                            shape=(n,n)).toarray()
        mat_dict[month] = mat_dict[month].real
           
    return mat_dict, idx_dict, ctry_dict

def reduce_dimensions(mat_dict, percentile_cut=30):
    adj_dict = dict.fromkeys(list(range(1,13)))
    
    for key, matrix in mat_dict.items():
        mat_dict[key], adj_dict[key] = run_isomap(matrix, percentile_cut)
    
    return mat_dict, adj_dict

def run_isomap(matrix, percentile_cut):
    m = matrix.shape[0]
    
    adj_mat = cdist(matrix, matrix)
    adj_mat = cutoff_distances(adj_mat, percentile_cut)
    
    adj_mat = csr_matrix(adj_mat).toarray()
    dist_mat = shortest_path(adj_mat, directed=True)

    H_mat = np.identity(m) - (1/m) * np.ones((m,m))
    dist_mat = dist_mat ** 2
    C_mat = -0.5 * H_mat @ dist_mat @ H_mat

    N_COMPONENTS = 3
    lambdas, x_mapped = ll.eigs(C_mat, k=N_COMPONENTS)
    # x_mapped = x_mapped @ np.diag(lambdas ** 2)
    x_mapped = x_mapped.real

    return x_mapped, adj_mat

def export_adj_mat(adj_mat):
    dir_name = os.path.dirname(__file__)
    file_path = f"{dir_name}/exports/project_adj_export.csv"
    np.savetxt(file_path, adj_mat, delimiter=",")

def cutoff_distances(adj_mat, n_percentile):
    row_percentile = np.percentile(a=adj_mat, q=n_percentile, axis=1)
    adj_mat[adj_mat > row_percentile.reshape(-1,1)] = 0
    return adj_mat    

def cutoff_network(adj_dict, n_percentile):
    weight_pct = 100 - n_percentile
    for key,matrix in adj_dict.items():
        row_percentile = np.percentile(a=matrix, q=weight_pct)
        adj_dict[key] = matrix[matrix < row_percentile.reshape(-1,1)] = 0
    
    return adj_dict

def cluster_matrices(mat_dict, n_clusters):
    month_range = (1,13)
    clusters = dict.fromkeys(range(month_range[0], month_range[1]))

    for key, matrix in mat_dict.items():
        clusters[key] = SpectralClustering(n_clusters=n_clusters).fit(matrix)
        # train_data = matrix.real
        # clusters[key] = KMeans(n_clusters=n_clusters, init='k-means++').fit(matrix)

    #TODO: Score the model for silhouette score
    return clusters

def visualize_full_year(clusters, matrix, countries):
    
    labels = clusters.labels_
    test_data = matrix

    trace = go.Scatter3d(x=test_data[:,0],
                        y=test_data[:,1],
                        z=test_data[:,2],
                        mode='markers',
                        name='countries',
                        marker=dict(symbol='circle',
                                    size=6,
                                    color=labels,
                                    colorscale='Viridis',
                                    line=dict(color='rgb(50,50,50)',width=0.5)
                                    ),
                        text=list(countries.keys()),
                        hoverinfo='text')
    
    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
        )

    layout = go.Layout(
        title="Countries clustered by how they interact with the world",
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
    margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
        dict(
        showarrow=False,
            text="Data source: <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AJGVIT'>[1] POLECAT Dataset</a>",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )
    
    data=trace
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def visualize_graph(clusters, mat_dict, countries):
    for month in range(1,13):
    
        labels = clusters[month].labels_
        test_data = mat_dict[month]

        trace = go.Scatter3d(x=test_data[:,0],
                            y=test_data[:,1],
                            z=test_data[:,2],
                            mode='markers',
                            name='countries',
                            marker=dict(symbol='circle',
                                        size=6,
                                        color=labels,
                                        colorscale='Viridis',
                                        line=dict(color='rgb(50,50,50)',width=0.5)
                                        ),
                            text=list(countries.keys()),
                            hoverinfo='text')
        
        axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

        layout = go.Layout(
            title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
            dict(
            showarrow=False,
                text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ],    )
        
        data=trace
        fig = go.Figure(data=data, layout=layout)
        fig.show()

def score_clusters(clusters, mat_dict, cluster_num, cut_num):
    sil_scores = []
    
    for month in range(1,13):
        labels = clusters[month].labels_
        sil_scores.append(silhouette_score(X=mat_dict[month],
                                           labels=labels))
    df = pd.DataFrame(zip(list(range(1,13)),
                          sil_scores,
                          [cluster_num]*12,
                          [cut_num]*12),
                          columns=["month","silhouette_score",
                                   "n_clusters","percentile_cut"])
    
    # export_df(df, "cluster_scores.csv")
    return df

def tune_parameters(mat_dict, using_spectral = False):
    percentiles = np.arange(start=30, stop=42, step=3)
    n_clusters = list(range(5,11))

    full_df = pd.DataFrame(columns=["month", "silhoutte_score",
                                    "n_clusters","percentile_cut"])
    
    for pct_val in percentiles:        
        if using_spectral:
            # data_dict = cutoff_network(mat_dict, pct_val)

            for k in n_clusters:
                clusters = run_spectral(mat_dict, n_clusters=k)
                results_df = score_clusters(clusters, mat_dict, k, 100 - pct_val)
                full_df = pd.concat([full_df, results_df], ignore_index=True)
                print(f"Spectral Clusters: {k}, Pct Val: {pct_val}")

        else:
            iso_dict, adj_dict = reduce_dimensions(mat_dict, pct_val)

            for k in n_clusters:         
                clusters = cluster_matrices(iso_dict, k)

                results_df = score_clusters(clusters, iso_dict, k, pct_val)
                full_df = pd.concat([full_df, results_df], ignore_index=True)
                print(f"Clusters: {k}, Pct Val: {pct_val}")

    export_df(full_df, "tuning_results.csv")

def run_spectral(data_dict, n_clusters):
    month_range = (1,13)
    clusters = dict.fromkeys(range(month_range[0], month_range[1]))

    for key, matrix in data_dict.items():
        clusters[key] = SpectralClustering(n_clusters=n_clusters, 
                                           affinity='precomputed').fit(matrix)

    return clusters

def export_df(df, file_name):
    dir_name = os.path.dirname(__file__)
    file_path = f"{dir_name}/outputs/{file_name}"
    df.to_csv(file_path, index=False)

def export_data_to_excel(mat_dict, clusters, countries):
    df = pd.DataFrame(columns=["country", "comp_1", "comp_2",
                               "comp_3", "labels", "month"])
    for month in range(1,13):
        matrix = mat_dict[month]
        labels = clusters[month].labels_

        temp_df = pd.DataFrame(matrix, columns=["comp_1", "comp_2",
                               "comp_3"])
        temp_df["country"] = pd.Series(countries.keys())
        temp_df["labels"] = pd.Series(labels)
        temp_df["month"] = month
        df = pd.concat([df, temp_df], ignore_index=True)
    
    export_df(df, "clustered_dataset.csv")    

def find_similar_countries(clusters, ctry_dict):
    month_range = range(1,13)
    pair_dict = defaultdict(int)

    for month in month_range:
        month_set = clusters[month]
        labels = month_set.labels_
        k_nums = np.unique(labels)

        for k_num in k_nums:
            ctry_idxes = np.argwhere(labels == k_num)
            count_pairs(ctry_idxes, pair_dict, ctry_dict)

    export_similar_countries(pair_dict)
    print("Pairs dictionary exported")
    # return pair_dict    
            
def count_pairs(ctry_indices, pair_dict, ctry_dict):
    ctry_combos = combinations(list(ctry_indices.flatten()), r=2)        
    for combo in ctry_combos:
        pair = f"{ctry_dict[combo[0]]}|{ctry_dict[combo[1]]}"
        pair_dict[pair] += 1

def export_similar_countries(pairs_dict):
    country1_list = []
    country2_list = []
    count_list = []

    for item in pairs_dict.items():
        pair_split = item[0].split("|")
        country1_list.append(pair_split[0])
        country2_list.append(pair_split[1])
        count_list.append(item[1])
    
    df = pd.DataFrame(zip(country1_list, country2_list, count_list),
                      columns=["country_1", "country_2", "pair_count"])
    export_df(df, "pairs.csv")

def create_excel_graph(df, countries):
    avg_df = (df
              .groupby(["actor_country", "recipient_country", "month"],
                       as_index=False)
              .mean()
              .rename(columns={"intensity":"intensity_avg"}))
    
    count_df = (df
              .groupby(["actor_country", "recipient_country", "month"],
                       as_index=False)
              .count()
              .rename(columns={"intensity":"intensity_count"}))
    
    merged_df = avg_df.merge(count_df, 
                             on=["actor_country", "recipient_country", "month"])
    
    # merged_df["intensity_county"] = ((merged_df.intensity_count - merged_df.intensity_count.min()) 
    #                   / (merged_df.intensity_count.max() - merged_df.intensity_count.min()))
    merged_df["combined_score"] = merged_df.intensity_count * merged_df.intensity_avg

    merged_df = (merged_df
                 .groupby(["actor_country", "recipient_country"],
                          as_index=False)
                          .mean())
    
    idx_dict, ctry_dict = create_index_dict(merged_df)
    n = len(idx_dict.keys())

    VAL_COLUMN = "combined_score"

    merged_df["actor_index"] = merged_df.actor_country.map(idx_dict)    
    merged_df["recipient_index"] = merged_df.recipient_country.map(idx_dict)
    merged_df[VAL_COLUMN] = ((merged_df[VAL_COLUMN] - merged_df[VAL_COLUMN].min()) 
                      / (merged_df[VAL_COLUMN].max() - merged_df[VAL_COLUMN].min()))
    
    matrix = coo_matrix((merged_df[VAL_COLUMN], 
                        (merged_df.actor_index, merged_df.recipient_index)),
                        shape=(n,n)).toarray()
    
    x_reduced, adj_mat = run_isomap(matrix, 35)

    # x_reduced = Isomap(n_neighbors=20).fit_transform(matrix)

    clusters = SpectralClustering(n_clusters=5).fit(x_reduced)

    # df_export = pd.DataFrame(x_reduced, columns=["comp_1", "comp_2", "comp_3"])
    # df_export["country"] = pd.Series(countries.keys())
    # df_export["labels"] = pd.Series(clusters.labels_)
    # export_df(df_export, "threedim_yearavg.csv")

    visualize_full_year(clusters, x_reduced, countries)


if __name__ == "__main__":
    main()