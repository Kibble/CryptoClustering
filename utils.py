import pandas as pd
from sklearn.cluster import KMeans

def create_elbow_df(df, num_k):
    inertia = []
    k = list(range(1, num_k + 1))
    for i in k:
        m = KMeans(n_clusters=i, n_init='auto')
        m.fit(df)
        inertia.append(m.inertia_)
    
    elbow_df = pd.DataFrame({
        'k': k,
        'inertia': inertia
    })

    return elbow_df
