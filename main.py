import numpy as np
import pandas as pd
from sklearn import datasets

#%%
if __name__ == '__main__':
    #%%
    # load wine dataset
    wine = datasets.load_wine()
    # convert to dataframe (Just for easy access)
    wine_dataset = pd.DataFrame(wine.data)
    # load wine labels
    wine_labels = wine.target
    print("Head of dataset:\n",wine_dataset.head())

    #%%
    # less use standary scaler on the data
    from sklearn.preprocessing import StandardScaler

    standard = StandardScaler()
    wine_dataset_standar = pd.DataFrame(standard.fit_transform(wine_dataset))
    wine_dataset_standar.head()

    #%%
    # less use standary scaler on the data
    from sklearn.preprocessing import StandardScaler

    standard = StandardScaler()
    wine_dataset_standar = pd.DataFrame(standard.fit_transform(wine_dataset))
    wine_dataset_standar.head()

    #%%

    from som import SOM

    # initialize the SOM
    som = SOM(30, 30)
    # som = SOM(10, 10,alpha_start=0.8)

    #%%
    # fit the SOM for 1000 epochs, save the error every 20 steps

    # som.fit(wine_dataset_standar.to_numpy(), 1000, save_e=True, interval=20)
    # som.fit(wine_dataset_standar.to_numpy(), 700, save_e=True, interval=20)

    # som.save('som_model')

    #%%
    som.load('som_model')
    #%%
    som.plot_error_history()
    #%%
    som.plot_point_map(wine_dataset_standar.to_numpy(), wine_labels, np.unique(wine_labels))
    #%%
    # som.distance_map()
    som.plot_distance_map()
    # print(som.distmap)
    #%%
    np.random.seed(5)
    test=wine_dataset_standar.iloc[1].to_numpy()
    test=test+np.random.rand(test.size)*0.05
    print("Test exaple: ",test)
    som.get_neighbors(test,wine_dataset_standar,wine_dataset[0],wine_labels)